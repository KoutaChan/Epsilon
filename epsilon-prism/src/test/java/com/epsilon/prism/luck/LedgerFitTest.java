package com.epsilon.prism.luck;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.noise.LedgerCells;
import com.epsilon.prism.noise.LedgerGroup;
import com.epsilon.prism.rows.Row;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 運の精算の係数のクロスフィット（{@link LedgerFit}）のテスト。確かめるのは次のこと。植えた係数を取り戻せる。ホールドアウトの R² が
 * 理論値に合う。偶奇 g の行に掛ける β は、偶奇 g の行に依らない。行の少ない判断のクラスは、まとめた当てはめを使う。帳簿の無い行と欠けた
 * 行は、そのまま通る。ridge は特徴の大きさに依らない。報告の組の R² は帳簿の 5 組ごとにあり、その組だけで当てはめた R² と同じ。係数が
 * 付くのは補正に使う組だけ。区分に要る行の数と ridge は、当てはめの値から決まる。局・順位の区分は、どのルールでも判断時の局と点と
 * 絶対席の順で決まる。行を貯めて当てはめても係数と報告はビットまで同じで、貯めた行は足した順に元の行と同じ値と局面を持つ。
 */
public class LedgerFitTest {
  private static final int F = LedgerColumn.COUNT;

  /** 古典の 5 組を全て補正に使い、同梱の設定の既定値（ridge 0.001、列 1 つあたり 32 行）で当てはめる。 */
  public static final LedgerFit CLASSIC = new LedgerFit(LedgerGroup.computed(), 0.001, 32);

  /**
   * テストで使う行。帳簿の無い行は {@code ledger} が null。
   *
   * @param ledger 手ごとの帳簿（長さは 合法手の数 × F）
   * @param standing 判断時の局面（補正率のテストのほかは東 1 局の首位・点差 0）
   * @param played 打った手の位置（補正率の監査のテストのほかは 0）
   * @param source 行の種類（補正率の監査のテストのほかは MAIN）
   */
  record SyntheticRow(
      long gameId,
      DecisionClass decisionClass,
      float[] values,
      float[] ledger,
      int cell,
      LedgerCell.Standing standing,
      int played,
      Row.Source source)
      implements FitRow {
    SyntheticRow(
        long gameId,
        DecisionClass decisionClass,
        float[] values,
        float[] ledger,
        int cell,
        LedgerCell.Standing standing) {
      this(gameId, decisionClass, values, ledger, cell, standing, 0, Row.Source.MAIN);
    }

    SyntheticRow(
        long gameId, DecisionClass decisionClass, float[] values, float[] ledger, int cell) {
      this(gameId, decisionClass, values, ledger, cell, new LedgerCell.Standing(0, 0, 0));
    }

    SyntheticRow(long gameId, DecisionClass decisionClass, float[] values, float[] ledger) {
      this(gameId, decisionClass, values, ledger, 0);
    }

    @Override
    public int ledgerFeatures() {
      return ledger == null ? 0 : ledger.length / values.length;
    }

    @Override
    public float ledger(int action, int column) {
      return ledger == null ? Float.NaN : ledger[action * ledgerFeatures() + column];
    }

    int actions() {
      return values.length;
    }
  }

  private static List<SyntheticRow> cellRows(int rareOdd, int rareEven) {
    var rows = new ArrayList<SyntheticRow>();
    float[] ledger = new float[2 * F];
    ledger[0] = 1;
    ledger[F] = -1;
    for (int parity = 0; parity < 2; parity++) {
      int rare = parity == 0 ? rareEven : rareOdd;
      for (int i = 0; i < rare + 530; i++) {
        int cell = i < rare ? 11 : 0;
        float slope = (cell == 11 ? 3 : -1) + parity;
        rows.add(
            new SyntheticRow(
                2L * i + parity, DecisionClass.CALL, new float[] {slope, -slope}, ledger, cell));
      }
    }
    return rows;
  }

  /** 区分の番号は 段階 × 席の数 ＋ 順位で、段階の区切りは場風の境と南場の最後の局。順位は判断時の点と起家順の同点処理で決まる。 */
  @Test
  public void cellsFollowTheRoundStageAndTheAbsoluteSeatRank() {
    for (Variant variant : Variant.values()) {
      LedgerCell cells = LedgerCell.of(variant);
      int seats = variant.seats();
      Assert.assertEquals(cells.count(), 3 * seats);
      var writer = new ObservationWriter(variant);
      var batch = new ObservationBatch(variant, 1 << 12);
      boolean[] stages = new boolean[3];
      boolean[] tiedRanks = new boolean[seats];
      new TableWalk(variant, TableWalk.Policy.GREEDY)
          .run(
              23,
              3,
              (engine, slot) -> {
                int seat = engine.decisions().seat(slot);
                var view = engine.full().seat(seat);
                int[] scores = new int[seats];
                for (int s = 0; s < seats; s++) scores[s] = view.score(s);
                int rank = Placement.rank(scores, seat);
                int stage = view.round() >= variant.lastRound() ? 2 : view.round() >= seats ? 1 : 0;
                batch.clear();
                int index = writer.capture(engine.view(seat), engine.decisions(), slot, batch);
                Assert.assertEquals(
                    cells.of(batch.words(), batch.start(index)),
                    stage * seats + rank,
                    variant.name());
                stages[stage] = true;
                if (Arrays.stream(scores).allMatch(score -> score == scores[0]))
                  tiedRanks[rank] = true;
              });
      for (boolean seen : stages) Assert.assertTrue(seen, variant.name());
      for (boolean seen : tiedRanks) Assert.assertTrue(seen, variant.name());
    }
    Assert.assertEquals(
        LedgerCell.of(Variant.TENHOU_4P).name(DecisionClass.CALL, 11), "CALL_SOUTH_4_PLUS_4");
    Assert.assertEquals(
        LedgerCell.of(Variant.TENHOU_4P).name(DecisionClass.CALL, 4), "CALL_SOUTH_1_3_1");
    Assert.assertEquals(
        LedgerCell.of(Variant.TENHOU_3P).name(DecisionClass.CALL, 8), "CALL_SOUTH_3_PLUS_3");
    Assert.assertEquals(
        LedgerCell.of(Variant.TENHOU_3P).name(DecisionClass.CALL, 0), "CALL_EAST_1");
  }

  @Test
  public void sparseCellsReproduceTheClassFitBitForBit() {
    var rows = cellRows(543, 543);
    var baseline = CLASSIC.fitRows(rows);
    var cells =
        new LedgerFit(
                LedgerGroup.computed(), 0.001, 32, LedgerCells.CLASS_STAGE_RANK, Variant.TENHOU_4P)
            .fitRows(rows);
    Assert.assertTrue(cells.report().byCell().isEmpty());
    Assert.assertEquals(cells.report().all().r2(), baseline.report().all().r2());
    Assert.assertEquals(cells.report().all().r2ByGroup(), baseline.report().all().r2ByGroup());
    for (var row : rows) {
      Assert.assertTrue(
          Arrays.equals(cells.correction().values(row), baseline.correction().values(row)));
    }
  }

  @Test
  public void cellThresholdUsesOnlyTheOppositeParityAndSurvivesStreaming() {
    var rows = cellRows(544, 543);
    var fitter =
        new LedgerFit(
            LedgerGroup.computed(), 0.001, 32, LedgerCells.CLASS_STAGE_RANK, Variant.TENHOU_4P);
    var result = fitter.fitRows(rows);
    var tap = fitter.tap(2, index -> true);
    for (int i = 0; i < rows.size(); i++) tap.add(i < rows.size() / 2 ? 0 : 1, rows.get(i));
    var streamed = tap.fit();
    Assert.assertEquals(streamed.report().byCell(), result.report().byCell());
    Assert.assertEquals(streamed.report().all().r2(), result.report().all().r2());
    var folds = result.report().byCell().get("CALL_SOUTH_4_PLUS_4");
    Assert.assertFalse(folds.get(0).fallback());
    Assert.assertTrue(folds.get(1).fallback());
    int c = DecisionClass.CALL.ordinal();
    Assert.assertEquals(result.correction().coefficients(0, c, 11)[0], 4 / 1.001, 1e-12);
    Assert.assertSame(result.correction().coefficients(1, c, 11), result.correction().beta()[1][c]);
    double raw = 0, residual = 0;
    for (var row : rows) {
      double y = row.values()[0];
      double corrected = result.correction().values(row)[0];
      residual += corrected * corrected;
      raw += y * y;
    }
    Assert.assertEquals(result.report().all().r2(), 1 - residual / raw, 1e-7);
  }

  /** テスト用に作る行。3 手を測り、帳簿の特徴 f は scale[f]·N(0, 1)、Y_a = Q_a + Σ_f beta[f]·L_af + N(0, noise²)。 */
  static SyntheticRow row(
      long gameId,
      DecisionClass decisionClass,
      double[] scale,
      double[] beta,
      double noise,
      SplittableRandom random) {
    int n = 4;
    float[] values = new float[n], ledger = new float[n * F];
    Arrays.fill(values, Float.NaN);
    Arrays.fill(ledger, Float.NaN);
    for (int a = 0; a < 3; a++) {
      double y = 0.1 * a + noise * random.nextGaussian();
      for (int f = 0; f < F; f++) {
        double l = scale[f] * random.nextGaussian();
        ledger[a * F + f] = (float) l;
        y += beta[f] * (float) l;
      }
      values[a] = (float) y;
    }
    return new SyntheticRow(gameId, decisionClass, values, ledger);
  }

  static double[] ones() {
    double[] scale = new double[F];
    Arrays.fill(scale, 1);
    return scale;
  }

  /** 植えた係数（特徴 0・3・15 だけが 0 でない）。 */
  static double[] planted() {
    double[] beta = new double[F];
    beta[0] = 0.3;
    beta[3] = -0.2;
    beta[15] = 0.5;
    return beta;
  }

  static List<SyntheticRow> rows(
      int count, long firstGame, double[] scale, double[] beta, double noise, long seed) {
    var random = new SplittableRandom(seed);
    List<SyntheticRow> rows = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      // 判断のクラスと偶奇（半荘 ID の下位ビット）は互いに独立
      DecisionClass c = i % 3 == 0 ? DecisionClass.CALL : DecisionClass.DISCARD_MID;
      rows.add(row(firstGame + i, c, scale, beta, noise, random));
    }
    return rows;
  }

  /**
   * 植えた係数を取り戻す。判断のクラスは 2 つ、1 行に 3 手、L は平均 0 の正規分布、Y = Q + β*·L + ε（Q_a = 0.1·a）。このとき β_{g,c} ≈ β*
   * （相対 5%）、ホールドアウトの R² は理論値 ± 0.02 で、補正した Y の平均と Y の平均の差は ≈ 0 になる。理論値は、行の中で中心化した分散の比
   * (2/3)|β*|² / ((2/3)(|β*|² + σ²) + mean(Q̃²))。
   */
  @Test
  public void recoversAPlantedCoefficient() {
    double[] beta = planted();
    double noise = 0.2;
    List<SyntheticRow> rows = rows(20_000, 0, ones(), beta, noise, 1);
    LedgerFit.Result fit = CLASSIC.fitRows(rows);
    LedgerCorrection correction = fit.correction();
    for (int g = 0; g < 2; g++)
      for (DecisionClass c : new DecisionClass[] {DecisionClass.DISCARD_MID, DecisionClass.CALL})
        for (int f : new int[] {0, 3, 15}) {
          double got = correction.beta()[g][c.ordinal()][f];
          Assert.assertEquals(
              got, beta[f], 0.05 * Math.abs(beta[f]), "fold " + g + " " + c + " feature " + f);
        }
    double signal = 0;
    for (double b : beta) signal += b * b;
    // 中心化すると L と ε の分散は 2/3 倍になる。Q̃ = (−0.1, 0, 0.1) の 2 乗の平均 0.02/3 は L では説明できない
    double theory = (2.0 / 3) * signal / ((2.0 / 3) * (signal + noise * noise) + 0.02 / 3);
    Assert.assertEquals(fit.report().all().r2(), theory, 0.02);
    Assert.assertEquals(fit.report().rows(), rows.size());
    double raw = 0, corrected = 0;
    long count = 0;
    for (SyntheticRow row : rows) {
      float[] fixed = correction.values(row);
      for (int a = 0; a < row.actions(); a++)
        if (!Float.isNaN(row.values()[a])) {
          raw += row.values()[a];
          corrected += fixed[a];
          count++;
        } else Assert.assertTrue(Float.isNaN(fixed[a]));
    }
    Assert.assertEquals((corrected - raw) / count, 0, 0.02);
    for (LedgerReport.FeatureBias bias : fit.report().bias().all())
      Assert.assertTrue(Math.abs(bias.z()) < 4.5, bias.name() + " z=" + bias.z());
    Assert.assertTrue(fit.report().all().equivalentForks() > 1);
  }

  /** 偶数の半荘の行の Y だけを変えても、偶数の行に掛ける β（奇数の行で当てはめたもの）は変わらない。 */
  @Test
  public void crossFitNeverUsesTheRowsOwnParity() {
    List<SyntheticRow> rows = rows(4000, 0, ones(), planted(), 0.5, 2);
    List<SyntheticRow> shifted = new ArrayList<>();
    for (SyntheticRow row : rows) {
      if ((row.gameId() & 1) != 0) {
        shifted.add(row);
        continue;
      }
      float[] values = row.values().clone();
      for (int a = 0; a < values.length; a++)
        if (!Float.isNaN(values[a])) values[a] = values[a] * 3 + 1;
      shifted.add(withValues(row, values, row.ledger()));
    }
    double[][][] before = CLASSIC.fitRows(rows).correction().beta();
    double[][][] after = CLASSIC.fitRows(shifted).correction().beta();
    for (int c = 0; c < DecisionClass.COUNT; c++)
      Assert.assertTrue(Arrays.equals(after[0][c], before[0][c]), "class " + c);
    Assert.assertFalse(
        Arrays.equals(
            after[1][DecisionClass.CALL.ordinal()], before[1][DecisionClass.CALL.ordinal()]));
  }

  /** 行の少ない判断のクラス（偶奇ごとに、区分に要る行の数に足りない）は、もう一方の偶奇の全てのクラスをまとめた当てはめを使う。全体でも足りなければ係数は 0。 */
  @Test
  public void sparseClassesFallBackToThePooledFit() {
    var random = new SplittableRandom(4);
    List<SyntheticRow> rows = new ArrayList<>(rows(6000, 0, ones(), planted(), 0.5, 4));
    for (int i = 0; i < 100; i++)
      rows.add(row(100_000 + i, DecisionClass.KAN, ones(), planted(), 0.5, random));
    LedgerFit.Result fit = CLASSIC.fitRows(rows);
    LedgerReport.Line kan = fit.report().byClass().get(DecisionClass.KAN.name());
    Assert.assertTrue(kan.fallback()[0] && kan.fallback()[1]);
    Assert.assertFalse(fit.report().byClass().get(DecisionClass.CALL.name()).fallback()[0]);
    double[][][] beta = fit.correction().beta();
    Assert.assertTrue(
        Arrays.equals(beta[0][DecisionClass.KAN.ordinal()], fit.report().all().betaByFold()[0]));
    Assert.assertTrue(
        Arrays.equals(beta[1][DecisionClass.KAN.ordinal()], fit.report().all().betaByFold()[1]));
    // 全体でも足りなければ係数は 0
    LedgerFit.Result tiny = CLASSIC.fitRows(rows.subList(0, 50));
    for (double value : tiny.correction().beta()[0][DecisionClass.CALL.ordinal()])
      Assert.assertEquals(value, 0.0);
  }

  /** 帳簿の無い行と、測った手の帳簿に NaN を含む行は、値を変えずに返す（当てはめにも入れず、{@code skippedRows} に数える）。 */
  @Test
  public void rowsWithoutLedgerPassThrough() {
    List<SyntheticRow> rows = new ArrayList<>(rows(3000, 0, ones(), planted(), 0.5, 5));
    SyntheticRow bare = withValues(rows.get(10), rows.get(10).values(), null);
    float[] broken = rows.get(11).ledger().clone();
    broken[F + 2] = Float.NaN;
    SyntheticRow partial = withValues(rows.get(11), rows.get(11).values(), broken);
    rows.add(bare);
    rows.add(partial);
    LedgerFit.Result fit = CLASSIC.fitRows(rows);
    Assert.assertEquals(fit.report().skippedRows(), 2);
    Assert.assertSame(fit.correction().values(bare), bare.values());
    Assert.assertSame(fit.correction().values(partial), partial.values());
    Assert.assertNotSame(fit.correction().values(rows.get(0)), rows.get(0).values());
  }

  /**
   * ridge の強さは特徴ごとの対角成分に比例するので、1 つの特徴を 10^4 倍しても補正した Y は変わらない（相対 1e-5）。方策の項くらいの
   * 大きさ（0.05）の特徴に植えた係数は、点数の項くらいの大きさ（3）の特徴と並べても、縮まずに取り戻せる。
   */
  @Test
  public void ridgeIsScaleInvariant() {
    double[] scale = ones();
    double[] beta = planted();
    List<SyntheticRow> rows = rows(4000, 0, scale, beta, 0.5, 6);
    List<SyntheticRow> scaled = new ArrayList<>();
    for (SyntheticRow row : rows) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++)
        if (!Float.isNaN(row.values()[a])) ledger[a * F + 7] *= 10_000f;
      scaled.add(withValues(row, row.values(), ledger));
    }
    LedgerCorrection plain = CLASSIC.fitRows(rows).correction();
    LedgerCorrection wide = CLASSIC.fitRows(scaled).correction();
    for (int i = 0; i < 200; i++) {
      float[] a = plain.values(rows.get(i)), b = wide.values(scaled.get(i));
      for (int k = 0; k < a.length; k++)
        if (!Float.isNaN(a[k]))
          Assert.assertEquals(b[k], a[k], 1e-5 * Math.max(1, Math.abs(a[k])), "row " + i);
    }
    // 小さい特徴（0.05）と大きい特徴（3）を並べる
    double[] mixed = new double[F];
    Arrays.fill(mixed, 3);
    mixed[15] = 0.05;
    double[] coefficients = new double[F];
    coefficients[15] = 4.0;
    coefficients[3] = 0.05;
    LedgerCorrection fit =
        CLASSIC.fitRows(rows(20_000, 0, mixed, coefficients, 0.2, 7)).correction();
    for (int g = 0; g < 2; g++)
      Assert.assertEquals(fit.beta()[g][DecisionClass.CALL.ordinal()][15], 4.0, 0.2, "fold " + g);
  }

  /**
   * ライターごとに書いた順に足す {@link LedgerFit.WriterTap} の結果は、シャード 0、1、2 の順に並べた行を {@link LedgerFit#fitRows} に
   * 渡した結果とビット単位で一致する（シャードをまたいでどの順に呼んでも変わらない）。帳簿は fp16 に丸め、行ごとに一部の特徴を全ての手で 0 に
   * したり（中心化しても 0）、1 手だけ 0 にしたりする。飛ばす行（帳簿の無い行、測った手が 1 つの行）も混ぜる。
   */
  @Test
  public void streamMatchesTheReadBackOrderBitwise() {
    int shards = 3;
    List<SyntheticRow> written = new ArrayList<>();
    for (SyntheticRow row : rows(6000, 0, ones(), planted(), 0.3, 11)) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++)
        for (int f = 0; f < F; f++) {
          int k = a * F + f;
          if (Float.isNaN(ledger[k])) continue;
          boolean zero = (row.gameId() + f) % 3 == 0 || (a == 1 && f % 5 == 0);
          ledger[k] = zero ? 0f : Float.float16ToFloat(Float.floatToFloat16(ledger[k]));
        }
      written.add(withValues(row, row.values(), ledger));
    }
    written.set(17, withValues(written.get(17), written.get(17).values(), null));
    float[] single = written.get(29).values().clone();
    single[1] = single[2] = Float.NaN;
    float[] singleLedger = written.get(29).ledger().clone();
    Arrays.fill(singleLedger, F, 3 * F, Float.NaN);
    written.set(29, withValues(written.get(29), single, singleLedger));
    List<List<SyntheticRow>> byShard = new ArrayList<>();
    for (int shard = 0; shard < shards; shard++) byShard.add(new ArrayList<>());
    for (int i = 0; i < written.size(); i++) byShard.get(i % shards).add(written.get(i));
    List<SyntheticRow> readBack = new ArrayList<>();
    for (List<SyntheticRow> shard : byShard) readBack.addAll(shard);
    var stream = CLASSIC.tap(shards, index -> true);
    for (int shard : new int[] {2, 0, 1})
      for (SyntheticRow row : byShard.get(shard)) stream.add(shard, row);
    LedgerFit.Result streamed = stream.fit();
    Assert.assertEquals(streamed.report().skippedRows(), 2);
    assertBitwiseEqual(streamed, CLASSIC.fitRows(readBack));
  }

  /**
   * 報告には、帳簿の 5 組（IMPROVE・POINTS・DORA・URA・POLICY）の R² を宣言の順に載せる。どの組の R² も、全体でもクラスごとでも、
   * その組だけを補正に使って当てはめた R² と同じ（違うのは残差を足す順だけなので、差は 1e-12 まで）。1 つの組の列にだけ係数を植えると、
   * その組の R² は全体の R² に近く、ほかの組の R² は 0 に近い。組が列を 1 つでも取りこぼすか、ほかの組の列を含むと、どちらかの確認が外れる。
   */
  @Test
  public void reportGroupsAreTheFiveLedgerGroups() {
    List<LedgerGroup> groups = List.copyOf(LedgerGroup.computed());
    Assert.assertEquals(
        groups,
        List.of(
            LedgerGroup.IMPROVE,
            LedgerGroup.POINTS,
            LedgerGroup.DORA,
            LedgerGroup.URA,
            LedgerGroup.POLICY));
    for (LedgerGroup planted : groups) {
      double[] beta = new double[F];
      for (LedgerColumn column : LedgerColumn.values())
        if (column.group() == planted) beta[column.ordinal()] = 0.3;
      List<SyntheticRow> rows = rows(6000, 0, ones(), beta, 0.2, 20 + planted.ordinal());
      LedgerReport report = CLASSIC.fitRows(rows).report();
      LedgerReport.Line all = report.all();
      Assert.assertEquals(List.copyOf(all.r2ByGroup().keySet()), groups, planted + " planted");
      for (LedgerGroup group : groups) {
        LedgerReport alone = new LedgerFit(Set.of(group), 0.001, 32).fitRows(rows).report();
        double r2 = all.r2ByGroup().get(group);
        Assert.assertEquals(r2, alone.all().r2(), 1e-12, planted + " planted, " + group + " alone");
        for (Map.Entry<String, LedgerReport.Line> line : report.byClass().entrySet()) {
          Assert.assertEquals(
              List.copyOf(line.getValue().r2ByGroup().keySet()), groups, line.getKey());
          Assert.assertEquals(
              line.getValue().r2ByGroup().get(group),
              alone.byClass().get(line.getKey()).r2(),
              1e-12,
              planted + " planted, " + group + " alone in " + line.getKey());
        }
        if (group == planted)
          Assert.assertEquals(r2, all.r2(), 0.005, planted + " planted, " + group);
        else Assert.assertTrue(r2 < 0.005, planted + " planted, " + group + " r2=" + r2);
      }
      Assert.assertTrue(all.r2() > 0.5, planted + " r2=" + all.r2());
    }
  }

  /**
   * 係数が付くのは補正に使う組（POINTS と POLICY）の列だけで、ほかの列の係数は、全ての偶奇・クラスの補正でも、報告の全体・偶奇・クラスでも 0。
   * 係数は、ほかの列を全ての手で 0 にした行を古典の 5 組で当てはめた係数とビット単位で一致する。補正した Y は、手で計算した
   * Y_a − Σ_f β_f·L_af（補正に使う列だけを double で引き、float に丸める）とビット単位で一致する。
   */
  @Test
  public void onlyTheAppliedGroupsCarryCoefficients() {
    Set<LedgerGroup> apply = EnumSet.of(LedgerGroup.POINTS, LedgerGroup.POLICY);
    boolean[] applied = new boolean[F];
    for (LedgerColumn column : LedgerColumn.values())
      applied[column.ordinal()] = apply.contains(column.group());
    double[] beta = new double[F];
    Arrays.fill(beta, 0.1);
    List<SyntheticRow> rows = rows(6000, 0, ones(), beta, 0.3, 30);
    LedgerFit.Result fit = new LedgerFit(apply, 0.001, 32).fitRows(rows);
    double[][][] coefficients = fit.correction().beta();
    for (int g = 0; g < 2; g++)
      for (int c = 0; c < DecisionClass.COUNT; c++)
        for (int f = 0; f < F; f++)
          if (!applied[f])
            Assert.assertEquals(
                coefficients[g][c][f], 0.0, 0.0, "fold " + g + " class " + c + " column " + f);
    for (DecisionClass c : new DecisionClass[] {DecisionClass.CALL, DecisionClass.DISCARD_MID})
      for (int f = 0; f < F; f++)
        if (applied[f]) Assert.assertTrue(coefficients[0][c.ordinal()][f] != 0, c + " column " + f);
    List<LedgerReport.Line> lines = new ArrayList<>(fit.report().byClass().values());
    lines.add(fit.report().all());
    for (LedgerReport.Line line : lines)
      for (int f = 0; f < F; f++) {
        if (applied[f]) continue;
        Assert.assertEquals(line.beta()[f], 0.0, 0.0, "column " + f);
        for (double[] fold : line.betaByFold())
          Assert.assertEquals(fold[f], 0.0, 0.0, "column " + f);
      }

    List<SyntheticRow> zeroed = new ArrayList<>();
    for (SyntheticRow row : rows) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++)
        if (!Float.isNaN(row.values()[a]))
          for (int f = 0; f < F; f++) if (!applied[f]) ledger[a * F + f] = 0f;
      zeroed.add(withValues(row, row.values(), ledger));
    }
    double[][][] reference = CLASSIC.fitRows(zeroed).correction().beta();
    for (int g = 0; g < 2; g++)
      for (int c = 0; c < DecisionClass.COUNT; c++)
        for (int f = 0; f < F; f++)
          Assert.assertEquals(
              Double.doubleToRawLongBits(coefficients[g][c][f]),
              Double.doubleToRawLongBits(reference[g][c][f]),
              "fold " + g + " class " + c + " column " + f);

    int changed = 0;
    for (SyntheticRow row : rows) {
      float[] corrected = fit.correction().values(row);
      double[] b = coefficients[(int) (row.gameId() & 1)][row.decisionClass().ordinal()];
      for (int a = 0; a < row.actions(); a++) {
        float y = row.values()[a];
        if (Float.isNaN(y)) {
          Assert.assertTrue(Float.isNaN(corrected[a]));
          continue;
        }
        double expected = y;
        for (int f = 0; f < F; f++) if (applied[f]) expected -= b[f] * row.ledger(a, f);
        Assert.assertEquals(
            Float.floatToRawIntBits(corrected[a]),
            Float.floatToRawIntBits((float) expected),
            "row " + row.gameId());
        if (corrected[a] != y) changed++;
      }
    }
    Assert.assertTrue(changed > 0, "the correction changed nothing");
  }

  /**
   * 区分に要る行の数は、列 1 つあたりの数 × 当てはめる列の数。偶奇ごとに 50 行ある KAN のクラスは、URA（1 列）なら、列 1 つあたり 50 行で
   * そのクラスだけで当てはめ、51 行ならまとめた当てはめを使う。POLICY（2 列）なら 25 行と 26 行で分かれる。古典の 5 組（17 列）は 32 行でも
   * まとめた当てはめを使う。
   */
  @Test
  public void rowsPerCellScaleWithTheFittedColumns() {
    var random = new SplittableRandom(31);
    List<SyntheticRow> rows = new ArrayList<>(rows(6000, 0, ones(), planted(), 0.5, 31));
    for (int i = 0; i < 100; i++)
      rows.add(row(100_000 + i, DecisionClass.KAN, ones(), planted(), 0.5, random));
    Object[][] cases = {
      {LedgerGroup.URA, 50, false},
      {LedgerGroup.URA, 51, true},
      {LedgerGroup.POLICY, 25, false},
      {LedgerGroup.POLICY, 26, true}
    };
    for (Object[] c : cases) {
      LedgerFit fit = new LedgerFit(Set.of((LedgerGroup) c[0]), 0.001, (int) c[1]);
      LedgerReport.Line kan = fit.fitRows(rows).report().byClass().get(DecisionClass.KAN.name());
      Assert.assertEquals(kan.rows(), 100L);
      for (int fold = 0; fold < 2; fold++)
        Assert.assertEquals(kan.fallback()[fold], (boolean) c[2], c[0] + " per column " + c[1]);
    }
    LedgerReport.Line classic =
        CLASSIC.fitRows(rows).report().byClass().get(DecisionClass.KAN.name());
    Assert.assertTrue(classic.fallback()[0] && classic.fallback()[1]);
  }

  /**
   * ridge は当てはめの値から読む。1 列だけを当てはめるときは、相関行列の形にした対角に ridge を足すので β = b / (A·(1 + ridge)) になる。
   * よって ridge 1 の β·2 と、ridge 0.001 の β·1.001 は同じ値になる。
   */
  @Test
  public void ridgeComesFromTheFit() {
    List<SyntheticRow> rows = rows(6000, 0, ones(), planted(), 0.5, 32);
    int ura = LedgerColumn.URA.ordinal();
    double strong =
        new LedgerFit(Set.of(LedgerGroup.URA), 1.0, 32).fitRows(rows).report().all().beta()[ura];
    double weak =
        new LedgerFit(Set.of(LedgerGroup.URA), 0.001, 32).fitRows(rows).report().all().beta()[ura];
    Assert.assertTrue(strong != weak, "ridge 1 and 0.001 gave the same coefficient " + weak);
    Assert.assertEquals(strong * 2.0, weak * 1.001, 1e-12 * Math.abs(weak));
  }

  /**
   * 行を貯める受け取り先は、貯めない受け取り先と同じ行を同じ順に足すので、係数と報告はビットまで同じ。貯めた行は足した順（シャード 0、1、2 の
   * 順、シャードの中は書いた順）に、測った手の Y・帳簿・区分・局面を元の行と同じ値で持ち、飛ばした行は持たない。貯めた行は 1 度だけ
   * 受け取れ、貯めない受け取り先からは受け取れない。
   */
  @Test
  public void aRetainingStreamKeepsTheFitAndTheRows() {
    Variant variant = Variant.TENHOU_4P;
    List<SyntheticRow> written =
        new ArrayList<>(LedgerCalibratorTest.scaledRows(variant, 3000, 0, 13));
    written.set(5, withValues(written.get(5), written.get(5).values(), null));
    var fitter =
        new LedgerFit(LedgerGroup.computed(), 0.001, 32, LedgerCells.CLASS_STAGE_RANK, variant);
    int shards = 3;
    var plain = fitter.tap(shards, index -> true);
    var retaining = fitter.tapRetaining(shards, index -> true);
    List<List<SyntheticRow>> byShard = new ArrayList<>();
    for (int shard = 0; shard < shards; shard++) byShard.add(new ArrayList<>());
    for (int i = 0; i < written.size(); i++) byShard.get(i % shards).add(written.get(i));
    for (int shard : new int[] {1, 0, 2})
      for (SyntheticRow row : byShard.get(shard)) {
        plain.add(shard, row);
        retaining.add(shard, row);
      }
    assertBitwiseEqual(retaining.fit(), plain.fit());
    Assert.expectThrows(IllegalStateException.class, plain::retained);
    List<SyntheticRow> expected = new ArrayList<>();
    for (List<SyntheticRow> shard : byShard)
      for (SyntheticRow row : shard) if (row.ledger() != null) expected.add(row);
    List<FitRow> kept = new ArrayList<>();
    retaining.retained().forEach(kept::add);
    Assert.assertEquals(kept.size(), expected.size());
    for (int i = 0; i < kept.size(); i++) {
      FitRow row = kept.get(i);
      SyntheticRow source = expected.get(i);
      Assert.assertEquals(row.gameId(), source.gameId());
      Assert.assertEquals(row.decisionClass(), source.decisionClass());
      Assert.assertEquals(row.cell(), source.cell());
      Assert.assertEquals(row.standing(), source.standing());
      Assert.assertEquals(row.values(), Arrays.copyOf(source.values(), 3));
      for (int a = 0; a < 3; a++)
        for (int f = 0; f < F; f++) Assert.assertEquals(row.ledger(a, f), source.ledger(a, f));
    }
    Assert.expectThrows(IllegalStateException.class, retaining::retained);
    Assert.expectThrows(IllegalStateException.class, () -> CLASSIC.tapRetaining(1, index -> true));
  }

  /**
   * コホートの行を貯めてから当てはめても、係数と報告は行を直に足したものとビットまで同じ。貯めた行の局面は、行の O から読んだ局面と同じ。
   * 卓を打ち進めて集めた O で、四人打ちと三人打ちの行を作って確かめる。
   */
  @Test
  public void retainingCohortRowsKeepsTheFit() {
    for (Variant variant : Variant.values()) {
      List<long[]> observations =
          LedgerCalibratorTest.observations(variant, 61, 900, view -> view.legalActions() >= 2);
      var random = new SplittableRandom(17);
      List<Row> rows = new ArrayList<>();
      for (int i = 0; i < observations.size(); i++) {
        long[] words = observations.get(i);
        int actions = new Observation(variant).bind(words, 0).legalActions();
        float[] values = new float[actions], ledger = new float[actions * F];
        Arrays.fill(values, Float.NaN);
        Arrays.fill(ledger, Float.NaN);
        for (int a = 0; a < Math.min(actions, 3); a++) {
          values[a] = (float) random.nextGaussian();
          for (int f = 0; f < F; f++)
            ledger[a * F + f] = LedgerCalibratorTest.half(random.nextGaussian());
        }
        DecisionClass c = i % 2 == 0 ? DecisionClass.DISCARD_MID : DecisionClass.CALL;
        rows.add(LedgerCalibratorTest.row(i, c, Row.Source.MAIN, words, values, ledger, 0));
      }
      var fitter =
          new LedgerFit(LedgerGroup.computed(), 0.001, 2, LedgerCells.CLASS_STAGE_RANK, variant);
      LedgerFit.Retained retained = fitter.fitRetaining(rows);
      assertBitwiseEqual(retained.result(), fitter.fit(rows));
      LedgerCell cells = LedgerCell.of(variant);
      int i = 0;
      for (FitRow row : retained.rows()) {
        Row source = rows.get(i++);
        Assert.assertEquals(row.standing(), cells.standing(source.observation(), 0));
        Assert.assertEquals(row.cell(), cells.of(source.observation(), 0));
      }
      Assert.assertEquals(i, rows.size());
    }
    Assert.expectThrows(IllegalStateException.class, () -> CLASSIC.fitRetaining(List.of()));
  }

  /** 2 つの当てはめで、係数と報告（秒数を除く）がビット単位で一致することを確かめる。 */
  public static void assertBitwiseEqual(LedgerFit.Result actual, LedgerFit.Result expected) {
    double[][][] a = actual.correction().beta(), e = expected.correction().beta();
    for (int g = 0; g < e.length; g++)
      for (int c = 0; c < e[g].length; c++)
        for (int f = 0; f < e[g][c].length; f++)
          Assert.assertEquals(
              Double.doubleToRawLongBits(a[g][c][f]),
              Double.doubleToRawLongBits(e[g][c][f]),
              "beta " + g + " " + c + " " + f);
    Gson gson = new GsonBuilder().serializeSpecialFloatingPointValues().create();
    JsonObject left = gson.toJsonTree(actual.report()).getAsJsonObject();
    JsonObject right = gson.toJsonTree(expected.report()).getAsJsonObject();
    left.remove("seconds");
    right.remove("seconds");
    Assert.assertEquals(left.toString(), right.toString());
  }

  static SyntheticRow withValues(SyntheticRow row, float[] values, float[] ledger) {
    return new SyntheticRow(row.gameId(), row.decisionClass(), values, ledger);
  }
}
