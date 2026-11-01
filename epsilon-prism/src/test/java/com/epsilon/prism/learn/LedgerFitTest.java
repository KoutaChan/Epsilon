package com.epsilon.prism.learn;

import com.epsilon.prism.collect.EventLedger;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.rows.LedgerRow;
import com.epsilon.prism.rows.Row;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 運の精算の交差適合（{@link LedgerFit}）: 植えた係数を戻し、held-out の R² は理論値、偶奇 g の行に掛ける β は偶奇 g の行に依らない、行の少ない
 * 類はまとめた当てはめに落ち、帳簿のない行と欠けた行はそのまま、ridge は特徴の大きさに依らない。
 */
public class LedgerFitTest {
  private static final int F = EventLedger.FEATURES;

  /**
   * 試験の行（帳簿のない行は ledger が null）。
   *
   * @param ledger 手ごとの帳簿（長さ 合法手 × F）
   */
  private record SyntheticRow(
      long gameId, DecisionClass decisionClass, Row.Group group, float[] values, float[] ledger)
      implements LedgerRow {
    @Override
    public boolean hasLedger() {
      return ledger != null;
    }

    @Override
    public int ledgerFeatures() {
      return ledger == null ? 0 : ledger.length / values.length;
    }

    @Override
    public float ledger(int action, int feature) {
      return ledger == null ? Float.NaN : ledger[action * ledgerFeatures() + feature];
    }

    int actions() {
      return values.length;
    }
  }

  /** 合成の行（3 手を測る。帳簿の欄 f は scale[f]·N(0, 1)、Y_a = Q_a + Σ_f beta[f]·L_af + N(0, noise²)）。 */
  private static SyntheticRow row(
      long gameId, DecisionClass decisionClass, Row.Group group, double[] scale, double[] beta, double noise, SplittableRandom random) {
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
    return new SyntheticRow(gameId, decisionClass, group, values, ledger);
  }

  private static double[] ones() {
    double[] scale = new double[F];
    Arrays.fill(scale, 1);
    return scale;
  }

  /** 植えた係数（欄 0・3・15 だけ）。 */
  private static double[] planted() {
    double[] beta = new double[F];
    beta[0] = 0.3;
    beta[3] = -0.2;
    beta[15] = 0.5;
    return beta;
  }

  private static List<SyntheticRow> rows(int count, long firstGame, double[] scale, double[] beta, double noise, long seed) {
    var random = new SplittableRandom(seed);
    List<SyntheticRow> rows = new ArrayList<>();
    for (int i = 0; i < count; i++) {
      // 類・組・偶奇（半荘 ID の下位ビット）は互いに独立
      DecisionClass c = i % 3 == 0 ? DecisionClass.CALL : DecisionClass.DISCARD_MID;
      rows.add(row(firstGame + i, c, i % 8 < 4 ? Row.Group.A : Row.Group.B, scale, beta, noise, random));
    }
    return rows;
  }

  /**
   * 植えた係数を戻す: 類 2 つ・行あたり 3 手・L は平均 0 の正規・Y = Q + β*·L + ε（Q_a = 0.1·a）。β_{g,c} ≈ β*（相対 5%）、held-out の R² が
   * 理論値（行の中で中心化した分散の比 (2/3)|β*|² / ((2/3)(|β*|² + σ²) + mean(Q̃²))）± 0.02、補正した Y の平均と Y の平均の差 ≈ 0。
   */
  @Test
  public void recoversAPlantedCoefficient() {
    double[] beta = planted();
    double noise = 0.2;
    List<SyntheticRow> rows = rows(20_000, 0, ones(), beta, noise, 1);
    LedgerFit.Result fit = LedgerFit.fit(rows);
    LedgerFit.Correction correction = fit.correction();
    for (int g = 0; g < 2; g++)
      for (DecisionClass c : new DecisionClass[] {DecisionClass.DISCARD_MID, DecisionClass.CALL})
        for (int f : new int[] {0, 3, 15}) {
          double got = correction.beta()[g][c.ordinal()][f];
          Assert.assertEquals(got, beta[f], 0.05 * Math.abs(beta[f]), "fold " + g + " " + c + " feature " + f);
        }
    double signal = 0;
    for (double b : beta) signal += b * b;
    // 中心化で L と ε の分散は 2/3 倍、Q̃ = (−0.1, 0, 0.1) の平方の平均は 0.02/3（L では説明できない）
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
    for (LedgerFit.FeatureBias bias : fit.report().bias().all())
      Assert.assertTrue(Math.abs(bias.z()) < 4.5, bias.name() + " z=" + bias.z());
    Assert.assertTrue(fit.report().all().equivalentForks() > 1);
  }

  /** 偶の半荘の行の Y だけを変えても、偶の行に掛ける β（= 奇の行で当てたもの）は変わらない。 */
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
      for (int a = 0; a < values.length; a++) if (!Float.isNaN(values[a])) values[a] = values[a] * 3 + 1;
      shifted.add(withValues(row, values, row.ledger()));
    }
    double[][][] before = LedgerFit.fit(rows).correction().beta();
    double[][][] after = LedgerFit.fit(shifted).correction().beta();
    for (int c = 0; c < DecisionClass.COUNT; c++) Assert.assertTrue(Arrays.equals(after[0][c], before[0][c]), "class " + c);
    Assert.assertFalse(Arrays.equals(after[1][DecisionClass.CALL.ordinal()], before[1][DecisionClass.CALL.ordinal()]));
  }

  /** 行の少ない類（偶奇ごとに MIN_ROWS 未満）は、他方の偶奇の全ての類をまとめた当てはめを使う。全体でも足りなければ 0。 */
  @Test
  public void sparseClassesFallBackToThePooledFit() {
    var random = new SplittableRandom(4);
    List<SyntheticRow> rows = new ArrayList<>(rows(6000, 0, ones(), planted(), 0.5, 4));
    for (int i = 0; i < 100; i++)
      rows.add(row(100_000 + i, DecisionClass.KAN, Row.Group.A, ones(), planted(), 0.5, random));
    LedgerFit.Result fit = LedgerFit.fit(rows);
    LedgerFit.Line kan = fit.report().byClass().get(DecisionClass.KAN.name());
    Assert.assertTrue(kan.fallback()[0] && kan.fallback()[1]);
    Assert.assertFalse(fit.report().byClass().get(DecisionClass.CALL.name()).fallback()[0]);
    double[][][] beta = fit.correction().beta();
    Assert.assertTrue(Arrays.equals(beta[0][DecisionClass.KAN.ordinal()], fit.report().all().betaByFold()[0]));
    Assert.assertTrue(Arrays.equals(beta[1][DecisionClass.KAN.ordinal()], fit.report().all().betaByFold()[1]));
    // 全体でも足りなければ係数は 0
    LedgerFit.Result tiny = LedgerFit.fit(rows.subList(0, 50));
    for (double value : tiny.correction().beta()[0][DecisionClass.CALL.ordinal()]) Assert.assertEquals(value, 0.0);
  }

  /** 帳簿のない行と、測った手の帳簿に NaN を含む行は values のまま（当てはめにも入れず、skippedRows に数える）。 */
  @Test
  public void rowsWithoutLedgerPassThrough() {
    List<SyntheticRow> rows = new ArrayList<>(rows(3000, 0, ones(), planted(), 0.5, 5));
    SyntheticRow bare = withValues(rows.get(10), rows.get(10).values(), null);
    float[] broken = rows.get(11).ledger().clone();
    broken[F + 2] = Float.NaN;
    SyntheticRow partial = withValues(rows.get(11), rows.get(11).values(), broken);
    rows.add(bare);
    rows.add(partial);
    LedgerFit.Result fit = LedgerFit.fit(rows);
    Assert.assertEquals(fit.report().skippedRows(), 2);
    Assert.assertSame(fit.correction().values(bare), bare.values());
    Assert.assertSame(fit.correction().values(partial), partial.values());
    Assert.assertNotSame(fit.correction().values(rows.get(0)), rows.get(0).values());
  }

  /**
   * ridge は特徴ごとの対角に比例する: 1 つの特徴を 10^4 倍しても補正した Y が変わらない（相対 1e-6）。方策の項の大きさ（0.05）の特徴の植えた係数は、
   * 点の項の大きさ（3）の特徴と並べても縮まずに戻る。
   */
  @Test
  public void ridgeIsScaleInvariant() {
    double[] scale = ones();
    double[] beta = planted();
    List<SyntheticRow> rows = rows(4000, 0, scale, beta, 0.5, 6);
    List<SyntheticRow> scaled = new ArrayList<>();
    for (SyntheticRow row : rows) {
      float[] ledger = row.ledger().clone();
      for (int a = 0; a < row.actions(); a++) if (!Float.isNaN(row.values()[a])) ledger[a * F + 7] *= 10_000f;
      scaled.add(withValues(row, row.values(), ledger));
    }
    LedgerFit.Correction plain = LedgerFit.fit(rows).correction();
    LedgerFit.Correction wide = LedgerFit.fit(scaled).correction();
    for (int i = 0; i < 200; i++) {
      float[] a = plain.values(rows.get(i)), b = wide.values(scaled.get(i));
      for (int k = 0; k < a.length; k++)
        if (!Float.isNaN(a[k])) Assert.assertEquals(b[k], a[k], 1e-5 * Math.max(1, Math.abs(a[k])), "row " + i);
    }
    // 小さい特徴（0.05）と大きい特徴（3）を並べる
    double[] mixed = new double[F];
    Arrays.fill(mixed, 3);
    mixed[15] = 0.05;
    double[] coefficients = new double[F];
    coefficients[15] = 4.0;
    coefficients[3] = 0.05;
    LedgerFit.Correction fit = LedgerFit.fit(rows(20_000, 0, mixed, coefficients, 0.2, 7)).correction();
    for (int g = 0; g < 2; g++)
      Assert.assertEquals(fit.beta()[g][DecisionClass.CALL.ordinal()][15], 4.0, 0.2, "fold " + g);
  }

  /**
   * 書き手ごとに書いた順に足す {@link LedgerFit.Stream} は、シャード 0、1、2 の順に並べた列の {@link LedgerFit#fit} とビットで同じ（シャードの
   * 間の呼ぶ順に依らない）。帳簿は fp16 に丸め、行ごとに欄の一部を全ての手で 0（中心化して 0）、1 手だけ 0 にし、飛ばす行（帳簿なし・測った手が 1 つ）
   * も混ぜる。
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
    var stream = new LedgerFit.Stream(shards);
    for (int shard : new int[] {2, 0, 1}) for (SyntheticRow row : byShard.get(shard)) stream.add(shard, row);
    LedgerFit.Result streamed = stream.fit();
    Assert.assertEquals(streamed.report().skippedRows(), 2);
    assertBitwiseEqual(streamed, LedgerFit.fit(readBack));
  }

  /** 2 つの当てはめの係数と報告（秒を除く）がビットで同じ。 */
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

  private static SyntheticRow withValues(SyntheticRow row, float[] values, float[] ledger) {
    return new SyntheticRow(row.gameId(), row.decisionClass(), row.group(), values, ledger);
  }
}
