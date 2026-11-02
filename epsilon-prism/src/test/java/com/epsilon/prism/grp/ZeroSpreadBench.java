package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.core.ScoreRanking;
import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.rows.Row;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import org.testng.SkipException;
import org.testng.annotations.Test;

/**
 * 点数の順位の Φ でラベルの差が 0 になる割合（CPU だけ、収集はしない）。本番の封印済みのコホート（{@code -Dprism.grp.zero.cohort=<rows-v2/t>}）の
 * 主経路のラベル行で、測った全ての手（support ∪ played）の Φ が同じになる行の割合を、点数の順位・本番の Φ（行の値 Y、GRP ＋ 較正の表）・点数の
 * なめらかな関数（{@link ScoreSmoothPhi}）で、判断の種類（打牌・鳴き・立直・その他）と判断の局ごとに出す。点数の順位となめらかな関数の値は、
 * 手ごとの局末の記録（{@link OutcomeCodec}: 判断の席から見た 4 席の精算後の点数と次局の局）から収集と同じ式で作る（半荘が終わった手は最終順位の
 * 効用で、行の値と一致することを確かめる）。局末の記録のない手（ツモ・ロン・九種）を測った行は割合に入れず、数だけ出す。
 *
 * <p>事前登録の線: E1〜E4 の打牌の行か E1〜E4 の鳴きの行で、点数の順位の割合が 30% を超えれば傾向あり（CONFIRMED）。この割合には、測った
 * 全ての手の局末が同じでどの Φ でも差が 0 になる行（本番の Φ の差がちょうど 0、GRP0）も入るので、判定の行には点数の順位に固有の分を見る
 * ための GRP0 と、本番の Φ に差がある行での割合 P(SR0 | GRP > 0) も並べる（判定の規則は変えない）。結果は標準出力と、
 * {@code -Dprism.grp.zero.out=<file.json>} があればその JSON に書く。名前が *Test でないので、既定の試験には入らず {@code -Dtest} で名指ししたとき
 * だけ走る。
 */
public class ZeroSpreadBench {
  private static final String COHORT_PROPERTY = "prism.grp.zero.cohort";
  private static final String OUT_PROPERTY = "prism.grp.zero.out";

  /** 事前登録の線（点数の順位の割合）。 */
  private static final double THRESHOLD = 0.30;

  /** 「ほぼ 0」の差の線（効用の単位）。 */
  private static final double NEAR = 1e-4;

  /** 判断の局の区分（西入はまとめる）。 */
  private static final String[] HANDS = {"E1", "E2", "E3", "E4", "S1", "S2", "S3", "S4", "W"};

  /** 判断の種類。 */
  enum Kind {
    DISCARD,
    CALL,
    RIICHI,
    OTHER;

    static Kind of(DecisionClass decisionClass) {
      return switch (decisionClass) {
        case DISCARD_EARLY, DISCARD_MID, DISCARD_LATE, THREAT -> DISCARD;
        case CALL -> CALL;
        case RIICHI -> RIICHI;
        case KAN, END, FINAL_END -> OTHER;
      };
    }
  }

  /** 区分ごとの数。 */
  static final class Tally {
    long rows;
    long single;
    long incomplete;
    long scoreRankZero;
    long grpZero;
    long grpNear;
    long smoothZero;
    long smoothNear;
    long grpPositive;
    long scoreRankZeroGrpPositive;
    double grpSpread;
    double grpSpreadScoreRankZero;
    double smoothSpread;

    void add(Tally other) {
      rows += other.rows;
      single += other.single;
      incomplete += other.incomplete;
      scoreRankZero += other.scoreRankZero;
      grpZero += other.grpZero;
      grpNear += other.grpNear;
      smoothZero += other.smoothZero;
      smoothNear += other.smoothNear;
      grpPositive += other.grpPositive;
      scoreRankZeroGrpPositive += other.scoreRankZeroGrpPositive;
      grpSpread += other.grpSpread;
      grpSpreadScoreRankZero += other.grpSpreadScoreRankZero;
      smoothSpread += other.smoothSpread;
    }

    double scoreRankShare() {
      return share(scoreRankZero, rows);
    }

    Map<String, Object> report() {
      var out = new LinkedHashMap<String, Object>();
      out.put("rows", rows);
      out.put("singleCandidate", single);
      out.put("withoutOutcome", incomplete);
      out.put("scoreRankZero", share(scoreRankZero, rows));
      out.put("grpZero", share(grpZero, rows));
      out.put("grpNearZero", share(grpNear, rows));
      out.put("smoothZero", share(smoothZero, rows));
      out.put("smoothNearZero", share(smoothNear, rows));
      out.put("scoreRankZeroGivenGrpPositive", share(scoreRankZeroGrpPositive, grpPositive));
      out.put("meanGrpSpread", share(grpSpread, rows));
      out.put("meanGrpSpreadWhereScoreRankZero", share(grpSpreadScoreRankZero, scoreRankZero));
      out.put("meanSmoothSpread", share(smoothSpread, rows));
      return out;
    }

    String line(String name) {
      return String.format(
          "%-15s rows %9d  SR0 %6.2f%%  GRP0 %6.2f%%  GRP<1e-4 %6.2f%%  SM0 %6.2f%%  SM<1e-4 %6.2f%%"
              + "  P(SR0|GRP>0) %6.2f%%  spread GRP %.4f (SR0 %.4f) SM %.4f  single %d  noOutcome %d",
          name,
          rows,
          100 * share(scoreRankZero, rows),
          100 * share(grpZero, rows),
          100 * share(grpNear, rows),
          100 * share(smoothZero, rows),
          100 * share(smoothNear, rows),
          100 * share(scoreRankZeroGrpPositive, grpPositive),
          share(grpSpread, rows),
          share(grpSpreadScoreRankZero, scoreRankZero),
          share(smoothSpread, rows),
          single,
          incomplete);
    }

    private static double share(double count, double total) {
      return total == 0 ? Double.NaN : count / total;
    }
  }

  @Test
  public void measure() throws IOException {
    String given = System.getProperty(COHORT_PROPERTY);
    if (given == null || given.isBlank())
      throw new SkipException("set -D" + COHORT_PROPERTY + "=<rows-v2/t> to measure the zero label spread");
    EpsilonUtilityProfile profile = EpsilonUtilityProfile.TENHOU;
    Tally[][] tallies = new Tally[Kind.values().length][HANDS.length];
    Tally[] byClass = new Tally[DecisionClass.COUNT];
    for (Tally[] row : tallies) for (int h = 0; h < HANDS.length; h++) row[h] = new Tally();
    for (int c = 0; c < byClass.length; c++) byClass[c] = new Tally();
    long endedCandidates = 0;
    long endedMismatches = 0;
    long started = System.nanoTime();
    ExecutorService executor = Executors.newFixedThreadPool(4);
    try (CohortStore store = new CohortStore()) {
      store.addCohort(Path.of(given));
      int cohort = store.cohorts().getFirst();
      int[] rows = store.select(cohort, cohort, index -> index.source() == Row.Source.MAIN);
      Observation o = new Observation();
      int[] scores = new int[4];
      int[] measured = new int[Limits.MAX_LEGAL_ACTIONS];
      double[] scoreRank = new double[Limits.MAX_LEGAL_ACTIONS];
      double[] smooth = new double[Limits.MAX_LEGAL_ACTIONS];
      for (Row row : store.iterate(rows, 1 << 16, executor, 4)) {
        o.bind(row.observation(), 0);
        Tally tally =
            tallies[Kind.of(row.decisionClass()).ordinal()][Math.min(o.kyoku(), HANDS.length - 1)];
        Tally classTally = byClass[row.decisionClass().ordinal()];
        int count = 0;
        for (int a = 0; a < row.actions(); a++) if (row.measured(a)) measured[count++] = a;
        if (count < 2) {
          tally.single++;
          classTally.single++;
          continue;
        }
        boolean complete = true;
        long[] outcomes = row.outcomes();
        int observer = o.observer();
        for (int i = 0; i < count; i++) {
          int at = measured[i] * OutcomeCodec.WORDS;
          if (OutcomeCodec.outcome(outcomes, at) == null) {
            complete = false;
            break;
          }
          for (int relative = 0; relative < 4; relative++)
            scores[o.absoluteSeat(relative)] = OutcomeCodec.finalScore(outcomes, at, relative);
          scoreRank[i] = profile.utilityForRankDouble(ScoreRanking.byScoreThenSeat(scores)[observer]);
          int next = OutcomeCodec.nextKyoku(outcomes, at);
          if (next < 0) {
            smooth[i] = scoreRank[i];
            endedCandidates++;
            if (Math.abs(scoreRank[i] - row.values()[measured[i]]) > 1e-6) endedMismatches++;
          } else {
            smooth[i] = ScoreSmoothPhi.utilities(scores, next, profile)[observer];
          }
        }
        if (!complete) {
          tally.incomplete++;
          classTally.incomplete++;
          continue;
        }
        double srSpread = spread(scoreRank, count);
        double smSpread = spread(smooth, count);
        double grpMin = Double.POSITIVE_INFINITY;
        double grpMax = Double.NEGATIVE_INFINITY;
        for (int i = 0; i < count; i++) {
          double y = row.values()[measured[i]];
          grpMin = Math.min(grpMin, y);
          grpMax = Math.max(grpMax, y);
        }
        double grpSpread = grpMax - grpMin;
        for (Tally t : new Tally[] {tally, classTally}) {
          t.rows++;
          if (srSpread == 0) {
            t.scoreRankZero++;
            t.grpSpreadScoreRankZero += grpSpread;
          }
          if (grpSpread == 0) t.grpZero++;
          if (grpSpread < NEAR) t.grpNear++;
          if (smSpread == 0) t.smoothZero++;
          if (smSpread < NEAR) t.smoothNear++;
          if (grpSpread > 0) {
            t.grpPositive++;
            if (srSpread == 0) t.scoreRankZeroGrpPositive++;
          }
          t.grpSpread += grpSpread;
          t.smoothSpread += smSpread;
        }
      }
    } finally {
      executor.shutdownNow();
    }

    var report = new LinkedHashMap<String, Object>();
    report.put("cohort", given);
    report.put("seconds", (System.nanoTime() - started) / 1e9);
    report.put("endedCandidates", endedCandidates);
    report.put("endedMismatches", endedMismatches);
    System.out.printf(
        "cohort %s  ended candidates %d  mismatches %d%n", given, endedCandidates, endedMismatches);
    var kinds = new LinkedHashMap<String, Object>();
    var shares = new LinkedHashMap<String, Object>();
    Tally all = new Tally();
    boolean confirmed = false;
    for (Kind kind : Kind.values()) {
      var hands = new LinkedHashMap<String, Object>();
      Tally east = new Tally();
      Tally total = new Tally();
      for (int h = 0; h < HANDS.length; h++) {
        Tally t = tallies[kind.ordinal()][h];
        hands.put(HANDS[h], t.report());
        System.out.println(t.line(kind + " " + HANDS[h]));
        if (h < 4) east.add(t);
        total.add(t);
      }
      hands.put("E1-E4", east.report());
      hands.put("all", total.report());
      System.out.println(east.line(kind + " E1-E4"));
      System.out.println(total.line(kind + " all"));
      kinds.put(kind.name(), hands);
      all.add(total);
      if (kind == Kind.DISCARD || kind == Kind.CALL) {
        if (east.scoreRankShare() > THRESHOLD) confirmed = true;
        var entry = new LinkedHashMap<String, Object>();
        entry.put("scoreRankZero", east.scoreRankShare());
        entry.put("everyPhiZero", Tally.share(east.grpZero, east.rows));
        entry.put("scoreRankZeroGivenGrpPositive", Tally.share(east.scoreRankZeroGrpPositive, east.grpPositive));
        shares.put(kind.name(), entry);
      }
    }
    var classes = new LinkedHashMap<String, Object>();
    for (DecisionClass c : DecisionClass.values()) {
      classes.put(c.name(), byClass[c.ordinal()].report());
      System.out.println(byClass[c.ordinal()].line(c.name()));
    }
    System.out.println(all.line("ALL"));
    String verdict = confirmed ? "CONFIRMED" : "NOT_CONFIRMED";
    report.put("kinds", kinds);
    report.put("classes", classes);
    report.put("all", all.report());
    report.put("verdict", verdict);
    report.put("verdictShares", shares);
    System.out.println("verdict " + verdict + " (E1-E4: SR0 / every Φ zero / P(SR0 | GRP > 0)) " + shares);
    String out = System.getProperty(OUT_PROPERTY);
    if (out != null && !out.isBlank())
      Files.writeString(
          Path.of(out),
          new GsonBuilder()
              .setPrettyPrinting()
              .serializeSpecialFloatingPointValues()
              .create()
              .toJson(report),
          StandardCharsets.UTF_8);
  }

  private static double spread(double[] values, int count) {
    double min = Double.POSITIVE_INFINITY;
    double max = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < count; i++) {
      min = Math.min(min, values[i]);
      max = Math.max(max, values[i]);
    }
    return max - min;
  }
}
