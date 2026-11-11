package com.epsilon.prism.grp;

import com.epsilon.grp.Phi;
import com.epsilon.grp.train.GameCohort;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GateCohort;
import com.epsilon.grp.train.GateReport;
import com.epsilon.grp.train.GrpGate;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.GrpRows;
import com.epsilon.grp.train.GrpTrained;
import com.epsilon.grp.train.GrpTrainer;
import com.epsilon.grp.train.HandEndValues;
import com.epsilon.grp.train.HandEnds;
import com.epsilon.grp.train.MonotoneProbe;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.grp.train.TrustGames;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.OptionalInt;

/**
 * 点数だけの Φ を初めて置き換えるのに要る半荘の数を、確定したコホートの半荘で測った結果。
 *
 * <p>範囲のコホートの半荘をまとめ、学習のときと同じハッシュ（{@link GateCohort#GUARDED}）で関門用の半荘を分ける。関門用の約 1/4 を
 * 除いた半荘で、最初の切り替えの候補と同じ形・同じ学習の決まりのネットワークをゼロから学習する。
 *
 * <p>関門用の半荘で、コホートを集めた点数だけの Φ と候補を比べる。出すのは、最初の切り替えの関門（{@link GrpGate#replacement}）と同じ
 * 判定と、2 SE の判定に要る半荘の数（{@link TrustGames}）。学習とブートストラップの乱数の種は、run の種と範囲で決まる。結果はどの
 * ファイルにも書かない。
 *
 * <p>範囲の半荘から関門用を分けるのは、この測りだけ。学び直し（{@link GrpRefitter}）は範囲の半荘を全て学習に使い、その関門は範囲の次の
 * コホートの全ての半荘で比べる（{@link GateCohort#NEXT}）。そのため、{@link TrustGames} が数える半荘は、1 コホートの半荘の数と比べる
 * （{@link #cohortHasPower}）。設定の {@code trustGames} は学び直しの範囲の半荘の数で、この測りでは候補を学習した半荘の数に当たる。
 *
 * @param firstCohort 範囲の最初のコホート
 * @param lastCohort 範囲の最後のコホート
 * @param used コホートを集めた Φ の項目
 * @param learnedGames 学習と早期終了に使った半荘の数
 * @param heldGames 関門用に残した半荘の数
 * @param cohortGames 範囲のコホートのうち、半荘が続く局末を持つ半荘が最も少ないコホートの、その半荘の数
 * @param rows 学習の行の数
 * @param stopRows 早期終了用の検証の行の数
 * @param trained 学習の結果
 * @param gate 関門用の半荘での、最初の切り替えの関門の判定
 * @param trust 関門用の半荘での、2 SE の判定に要る半荘の数
 */
public record TrustTrial(
    int firstCohort,
    int lastCohort,
    PhiEntry used,
    int learnedGames,
    int heldGames,
    int cohortGames,
    int rows,
    int stopRows,
    GrpTrained trained,
    GateReport gate,
    TrustGames trust) {

  /** 学習とブートストラップの乱数の種を作るときのソルト（"trst"）。 */
  private static final long SALT = 0x74727374L;

  /**
   * 学び直しの関門で比べる半荘が、検出力 95% で 2 SE の判定に要る数（{@link TrustGames#withPower}）に届くなら true を返す。学び直しの関門が
   * 比べるのは範囲の次のコホートの半荘なので、範囲で最も小さいコホートの半荘の数と比べる。候補が良くなっていなければ false。
   */
  public boolean cohortHasPower() {
    return cohortGames >= trust.withPower();
  }

  /**
   * 点数だけの Φ で集めて半荘の記録がある確定済みのコホートのうち、最新のものを返す。無ければ空。
   *
   * @throws IOException phi.jsonl か記録のディレクトリが読めないとき
   */
  public static OptionalInt latestCohort(GrpFiles files) throws IOException {
    List<PhiEntry> history = PhiHistory.read(files);
    for (int cohort : files.recordedCohorts().descendingSet()) {
      if (files.recorded(cohort)
          && PhiHistory.effectiveAt(history, cohort).map(entry -> entry.kind().scoreBased()).orElse(false)) {
        return OptionalInt.of(cohort);
      }
    }
    return OptionalInt.empty();
  }

  /**
   * 範囲のコホートの半荘で測る。
   *
   * @param resamples ブートストラップの回数（2 以上）
   * @throws IOException 半荘の記録か phi.jsonl が読めないとき
   * @throws IllegalArgumentException 範囲が空のとき、範囲に確定済みでないか記録の無いコホートがあるとき、範囲の全てを同じ点数だけの
   *     Φ で集めていないとき、記録の Φ の項目かルールが run と違うとき
   */
  public static TrustTrial run(GrpFiles files, PhiSetup setup, int firstCohort, int lastCohort, int resamples)
      throws IOException {
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry used = null;
    List<GameRecord> games = new ArrayList<>();
    int cohortGames = Integer.MAX_VALUE;
    for (int cohort = firstCohort; cohort <= lastCohort; cohort++) {
      if (!files.recorded(cohort)) {
        throw new IllegalArgumentException(
            "Cohort " + cohort + " of " + files.run() + " is not sealed with game records");
      }
      int at = cohort;
      PhiEntry entry =
          PhiHistory.effectiveAt(history, cohort)
              .orElseThrow(
                  () -> new IllegalArgumentException("No Φ entry for cohort " + at + " in " + files.phiHistory()));
      if (!entry.kind().scoreBased()) {
        throw new IllegalArgumentException(
            "Cohort " + cohort + " was collected with " + entry.id() + " (" + entry.kind() + "), not a score-only Φ");
      }
      if (used == null) {
        used = entry;
      } else if (!used.sameValue(entry)) {
        throw new IllegalArgumentException(
            "Cohorts " + firstCohort + ".." + lastCohort + " were collected with " + used.id() + " and " + entry.id());
      }
      GameCohort recorded = GameRecordCodec.read(files.records(cohort));
      if (recorded.variant() != setup.variant()) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + setup.variant() + " but found " + recorded.variant() + " in "
                + files.records(cohort));
      }
      if (!recorded.phiId().equals(entry.id())) {
        throw new IllegalArgumentException(
            "Cohort " + cohort + " was recorded with " + recorded.phiId() + ", not " + entry.id() + " of "
                + files.phiHistory());
      }
      games.addAll(recorded.games());
      cohortGames =
          Math.min(cohortGames, (int) recorded.games().stream().filter(game -> !game.handEnds().isEmpty()).count());
    }
    if (used == null) {
      throw new IllegalArgumentException("Empty cohort range " + firstCohort + ".." + lastCohort);
    }
    // 全ての半荘を、学習に使える最初の番号のコホート 1 つとして足す。こうすると、学習に使わない始めのコホートも測れる
    int slot = RecencyWindow.SKIPPED_COHORTS + 1;
    GrpRows.Split rows =
        GrpRows.builder(setup.variant(), RecencyWindow.uniform(slot, 1), GateCohort.GUARDED).add(slot, games).build();
    List<GameRecord> held = GateCohort.GUARDED.gateGames(games);
    GrpOperations operations = setup.selfGrp().grp();
    long seed = SeedMixer.derive(setup.seed(), SALT, firstCohort, lastCohort);
    GrpTrained trained =
        GrpTrainer.train(
            GrpRefitter.shape(setup.variant()),
            null,
            rows.train(),
            rows.stop(),
            GrpRefitter.rule(operations.training(), GrpInit.SCRATCH, seed),
            operations.threads(),
            () -> false);
    RankUtility utility = setup.utility();
    HandEnds ends = HandEnds.of(setup.variant(), utility, held);
    Phi usedPhi = HandEndPhi.open(used, utility);
    Phi candidatePhi = Phi.of(trained.network(), utility);
    HandEndValues usedValues = HandEndValues.of(ends, usedPhi);
    HandEndValues candidateValues = HandEndValues.of(ends, candidatePhi);
    MonotoneProbe probe = MonotoneProbe.of(ends);
    GateReport gate =
        GrpGate.replacement(
            ends, usedValues, candidateValues, probe.violationRate(usedPhi), probe.violationRate(candidatePhi));
    TrustGames trust = TrustGames.measure(ends, usedValues, candidateValues, resamples, SeedMixer.derive(seed, SALT));
    return new TrustTrial(
        firstCohort,
        lastCohort,
        used,
        games.size() - held.size(),
        held.size(),
        cohortGames,
        rows.train().size(),
        rows.stop().size(),
        trained,
        gate,
        trust);
  }
}
