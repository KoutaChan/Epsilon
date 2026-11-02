package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.EnumSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.stream.Stream;

/**
 * 点数だけの Φ から最初に GRP へ置き換える信頼の条件: 窓の半荘が grpTrustGames 以上、取り置きのコホートで絶対の較正、同じ窓を半荘の偶奇で
 * 2 つに分けて学んだ 2 つの候補の一致、単調（{@link GateMode#FIRST_REPLACEMENT} の関門）。人の牌譜は使わない。
 *
 * <p>関門は 2 回通す。1 回目は窓の全てで学んだ候補（入れるもの）で、絶対の較正・単調・形・窓の半荘の数を見る（双子は候補そのもので、安定は
 * ここでは見ない）。2 回目は窓の偶数の半荘と奇数の半荘で白紙から学んだ 2 つの候補を候補と双子にして、その ΔΦ の RMS（推定の雑音）だけを使う。
 * 双子は候補と同じ計画（{@link GrpTrainingPlan#of} の白紙）で、それぞれの半分の窓から候補と同じ規則で検証を取り（{@link GrpRefitter.Split}）、
 * 早めに打ち切る（候補という推定のばらつきを測る。打ち切らない双子どうしは過学習の差を測ってしまう）。報告は 1 回目の関門に 2 回目の安定を
 * 合わせたもの（落ちた関門は表の順、2 回目の数値は {@value #PARITY_PREFIX} を付けて足す）。2 回目に双子の RMS が測れなければ（双子の出力が
 * 壊れている・取り置きが足りない）安定は落ちたとみなす。形の確かめは取り置きの全ての局末で {@link GrpRefitter#wellFormed}。構造が
 * grp-epsilon-v3 でない checkpoint は {@link CheckpointPhi#open} が拒む。
 */
public final class TrustCriteria {

  /** 2 回目の関門の数値の名前の頭。 */
  static final String PARITY_PREFIX = "parity.";

  /** 双子の学習が試行の締め切りを過ぎた（試行は TIMEOUT）。 */
  static final class DeadlinePassed extends IOException {
    DeadlinePassed(String message) {
      super(message);
    }
  }

  private TrustCriteria() {}

  /** 窓の半荘の数が足りれば true（足りなければ学習しない）。 */
  public static boolean ready(PrismConfig config, long windowGames) {
    return windowGames >= config.grpTrustGames();
  }

  /**
   * 候補を最初の置き換えの関門に通す。半荘の偶奇の半分ずつで双子の候補を白紙から学び（{@link GrpGameTrainer}）、{@link GateEvaluator} を
   * FIRST_REPLACEMENT で呼ぶ。双子は候補の隣の一時のディレクトリに置き、終わったら消す。
   *
   * @param window 窓の半荘（偶奇の半分ずつを双子に使う）
   * @param holdout 取り置きのコホートの半荘
   * @param candidate 窓の全てで学んだ候補の GRP の正確なディレクトリ
   * @param active 使用中の点数だけの Φ の種類
   * @param remaining 試行の締め切りまでの残り（双子 2 つの学習で分け合う）
   * @throws DeadlinePassed 双子の学習が締め切りを過ぎたとき
   * @throws IOException 学習か評価ができないとき
   */
  public static GateReport gate(
      PrismConfig config,
      List<RecordedGame> window,
      List<RecordedGame> holdout,
      Path candidate,
      PhiEntryKind active,
      EpsilonUtilityProfile profile,
      Duration remaining)
      throws IOException {
    if (!active.scoreBased())
      throw new IllegalArgumentException("The trust criteria replace a score-only Φ, not " + active);
    List<RecordedGame> even = new ArrayList<>();
    List<RecordedGame> odd = new ArrayList<>();
    for (RecordedGame game : window) (game.gameId() % 2 == 0 ? even : odd).add(game);
    Path absolute = candidate.toAbsolutePath();
    Path twins = Files.createTempDirectory(absolute.getParent(), absolute.getFileName() + "-parity-");
    long deadline = System.nanoTime() + remaining.toNanos();
    try {
      Path evenGrp = train(even, config, holdout, 0, twins.resolve("even"), deadline);
      Path oddGrp = train(odd, config, holdout, 1, twins.resolve("odd"), deadline);
      try (HandEndPhi score = ScorePhi.offline(active, profile);
          HandEndPhi full = CheckpointPhi.open(candidate, new double[0], null, profile);
          HandEndPhi evenPhi = CheckpointPhi.open(evenGrp, new double[0], null, profile);
          HandEndPhi oddPhi = CheckpointPhi.open(oddGrp, new double[0], null, profile)) {
        GateReport main =
            GateEvaluator.evaluate(
                new GateInput(
                    GateMode.FIRST_REPLACEMENT,
                    holdout,
                    score,
                    full,
                    full,
                    structurallyValid(full.gameValues(holdout)),
                    window.size(),
                    config.grpTrustGames()),
                profile);
        GateReport parity =
            GateEvaluator.evaluate(
                new GateInput(
                    GateMode.FIRST_REPLACEMENT,
                    holdout,
                    score,
                    evenPhi,
                    oddPhi,
                    structurallyValid(evenPhi.gameValues(holdout))
                        && structurallyValid(oddPhi.gameValues(holdout)),
                    window.size(),
                    config.grpTrustGames()),
                profile);
        return merge(main, parity);
      }
    } finally {
      delete(twins);
    }
  }

  /** 1 回目の関門の報告に、2 回目の安定の結果と数値を合わせる（2 回目に双子の RMS が測れていなければ UNSTABLE）。 */
  static GateReport merge(GateReport main, GateReport parity) {
    Set<GateResult> failures = EnumSet.noneOf(GateResult.class);
    failures.addAll(main.failures());
    failures.remove(GateResult.UNSTABLE);
    Double twinRms = parity.measures().get(GateEvaluator.TWIN_RMS_MEASURE);
    if (parity.failures().contains(GateResult.UNSTABLE) || twinRms == null || !(twinRms <= GateEvaluator.TWIN_RMS))
      failures.add(GateResult.UNSTABLE);
    Map<String, Double> measures = new LinkedHashMap<>(main.measures());
    parity.measures().forEach((name, value) -> measures.put(PARITY_PREFIX + name, value));
    List<GateResult> ordered = new ArrayList<>(failures);
    return new GateReport(
        main.mode(),
        ordered.isEmpty() ? GateResult.PASS : ordered.getFirst(),
        List.copyOf(ordered),
        measures,
        main.active(),
        main.candidate());
  }

  /** 取り置きの局末があり、その Φ が形の確かめ（{@link GrpRefitter#wellFormed}）に通れば true。 */
  static boolean structurallyValid(PhiValues values) {
    return values.size() > 0 && GrpRefitter.wellFormed(values);
  }

  /**
   * 半分の窓で、候補と同じ計画と検証の取り方で白紙から学ぶ（種は run の種・取り置き・偶奇で決める。締め切りは試行の締め切り deadline、
   * {@link System#nanoTime} の時刻）。
   */
  private static Path train(
      List<RecordedGame> games, PrismConfig config, List<RecordedGame> holdout, int half, Path out, long deadline)
      throws IOException {
    long holdoutId = holdout.isEmpty() ? 0 : holdout.getFirst().gameId();
    long seed = config.seed() ^ (0x9E3779B97F4A7C15L * (2 * holdoutId + half + 1));
    GrpRefitter.Split split = GrpRefitter.Split.of(games);
    Duration left = Duration.ofNanos(Math.max(0, deadline - System.nanoTime()));
    GrpGameTrainer.Trained trained =
        GrpGameTrainer.train(
            null, split.training(), split.validation(), GrpTrainingPlan.of(GrpInit.SCRATCH, seed).within(left), out);
    if (trained.timedOut())
      throw new DeadlinePassed("the parity twin " + out.getFileName() + " ran past the attempt deadline");
    return trained.directory();
  }

  private static void delete(Path directory) throws IOException {
    if (!Files.exists(directory)) return;
    try (Stream<Path> paths = Files.walk(directory)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
