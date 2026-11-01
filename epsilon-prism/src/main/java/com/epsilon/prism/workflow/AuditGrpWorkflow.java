package com.epsilon.prism.workflow;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.config.settings.InferenceBatchingSettings;
import com.epsilon.core.GameState;
import com.epsilon.core.ScoreRanking;
import com.epsilon.engine.RoundResult;
import com.epsilon.nano.ai.grp.EpsilonGrpTrainingSession;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.spi.PolicyExecutionContext;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.BufferedWriter;
import java.io.OutputStreamWriter;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.zip.GZIPOutputStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 局末の評価 Φ（固定の GRP）の監査（audit-grp）。版を 4 席に置いた自己対局（Arena、牌山は評価と同じ）を打ち、主経路の全ての局末で Φ の入力
 * （次局 root の公開系列）と点数・局・本場・供託・親を記録し、半荘の最終順位と並べる。対局の後に GRP で全ての局末の順位分布（4 席 × 4 順位）を
 * 求め、1 局末 1 行の CSV（gzip）に書く。PRISM の打ち方のもとで E[最終効用 − Φ | 状況] = 0 か（GRP の較正）を、外の解析で調べるための記録。
 *
 * <p>列: game, hand, kyoku, honba, sticks, dealer, s0..s3（精算後の点数）, ended（半荘が終わった局は 1。Φ は最終順位の効用そのもので、
 * p は空）, p0..p15（GRP の順位分布、席 i・順位 r が p[4i+r]）, phi0..phi3（Φ = 期待効用）, rank0..rank3（最終順位 0〜3）,
 * u0..u3（最終の効用）。効用は設定の utility。
 */
public final class AuditGrpWorkflow implements CommandWorkflow<Path> {
  private static final Logger LOG = LoggerFactory.getLogger(AuditGrpWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "audit-grp", WorkflowKind.AUDIT, "audit-grp --snapshot DIR --games N --out FILE.csv.gz", 6, 6);
  }

  /** 1 局末の記録。prefix は半荘が続く局だけ（終わった局は null）。 */
  record HandRecord(int kyoku, int honba, int sticks, int dealer, int[] scores, float[] prefix) {}

  /** 1 半荘の記録（進行ワーカーから順に書かれる）。 */
  static final class GameLog {
    final List<HandRecord> hands = new ArrayList<>();
    int kyoku, honba, sticks, dealer;
    int[] finals;
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path snapshot = options.path("snapshot");
    int games = Integer.parseInt(options.optionalText("games", "0"));
    Path out = options.path("out");
    options.requireEmpty();
    if (games < 1) throw new IllegalArgumentException("audit-grp requires --games N");
    PrismConfig config = PrismConfig.fromSettings(arguments.settings());
    config.validateTraining();
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    Map<Long, GameLog> logs = new ConcurrentHashMap<>();
    CollectionDriver.GameObserver watcher =
        new CollectionDriver.GameObserver() {
          @Override
          public void gameEnded(long gameId, int[] finalScores) {
            logs.computeIfAbsent(gameId, id -> new GameLog()).finals = finalScores.clone();
          }

          @Override
          public void handEnded(long gameId, GameState state, RoundResult result) {
            GameLog log = logs.computeIfAbsent(gameId, id -> new GameLog());
            log.kyoku = state.roundIndex();
            log.honba = state.honba();
            log.sticks = state.riichiSticks();
            log.dealer = state.dealer();
          }

          @Override
          public void handValued(long gameId, float[] nextRootPrefix, int[] scores) {
            GameLog log = logs.computeIfAbsent(gameId, id -> new GameLog());
            log.hands.add(
                new HandRecord(
                    log.kyoku,
                    log.honba,
                    log.sticks,
                    log.dealer,
                    scores.clone(),
                    nextRootPrefix == null ? null : nextRootPrefix.clone()));
          }
        };
    try (var execution =
            new PolicyExecutionContext(config.encoders(), config.slots(), config.readyBatches());
        var driver =
            new CollectionDriver(
                new CollectionDriver.Settings(
                    config.maxActiveGames(),
                    config.advanceWorkers(),
                    new InferenceBatchingSettings(config.batchWaitMicros())));
        var policy =
            Snapshot.open(
                snapshot,
                PrismConfig.devices(config.collectDevices()),
                execution,
                config.inferBatch(),
                config.priorSettings(),
                config.inference())) {
      var result =
          Arena.duel(
              driver,
              policy,
              policy,
              "self",
              games,
              config.seed() ^ Arena.EVALUATION_SALT,
              Arena.HEURISTIC_BASE,
              profile,
              watcher);
      LOG.info("audit-grp self-play {} games in {}s", result.games(), String.format("%.0f", result.seconds()));
    }
    // 半荘が続く局末の Φ の入力を、GRP でまとめて評価する
    List<Long> ids = new ArrayList<>(logs.keySet());
    ids.sort(Long::compare);
    List<float[]> prefixes = new ArrayList<>();
    for (long id : ids)
      for (HandRecord hand : logs.get(id).hands) if (hand.prefix() != null) prefixes.add(hand.prefix());
    List<float[]> marginals = new ArrayList<>(prefixes.size());
    try (var grp =
        EpsilonGrpTrainingSession.openInferenceOnly(
            Path.of(config.grpCheckpoint()), PrismConfig.device(config.grpDevice()))) {
      for (int from = 0; from < prefixes.size(); from += config.grpBatch())
        marginals.addAll(
            grp.inference()
                .predictMarginalProbabilities(
                    prefixes.subList(from, Math.min(prefixes.size(), from + config.grpBatch()))));
    }
    Files.createDirectories(out.toAbsolutePath().getParent());
    int next = 0;
    long rows = 0;
    try (var writer =
        new BufferedWriter(
            new OutputStreamWriter(
                new GZIPOutputStream(Files.newOutputStream(out)), StandardCharsets.UTF_8))) {
      StringBuilder header = new StringBuilder("game,hand,kyoku,honba,sticks,dealer,s0,s1,s2,s3,ended");
      for (int k = 0; k < 16; k++) header.append(",p").append(k);
      for (String prefix : new String[] {"phi", "rank", "u"})
        for (int seat = 0; seat < 4; seat++) header.append(',').append(prefix).append(seat);
      writer.write(header.append('\n').toString());
      for (long id : ids) {
        GameLog log = logs.get(id);
        if (log.finals == null) throw new IllegalStateException("Game " + id + " has no final scores");
        int[] ranks = ScoreRanking.byScoreThenSeat(log.finals);
        double[] finalUtilities = new double[4];
        for (int seat = 0; seat < 4; seat++)
          finalUtilities[seat] = EpsilonUtilityTargets.utilityDouble(profile, log.finals[seat], ranks[seat]);
        for (int h = 0; h < log.hands.size(); h++) {
          HandRecord hand = log.hands.get(h);
          boolean ended = hand.prefix() == null;
          float[] p = ended ? null : marginals.get(next++);
          StringBuilder line = new StringBuilder();
          line.append(id).append(',').append(h).append(',').append(hand.kyoku()).append(',')
              .append(hand.honba()).append(',').append(hand.sticks()).append(',').append(hand.dealer());
          for (int seat = 0; seat < 4; seat++) line.append(',').append(hand.scores()[seat]);
          line.append(',').append(ended ? 1 : 0);
          for (int k = 0; k < 16; k++) line.append(',').append(ended ? "" : String.format(Locale.ROOT, "%.6f", p[k]));
          for (int seat = 0; seat < 4; seat++)
            line.append(',')
                .append(
                    String.format(
                        Locale.ROOT,
                        "%.6f",
                        ended
                            ? finalUtilities[seat]
                            : EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(profile, p, seat * 4)));
          for (int seat = 0; seat < 4; seat++) line.append(',').append(ranks[seat]);
          for (int seat = 0; seat < 4; seat++)
            line.append(',').append(String.format(Locale.ROOT, "%.6f", finalUtilities[seat]));
          writer.write(line.append('\n').toString());
          rows++;
        }
      }
    }
    if (next != marginals.size()) throw new IllegalStateException("GRP outputs were not all used");
    LOG.info("audit-grp wrote {} hand ends of {} games to {}", rows, ids.size(), out);
    return out;
  }

  @Override
  public String summarize(Path out) {
    return "audit-grp -> " + out;
  }
}
