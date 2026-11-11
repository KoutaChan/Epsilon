package com.epsilon.prism.grp;

import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.grp.FinalRound;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpField;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpShape;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.train.CountFile;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameRecordCodec;
import com.epsilon.grp.train.GateCohort;
import com.epsilon.grp.train.GrpInit;
import com.epsilon.grp.train.GrpRows;
import com.epsilon.grp.train.GrpTrained;
import com.epsilon.grp.train.GrpTrainer;
import com.epsilon.grp.train.GrpTrainingRule;
import com.epsilon.grp.train.OutcomeRows;
import com.epsilon.grp.train.OutcomeTrained;
import com.epsilon.grp.train.OutcomeTrainer;
import com.epsilon.grp.train.RecencyWindow;
import com.epsilon.grp.train.TrainingEnding;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.List;
import java.util.Optional;
import java.util.function.BooleanSupplier;
import java.util.regex.Matcher;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 時点 n の学び直しを 1 回行う。学習の JVM の {@link GrpBackground} のスレッドから呼ぶ。n までの範囲（{@link RecencyWindow}）の半荘の
 * 記録でネットワークを学び、候補を grp/candidates に書いて、refits.jsonl に TRAINED の行を書く。候補は、コホート n ＋ 1 が確定した後に
 * {@link GrpLifecycle} が関門に通す。停止のフラグが立てば試行を捨てて何も書かず、再起動の後に同じ乱数の種でやり直す。失敗したら
 * FAILED の行を書く。
 *
 * <p>範囲の半減期 τ は設定の値を使う。ただし Φ の構造を入れ替えた直後の 8 反復は 1 にし、監視が τ を 2 倍にしている間は 2 倍にする。始め方も設定の
 * 値を使うが、WARM でも、n ＋ 1 に使うと決まった項目が同じ形のネットワークでなければゼロから学ぶ。n ＋ 1 に使う項目が点数だけの Φ なら、
 * 範囲の半荘の数が設定の {@code trustGames} に届くまで学ばない。
 *
 * <p>ネットワークを学んだ後、同じ範囲・同じ半荘の分け方・同じ学び方で、q（局の結果のネットワーク）を学ぶ。q の学習に使うのは、数の
 * ファイルが覆う局（{@link FinalRound#covers}）だけで、数のファイルの無いコホートは外す。ウォームスタートの決まりもネットワークと同じで、
 * 使う項目が自己対局の候補で、同じ形の q を持つときだけ、その q から続けて学ぶ。候補は &lt;id&gt;、&lt;id&gt;.q の順に書き、最後に
 * TRAINED の行を書く。q を学べなくても試行は失敗にせず、q を持たない候補にする。
 */
public final class GrpRefitter {
  private static final Logger LOG = LoggerFactory.getLogger(GrpRefitter.class);

  /**
   * 学び直すネットワークの形を返す。入力のフィールドは局・本場・供託・点数・順位・点差で、隠れ層の幅は 128。出力層は段階ごとに分けない。
   *
   * @param variant ルール
   * @return 形
   */
  public static GrpShape shape(Variant variant) {
    return new GrpShape(
        variant,
        EnumSet.of(GrpField.ROUND, GrpField.HONBA, GrpField.KYOTAKU, GrpField.SCORE, GrpField.RANK, GrpField.GAP),
        128,
        false);
  }

  /**
   * 学び直す q の形を返す。入力のフィールドと隠れ層の幅は、学び直すネットワークと同じ。
   *
   * @param variant ルール
   * @return 形
   */
  public static OutcomeShape outcomeShape(Variant variant) {
    GrpShape network = shape(variant);
    return new OutcomeShape(variant, network.fields(), network.hidden());
  }

  /** 構造を入れ替えた直後に τ を 1 にする反復の数。監視で戻した後、戻した項目とも比べる関門を使う反復の数も同じ。 */
  static final int FAST_ITERATIONS = 8;

  private static final long SEED_SALT = 0x67727066L;

  private static final long OUTCOME_SALT = 0x71726566L;

  private GrpRefitter() {}

  /**
   * 学び直し 1 回の学び方を作る。重み減衰は掛けない。
   *
   * @param training 自分の GRP を運用する設定にある学び方
   */
  static GrpTrainingRule rule(GrpTraining training, GrpInit init, long seed) {
    boolean warm = init == GrpInit.WARM;
    return new GrpTrainingRule(
        init,
        warm ? training.warmRate() : training.scratchRate(),
        0,
        warm ? training.warmEpochs() : training.scratchEpochs(),
        training.batchRows(),
        training.clipNorm(),
        training.halvingPatience(),
        training.stopPatience(),
        seed);
  }

  /** 試行の id（g&lt;時点&gt;-a&lt;試行の番号&gt;）。 */
  static String attemptId(int slot, int attempt) {
    return String.format("g%03d-a%d", slot, attempt);
  }

  /**
   * 時点 {@code slot} の候補を学ぶ。その時点に試行が既にあるか、その時点のコホートに半荘の記録が無ければ何もしない。
   *
   * @param stop 停止のフラグ（学習の各ステップの前に見る）
   * @return 書いた refits.jsonl の行（学ばなかったか、止めたなら空）
   * @throws IOException refits.jsonl か phi.jsonl が読めないか書けないとき
   */
  static Optional<RefitRecord> refit(GrpFiles files, PhiSetup setup, int slot, RefitCause cause, BooleanSupplier stop)
      throws IOException {
    GrpOperations operations = setup.selfGrp().grp();
    List<RefitRecord> records = RefitRecord.readAll(files);
    if (RefitRecord.latestAt(records, slot).isPresent() || !files.recorded(slot)) return Optional.empty();
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry used = PhiHistory.effectiveAt(history, slot + 1).orElse(null);
    if (used == null) return Optional.empty();
    RecencyWindow window =
        new RecencyWindow(
            slot,
            halfLife(operations, history, SwitchWatchRecord.state(SwitchWatchRecord.readAll(files)), slot),
            operations.window());
    if (window.cohorts() == 0) return Optional.empty();
    long started = System.nanoTime();
    GrpRows.Builder builder = GrpRows.builder(setup.variant(), window, GateCohort.NEXT);
    long games = 0;
    for (int cohort = window.first(); cohort <= slot; cohort++) {
      if (!files.recorded(cohort)) {
        LOG.warn("GRP refit at {}: cohort {} has no game records and is left out", slot, cohort);
        continue;
      }
      var recorded = GameRecordCodec.read(files.records(cohort)).games();
      builder.add(cohort, recorded);
      games += recorded.size();
    }
    if (used.kind().scoreBased() && games < operations.trustGames()) {
      LOG.info("GRP refit at {} waits: {} games in {}, {} needed", slot, games, window, operations.trustGames());
      return Optional.empty();
    }
    GrpNetwork start = operations.init() == GrpInit.WARM ? warmStart(used, setup.variant()) : null;
    GrpInit init = start == null ? GrpInit.SCRATCH : GrpInit.WARM;
    int attempt = nextAttempt(files, records, slot);
    String id = attemptId(slot, attempt);
    var draft =
        new RefitRecord(id, slot, RefitStatus.FAILED, cause, window, init, used.id(), "", "", "", null, null, "", 0);
    RefitRecord record;
    try {
      GrpRows.Split rows = builder.build();
      long seed = SeedMixer.derive(setup.seed(), SEED_SALT, slot, attempt);
      GrpTrained trained =
          GrpTrainer.train(
              shape(setup.variant()),
              start,
              rows.train(),
              rows.stop(),
              rule(operations.training(), init, seed),
              operations.threads(),
              stop);
      if (trained.ending() == TrainingEnding.STOPPED) {
        LOG.info("GRP refit {} stopped after {}s", id, seconds(started));
        return Optional.empty();
      }
      Outcomes outcomes = learnOutcomes(files, setup, used, window, attempt, stop);
      if (outcomes == null) {
        LOG.info("GRP refit {} stopped after {}s", id, seconds(started));
        return Optional.empty();
      }
      Path candidate = files.candidate(id);
      GrpCheckpoint.write(trained.network(), candidate);
      outcomes = outcomes.write(files.candidateOutcomes(id));
      record =
          new RefitRecord(
              id,
              slot,
              RefitStatus.TRAINED,
              cause,
              window,
              init,
              used.id(),
              candidate.toAbsolutePath().normalize().toString(),
              outcomes.path(),
              "",
              null,
              null,
              String.format(
                  "games=%d rows=%d stopRows=%d halfEpochs=%d stopNll=%.6f clipped=%d; %s",
                  games,
                  rows.train().size(),
                  rows.stop().size(),
                  trained.halfEpochs(),
                  trained.stopNll(),
                  trained.clippedSteps(),
                  outcomes.detail()),
              seconds(started));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("GRP refit {} failed: {}", id, failure.toString(), failure);
      record = draft.with(RefitStatus.FAILED, null, failure.toString(), seconds(started));
    }
    record.append(files);
    LOG.info("GRP refit {} {} in {}s ({}): {}", id, record.status(), String.format("%.1f", record.seconds()), window, record.detail());
    return Optional.of(record);
  }

  /**
   * 時点 {@code slot} の範囲の半減期を返す。設定の τ を使うが、Φ の構造を入れ替えた直後の {@value #FAST_ITERATIONS} 反復は 1 にし、
   * 監視で τ を 2 倍にしている間は 2 倍にする。構造を入れ替えた行とは、自己対局の候補の行のうち、親が自己対局の候補でない
   * （点数だけの Φ か g000 の）もの。戻しの行は含めない。
   */
  static double halfLife(GrpOperations operations, List<PhiEntry> history, SwitchWatchRecord.State watch, int slot) {
    double halfLife = operations.halfLife();
    for (PhiEntry entry : history)
      if (entry.candidate()
          && !entry.rollback()
          && !PhiEntry.CANDIDATE.matcher(entry.parent()).matches()
          && entry.effectiveFrom() <= slot
          && slot < entry.effectiveFrom() + FAST_ITERATIONS) halfLife = 1;
    return watch.widened() ? 2 * halfLife : halfLife;
  }

  /** ウォームスタートの重み。使う項目が自己対局の候補で、同じ形のネットワークならその重みを返す。それ以外は null（ゼロから学ぶ）。 */
  private static GrpNetwork warmStart(PhiEntry used, Variant variant) throws IOException {
    if (!used.candidate()) return null;
    GrpModel model = GrpCheckpoint.read(Path.of(used.path()));
    return model instanceof GrpNetwork network && network.shape().equals(shape(variant)) ? network : null;
  }

  /**
   * 範囲の数のファイルが覆う局で q を学ぶ。学び直す時点は、範囲の最新のコホート。q の失敗で試行を失敗にはしないので、数のファイルや
   * 使う項目の q が読めないときも、学習が失敗したときも例外は投げず、q を持たない結果を返す。
   *
   * @param attempt 試行の番号（乱数の種に混ぜる）
   * @return 学んだ q（まだ書いていない）。停止のフラグで止めたら null
   */
  private static Outcomes learnOutcomes(
      GrpFiles files, PhiSetup setup, PhiEntry used, RecencyWindow window, int attempt, BooleanSupplier stop) {
    long started = System.nanoTime();
    GrpOperations operations = setup.selfGrp().grp();
    int slot = window.newest();
    try {
      OutcomeRows.Builder builder = OutcomeRows.builder(setup.variant(), window, GateCohort.NEXT, FinalRound::covers);
      long games = 0;
      for (int cohort = window.first(); cohort <= slot; cohort++) {
        if (!files.counted(cohort)) {
          LOG.warn("GRP refit at {}: cohort {} has no count file and is left out of the outcome rows", slot, cohort);
          continue;
        }
        CountedHands counted = CountFile.read(files.counts(cohort));
        builder.add(cohort, counted);
        games += counted.games();
      }
      OutcomeRows.Split rows = builder.build();
      if (rows.train().size() == 0 || rows.stop().size() == 0)
        return Outcomes.none("no TRAIN or STOP rows in " + games + " counted games of " + window);
      OutcomeNetwork start = operations.init() == GrpInit.WARM ? warmOutcomes(used, setup.variant()) : null;
      GrpInit init = start == null ? GrpInit.SCRATCH : GrpInit.WARM;
      long seed = SeedMixer.derive(setup.seed(), OUTCOME_SALT, slot, attempt);
      OutcomeTrained trained =
          OutcomeTrainer.train(
              outcomeShape(setup.variant()),
              start,
              rows.train(),
              rows.stop(),
              rule(operations.training(), init, seed),
              operations.threads(),
              stop);
      if (trained.ending() == TrainingEnding.STOPPED) return null;
      return new Outcomes(
          trained.network(),
          "",
          String.format(
              "q init=%s games=%d rows=%d stopRows=%d excluded=%d unexplained=%d halfEpochs=%d stopNll=%.6f clipped=%d"
                  + " seconds=%.1f",
              init,
              games,
              rows.train().size(),
              rows.stop().size(),
              rows.train().excluded() + rows.stop().excluded(),
              rows.train().unexplained() + rows.stop().unexplained(),
              trained.halfEpochs(),
              trained.stopNll(),
              trained.clippedSteps(),
              seconds(started)));
    } catch (IOException | RuntimeException failure) {
      LOG.warn("GRP refit at {} learned no outcome network: {}", slot, failure.toString(), failure);
      return Outcomes.none(failure.toString());
    }
  }

  /** q のウォームスタートの重み。使う項目が自己対局の候補で、同じ形の q を持てばその重みを返す。それ以外は null（ゼロから学ぶ）。 */
  private static OutcomeNetwork warmOutcomes(PhiEntry used, Variant variant) throws IOException {
    if (!used.candidate() || !used.hasOutcomes()) return null;
    OutcomeNetwork network = HandEndPhi.outcomes(used);
    return network.shape().equals(outcomeShape(variant)) ? network : null;
  }

  /**
   * 学び直した q。
   *
   * @param network 学んだ q（学べなかったら null）
   * @param path 書いた q のチェックポイントのディレクトリの絶対パス（書く前と、q を持たないときは空）
   * @param detail q の学習の要約（半荘や行の数など）か、q を持たない理由
   */
  private record Outcomes(OutcomeNetwork network, String path, String detail) {

    static Outcomes none(String why) {
      return new Outcomes(null, "", "q none: " + why);
    }

    /** q を書き、書いた先を持つ結果を返す。書けなければ q を持たない結果にする（試行は失敗にしない）。 */
    Outcomes write(Path directory) {
      if (network == null) return this;
      try {
        GrpCheckpoint.write(network, directory);
        return new Outcomes(network, directory.toAbsolutePath().normalize().toString(), detail);
      } catch (IOException | RuntimeException failure) {
        LOG.warn("GRP refit could not write the outcome network {}: {}", directory, failure.toString(), failure);
        return none("not written: " + failure);
      }
    }
  }

  /** 時点 {@code slot} の次の試行の番号。refits.jsonl と候補のディレクトリの名前で使った最大の番号に 1 を足す。 */
  private static int nextAttempt(GrpFiles files, List<RefitRecord> records, int slot) throws IOException {
    int last = 0;
    for (RefitRecord record : records) last = Math.max(last, attemptOf(record.id(), slot));
    Path candidates = files.candidate("x").getParent();
    if (Files.isDirectory(candidates)) {
      try (Stream<Path> names = Files.list(candidates)) {
        for (Path name : names.toList()) {
          String text = name.getFileName().toString();
          // 書きかけの一時ディレクトリ（.名前.writing）の番号も、使った番号に数える
          if (text.startsWith(".")) text = text.substring(1);
          int dot = text.indexOf('.');
          last = Math.max(last, attemptOf(dot < 0 ? text : text.substring(0, dot), slot));
        }
      }
    }
    return last + 1;
  }

  private static int attemptOf(String id, int slot) {
    Matcher matcher = PhiEntry.CANDIDATE.matcher(id);
    return matcher.matches() && Integer.parseInt(matcher.group(1)) == slot ? Integer.parseInt(matcher.group(2)) : 0;
  }

  private static double seconds(long started) {
    return (System.nanoTime() - started) / 1e9;
  }
}
