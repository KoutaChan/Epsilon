package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Optional;
import java.util.function.BooleanSupplier;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GRP の学習 1 回（学習の JVM の {@link GrpBackground} のスレッドで呼ぶ）: 取り置き h の窓（{@link RefitWindow}）の対局の記録で GRP を学習し
 * （{@link GrpGameTrainer}。使用中が GRP なら grpInit、点数だけの Φ なら白紙）、取り置きのコホートで関門を通し（{@link GateEvaluator}。列は
 * 使用中の種類の {@link GateMode}）、合格なら候補を公開する: 試行ごとの名前で staging に保存 → fsync → candidates へ rename → 境目の点検の
 * 見本（{@link BoundaryCheck}）→ refits.jsonl に READY。同じ名前の再試行はしない（保存は既存を拒む）。締め切り（{@link
 * GrpTrainingPlan#deadline}）を過ぎたら止めて TIMEOUT を書く。失敗は全て refits.jsonl の行にし、系譜は止めない。止める合図が立てば試行を
 * 捨て（書きかけの候補と見本を消す）、refits.jsonl には何も書かない。
 *
 * <p>使用中の項目は phi.jsonl の次の反復（state.json の次のコホート）に効く項目。窓の半荘の 1/{@value #VALIDATION_MODULUS}（半荘 ID で決まる。
 * {@link Split}）は学習の早めの打ち切りの検証に回し、取り置きのコホートは学習にも検証にも使わない。候補の形は、checkpoint の構造が grp-epsilon-v3 で、境目の
 * 見本の局末で順位分布が有限・席ごとと順位ごとの和が 1（{@value #MARGINAL_SUM_ERROR} 以内）であること。
 */
public final class GrpRefitter {
  private static final Logger LOG = LoggerFactory.getLogger(GrpRefitter.class);

  /** 半荘 ID をこれで割った余りが最後の値の半荘を検証に回す。 */
  static final int VALIDATION_MODULUS = 16;

  /** 順位分布の席ごと・順位ごとの和の 1 からのずれの上限（{@code EpsilonGrpValidationGate} と同じ）。 */
  static final double MARGINAL_SUM_ERROR = 1e-5;

  private static final Pattern ATTEMPT = Pattern.compile("g(\\d+)-a(\\d+)");

  private GrpRefitter() {}

  /**
   * 学習と早めの打ち切りの検証に分けた半荘。半荘 ID を 1 つ右にずらした値を {@value #VALIDATION_MODULUS} で割った余りが最後の値の半荘を
   * 検証に回す。
   *
   * @param training 学習の半荘
   * @param validation 検証の半荘
   */
  record Split(List<RecordedGame> training, List<RecordedGame> validation) {
    static Split of(List<RecordedGame> games) {
      List<RecordedGame> training = new ArrayList<>(), validation = new ArrayList<>();
      for (RecordedGame game : games)
        (Math.floorMod(game.gameId() >> 1, VALIDATION_MODULUS) == VALIDATION_MODULUS - 1 ? validation : training).add(game);
      return new Split(training, validation);
    }
  }

  /**
   * 取り置き h で 1 回学習する。
   *
   * @param holdout 取り置きのコホート h（封印済みで対局の記録があること）
   * @param stop 止める合図（学習の毎歩の前と関門の前に見る）
   * @return 書いた refits.jsonl の行（止める合図で捨てたら空）
   * @throws IOException refits.jsonl が書けないとき
   */
  public static Optional<RefitRecord> refit(
      GrpFiles files, PrismConfig config, int holdout, RefitCause cause, BooleanSupplier stop) throws IOException {
    return refit(files, config, holdout, cause, stop, Steps.PRODUCTION);
  }

  /** 試行の id（g&lt;k&gt;-a&lt;n&gt;）。 */
  static String attemptId(int holdout, int attempt) {
    return String.format("g%03d-a%d", holdout, attempt);
  }

  /** 学習の部品（試験で差し替える）。 */
  interface Steps {
    /** 本番の部品。 */
    Steps PRODUCTION =
        new Steps() {
          @Override
          public CohortGames games(Path file) throws IOException {
            return GameLog.read(file);
          }

          @Override
          public GrpTrainingPlan plan(GrpInit init, long seed) {
            return GrpTrainingPlan.of(init, seed);
          }

          @Override
          public GrpGameTrainer.Trained train(
              Path start,
              List<RecordedGame> training,
              List<RecordedGame> validation,
              GrpTrainingPlan plan,
              BooleanSupplier stop,
              Path out)
              throws IOException {
            return GrpGameTrainer.train(start, training, validation, plan, stop, out);
          }

          @Override
          public HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException {
            return HandEndPhi.of(entry, profile);
          }

          @Override
          public boolean grpEpsilonV3(Path checkpoint) {
            try {
              return EpsilonGrpCheckpointManager.loadManifest(checkpoint).architecture.startsWith(ARCHITECTURE);
            } catch (IOException invalid) {
              return false;
            }
          }

          @Override
          public String sha256(Path checkpoint) throws IOException {
            return EpsilonGrpCheckpointManager.checkpointSha256(checkpoint);
          }

          @Override
          public GateReport gate(GateInput input, EpsilonUtilityProfile profile) throws IOException {
            return GateEvaluator.evaluate(input, profile);
          }
        };

    /** GRP の構造の名前の頭。 */
    String ARCHITECTURE = EpsilonGrpNetwork.architectureSummary().split(" ")[0];

    CohortGames games(Path file) throws IOException;

    GrpTrainingPlan plan(GrpInit init, long seed);

    GrpGameTrainer.Trained train(
        Path start,
        List<RecordedGame> training,
        List<RecordedGame> validation,
        GrpTrainingPlan plan,
        BooleanSupplier stop,
        Path out)
        throws IOException;

    HandEndPhi open(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException;

    boolean grpEpsilonV3(Path checkpoint);

    String sha256(Path checkpoint) throws IOException;

    GateReport gate(GateInput input, EpsilonUtilityProfile profile) throws IOException;
  }

  static Optional<RefitRecord> refit(
      GrpFiles files, PrismConfig config, int holdout, RefitCause cause, BooleanSupplier stop, Steps steps)
      throws IOException {
    long started = System.nanoTime();
    var profile = EpsilonUtilityProfile.valueOf(config.utility());
    int attempt = nextAttempt(files, RefitRecord.readAll(files), holdout);
    String id = attemptId(holdout, attempt);
    RefitWindow window = RefitWindow.ending(holdout, config.grpWindow()).orElse(null);
    Path staging = files.staging().resolve(id);
    Path candidate = files.candidate(id);
    var draft = new Draft(id, cause, window, config.grpInit(), started);
    Optional<RefitRecord> outcome;
    try {
      outcome = attempt(files, config, holdout, profile, stop, steps, draft, staging, candidate, attempt);
    } catch (IOException | RuntimeException failure) {
      LOG.warn("GRP refit {} failed: {}", id, failure.toString(), failure);
      outcome = Optional.of(draft.record(RefitStatus.FAILED, "", "", null, failure.toString()));
    }
    if (outcome.map(record -> record.status() != RefitStatus.READY).orElse(true)) {
      deleteQuietly(staging);
      deleteQuietly(candidate);
      deleteQuietly(files.boundary(id));
    }
    if (outcome.isEmpty()) {
      LOG.info("GRP refit {} stopped after {}s", id, String.format("%.1f", (System.nanoTime() - started) / 1e9));
      return outcome;
    }
    RefitRecord record = outcome.get();
    record.append(files);
    LOG.info("GRP refit {} {} in {}s: {}", id, record.status(), String.format("%.1f", record.seconds()), record.detail());
    return outcome;
  }

  /** 試行の本体（止める合図で捨てたら空）。 */
  private static Optional<RefitRecord> attempt(
      GrpFiles files,
      PrismConfig config,
      int holdout,
      EpsilonUtilityProfile profile,
      BooleanSupplier stop,
      Steps steps,
      Draft draft,
      Path staging,
      Path candidate,
      int attempt)
      throws IOException {
    PhiEntry active =
        PhiHistory.effectiveAt(PhiHistory.read(files), PhiHistory.nextIteration(files))
            .orElseThrow(() -> new IOException("No Φ entry in " + files.phiHistory()));
    draft.parent = active.id();
    draft.init = active.kind().scoreBased() ? GrpInit.SCRATCH : config.grpInit();
    if (draft.window == null)
      return Optional.of(draft.record(RefitStatus.FAILED, "", "", null, "no usable cohort before holdout " + holdout));
    if (!recorded(files, holdout))
      return Optional.of(
          draft.record(RefitStatus.FAILED, "", "", null, "holdout " + holdout + " is not sealed with a game log"));
    List<RecordedGame> held = steps.games(files.games(holdout)).games();
    List<RecordedGame> window = new ArrayList<>();
    for (int c = draft.window.first(); c <= draft.window.last(); c++) {
      if (recorded(files, c)) window.addAll(steps.games(files.games(c)).games());
      else LOG.warn("GRP refit {}: cohort {} has no sealed game log and is left out", draft.id, c);
    }
    if (window.isEmpty() || held.isEmpty())
      return Optional.of(draft.record(RefitStatus.FAILED, "", "", null, "no recorded games in window " + draft.window));
    Split split = Split.of(window);
    long seed = SeedMixer.derive(config.seed(), 0x67727066L, holdout, attempt);
    GrpTrainingPlan plan = steps.plan(draft.init, seed);
    Files.createDirectories(files.staging());
    GrpGameTrainer.Trained trained =
        steps.train(
            draft.init == GrpInit.WARM ? Path.of(active.path()) : null,
            split.training(),
            split.validation(),
            plan,
            stop,
            staging);
    switch (trained.ending()) {
      case STOPPED -> {
        return Optional.empty();
      }
      case TIMED_OUT -> {
        return Optional.of(
            draft.record(
                RefitStatus.TIMEOUT, "", "", null, "deadline " + plan.deadline() + " after " + trained.epochs() + " epochs"));
      }
      case COMPLETED -> {}
    }
    var probe =
        new PhiEntry(
            draft.id, PhiEntryKind.SELF_CHECKPOINT, trained.directory().toString(), trained.sha256(), "", "", 1, active.id(), "staging");
    List<HandEndInput> inputs = BoundaryCheck.inputs(held, seed);
    BoundaryCheck boundary;
    boolean valid;
    try (HandEndPhi phi = steps.open(probe, profile)) {
      PhiValues values = phi.values(inputs);
      valid = steps.grpEpsilonV3(trained.directory()) && wellFormed(values);
      boundary = new BoundaryCheck(inputs, values.utilities());
    }
    if (stop.getAsBoolean()) return Optional.empty();
    GateReport gate;
    try (HandEndPhi used = steps.open(active, profile);
        HandEndPhi proposed = steps.open(probe, profile)) {
      gate =
          steps.gate(
              new GateInput(
                  GateMode.of(active.kind()), held, used, proposed, valid, window.size(), config.grpTrustGames()),
              profile);
    }
    if (!gate.passed()) return Optional.of(draft.record(RefitStatus.REJECTED, "", "", gate, "gate " + gate.result()));
    // 公開: fsync → rename → 見本 → READY（学習の JVM は READY の行の候補しか読まない）
    syncTree(trained.directory());
    Files.createDirectories(files.candidates());
    Files.move(trained.directory(), candidate, StandardCopyOption.ATOMIC_MOVE);
    syncDirectory(files.candidates());
    String sha = steps.sha256(candidate);
    if (!sha.equals(trained.sha256()))
      throw new IOException("Candidate " + candidate + " has sha " + sha + " after the rename, not " + trained.sha256());
    boundary.write(files.boundary(draft.id));
    return Optional.of(
        draft.record(
            RefitStatus.READY,
            candidate.toAbsolutePath().normalize().toString(),
            sha,
            gate,
            String.format(
                "epochs=%d steps=%d trainLoss=%.6f validationLoss=%.6f games/s=%.1f window=%d",
                trained.epochs(),
                trained.steps(),
                trained.trainLoss(),
                trained.validationLoss(),
                trained.gamesPerSecond(),
                window.size())));
  }

  /** コホート c が封印済みで対局の記録があるなら true。 */
  static boolean recorded(GrpFiles files, int c) {
    return Files.exists(files.cohort(c).resolve(CohortWriter.MANIFEST)) && Files.exists(files.games(c));
  }

  /** 見本の局末の Φ と順位分布が有限で、順位分布の席ごと・順位ごとの和が 1 なら true。 */
  static boolean wellFormed(PhiValues values) {
    for (int i = 0; i < values.size(); i++) {
      double[] probabilities = values.rankProbabilities()[i];
      double[] utilities = values.utilities()[i];
      if (probabilities.length != 16 || utilities.length != 4) return false;
      for (double utility : utilities) if (!Double.isFinite(utility)) return false;
      for (int a = 0; a < 4; a++) {
        double seat = 0, rank = 0;
        for (int b = 0; b < 4; b++) {
          double p = probabilities[4 * a + b], q = probabilities[4 * b + a];
          if (!Double.isFinite(p) || p < 0 || p > 1) return false;
          seat += p;
          rank += q;
        }
        if (Math.abs(seat - 1) > MARGINAL_SUM_ERROR || Math.abs(rank - 1) > MARGINAL_SUM_ERROR) return false;
      }
    }
    return true;
  }

  /** 取り置き h の次の試行の番号（refits.jsonl と候補・書きかけ・見本の名前で使った番号の次）。 */
  private static int nextAttempt(GrpFiles files, List<RefitRecord> records, int holdout) throws IOException {
    int last = 0;
    for (RefitRecord record : records) last = Math.max(last, attemptOf(record.id(), holdout));
    for (Path directory : List.of(files.candidates(), files.staging(), files.boundary("x").getParent())) {
      if (!Files.isDirectory(directory)) continue;
      try (Stream<Path> names = Files.list(directory)) {
        for (Path name : names.toList()) {
          String text = name.getFileName().toString();
          int dot = text.indexOf('.');
          last = Math.max(last, attemptOf(dot < 0 ? text : text.substring(0, dot), holdout));
        }
      }
    }
    return last + 1;
  }

  private static int attemptOf(String id, int holdout) {
    Matcher matcher = ATTEMPT.matcher(id);
    return matcher.matches() && Integer.parseInt(matcher.group(1)) == holdout ? Integer.parseInt(matcher.group(2)) : 0;
  }

  /** ディレクトリの中のファイルとディレクトリ自身を fsync する。 */
  private static void syncTree(Path directory) throws IOException {
    try (Stream<Path> paths = Files.walk(directory)) {
      for (Path path : paths.toList()) {
        if (Files.isRegularFile(path)) {
          try (FileChannel channel = FileChannel.open(path, StandardOpenOption.WRITE)) {
            channel.force(true);
          }
        }
      }
    }
    syncDirectory(directory);
  }

  /** ディレクトリの項目を fsync する（開けない系では何もしない）。 */
  private static void syncDirectory(Path directory) {
    try (FileChannel channel = FileChannel.open(directory, StandardOpenOption.READ)) {
      channel.force(true);
    } catch (IOException unsupported) {
      // ディレクトリを開けない系（Windows）では rename の永続化を系に任せる
    }
  }

  private static void deleteQuietly(Path path) {
    try {
      if (Files.exists(path)) Snapshot.deleteTree(path);
    } catch (IOException failure) {
      LOG.warn("could not delete {}: {}", path, failure.toString());
    }
  }

  /** 試行の共通の欄（使用中の項目が分かれば parent と init を埋める）。 */
  private static final class Draft {
    final String id;
    final RefitCause cause;
    final RefitWindow window;
    final long started;
    GrpInit init;
    String parent = "";

    Draft(String id, RefitCause cause, RefitWindow window, GrpInit init, long started) {
      this.id = id;
      this.cause = cause;
      this.window = window;
      this.init = init;
      this.started = started;
    }

    RefitRecord record(RefitStatus status, String path, String sha256, GateReport gate, String detail) {
      return new RefitRecord(
          id, status, cause, window, init, parent, path, sha256, gate, detail, (System.nanoTime() - started) / 1e9);
    }
  }
}
