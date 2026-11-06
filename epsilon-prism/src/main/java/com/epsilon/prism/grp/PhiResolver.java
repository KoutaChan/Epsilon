package com.epsilon.prism.grp;

import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.HandValue;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 学習の JVM の SELF_GRP の Φ（{@code TrainingResources} が持つ）。反復の始めに AUTO の切り替えを入れてから phi.jsonl を 1 回読み、効く項目が
 * 開いている項目と違えば新しい {@link GrpPhi} を作る（GrpPhi のメモは接頭辞だけで引くので、推論だけを差し替えると古い値を返す）。GRP は項目の
 * 正確なパスと sha で読む。
 * 順伝播の行の数と GrpPhi の引数は FIXED_GRP と同じなので、g000 = 元の GRP ＋ 表のラベルは FIXED_GRP とビットで同じ。
 *
 * <p>失敗の扱い: 反復 t で初めて効く自己対局の候補が読めない・sha が違う・READY でない・境目の点検（{@link BoundaryCheck}）に落ちるなら、
 * refits.jsonl に REJECTED を書き、phi.jsonl に前の項目を反復 t から効かせる行を足して続ける（再開しても同じ歴史）。既に効いたことのある項目が
 * 読めなければ止める（黙って別の Φ に落ちない）。
 *
 * <p>開いているあいだ、学習の JVM の印の錠（{@link PhiHistory#firstOpenIteration}）を持つ。
 */
public final class PhiResolver implements AutoCloseable {
  private static final Logger LOG = LoggerFactory.getLogger(PhiResolver.class);

  /** Φ のメモの上限（FIXED_GRP の {@code TrainingResources} と同じ）。 */
  private static final int PHI_MEMO = 1 << 16;

  /** GRP の順伝播の行の数（{@code EpsilonGrpTrainingSession.openInferenceOnly} と同じ設定の値）。 */
  private static final int MAX_BATCH =
      EpsilonSettings.defaults().bind(GrpInferenceSettings.class).maxBatch();

  private final PrismConfig config;
  private final EpsilonUtilityProfile profile;
  private final Consumer<Runnable> driver;
  private final GrpFiles files;
  private final PhiHistory.Locked training;

  /** 開いている項目と値（まだ開いていなければ null）。 */
  private Opened opened;

  private PhiResolver(
      PrismConfig config,
      EpsilonUtilityProfile profile,
      Consumer<Runnable> driver,
      GrpFiles files,
      PhiHistory.Locked training) {
    this.config = config;
    this.profile = profile;
    this.driver = driver;
    this.files = files;
    this.training = training;
  }

  /**
   * phi.jsonl を読み（なければ g000 を書き。{@link PhiHistory#initialize}）、まだ何も開かない。
   *
   * @param profile 順位効用
   * @param driver GrpPhi の completion を回すドライバのスレッド
   * @throws IOException phi.jsonl が読めないか書けないとき、ほかの学習の JVM が同じ run を開いているとき
   */
  public static PhiResolver open(
      PrismConfig config, EpsilonUtilityProfile profile, Consumer<Runnable> driver, GrpFiles files)
      throws IOException {
    PhiHistory.Locked training = PhiHistory.holdTraining(files);
    try {
      PhiEntry initial = PhiHistory.initialize(files, config);
      LOG.info("Φ history {} starts with {}", files.phiHistory(), initial);
      return new PhiResolver(config, profile, driver, files, training);
    } catch (IOException | RuntimeException failure) {
      try {
        training.close();
      } catch (IOException closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  /**
   * 反復 t の Φ。grpSwitch = AUTO なら先に合格済みの候補を反復 t から入れ（{@link AutoSwitch}）、項目が変わったら前の GrpPhi を閉じて作り直す。
   *
   * @throws IOException 既に効いたことのある項目の GRP が読めないとき（系譜を止める）
   */
  public ActivePhi resolve(int t) throws IOException {
    AutoSwitch.apply(files, config, t).ifPresent(entry -> LOG.info("t={} AUTO switch to {}", t, entry));
    return resolveEffective(t);
  }

  /** 反復 t に効く項目の Φ（境目の点検で拒んだら、前の項目を効かせ直してもう一度）。 */
  private ActivePhi resolveEffective(int t) throws IOException {
    List<PhiEntry> entries = PhiHistory.readLocked(files);
    PhiEntry current =
        PhiHistory.effectiveAt(entries, t)
            .orElseThrow(() -> new IOException("No Φ entry is in effect at iteration " + t + " in " + files.phiHistory()));
    if (opened == null || !opened.entry.sameValue(current)) {
      boolean fresh = fresh(entries, current, t);
      Opened next;
      try {
        next = open(current, fresh);
      } catch (IOException | RuntimeException failure) {
        if (!fresh)
          throw new IOException("Φ entry " + current.id() + " in effect at iteration " + t + " cannot be opened", failure);
        reject(entries, current, t, failure);
        return resolveEffective(t);
      }
      if (opened != null) opened.close();
      opened = next;
      LOG.info("t={} Φ is now {}", t, current);
    }
    boolean resetsLedger =
        PhiHistory.effectiveAt(entries, t - 1)
            .map(previous -> previous.kind().scoreBased() && !current.kind().scoreBased())
            .orElse(false);
    return new ActivePhi(current, opened.value, resetsLedger);
  }

  /** 開いている GrpPhi と GRP を閉じ、学習の JVM の印の錠を外す。 */
  @Override
  public void close() {
    try {
      if (opened != null) opened.close();
      opened = null;
    } finally {
      try {
        training.close();
      } catch (IOException failure) {
        LOG.warn("could not release {}: {}", files.lock(PhiHistory.TRAINING_LOCK), failure.toString());
      }
    }
  }

  /** current が反復 t で初めて効く自己対局の候補（同じ id の行が t より前に効いていない）なら true。 */
  private static boolean fresh(List<PhiEntry> entries, PhiEntry current, int t) {
    if (current.kind() != PhiEntryKind.SELF_CHECKPOINT) return false;
    for (PhiEntry entry : entries)
      if (entry.id().equals(current.id()) && entry.effectiveFrom() < t) return false;
    return true;
  }

  /** 項目を開く。fresh なら READY であること・sha・境目の点検も確かめる。 */
  private Opened open(PhiEntry entry, boolean fresh) throws IOException {
    if (entry.kind().scoreBased()) return new Opened(entry, null, ScorePhi.handValue(entry.kind(), profile));
    if (fresh) {
      Optional<RefitRecord> record = RefitRecord.latest(RefitRecord.readAll(files), entry.id());
      if (record.isEmpty() || record.get().status() != RefitStatus.READY)
        throw new IOException("Candidate " + entry.id() + " is not READY: " + record.map(RefitRecord::status).orElse(null));
    }
    Path directory = Path.of(entry.path());
    String sha = EpsilonGrpCheckpointManager.checkpointSha256(directory);
    if (!sha.equals(entry.sha256()))
      throw new IOException("GRP checkpoint " + directory + " has sha " + sha + ", not " + entry.sha256());
    Model model = EpsilonGrpCheckpointManager.load(directory, PrismConfig.device(config.grpDevice()));
    try {
      var inference = new EpsilonGrpInference(model, MAX_BATCH);
      if (fresh) {
        Path boundary = files.boundary(entry.id());
        if (!Files.exists(boundary)) throw new IOException("No boundary check for " + entry.id() + ": " + boundary);
        if (!BoundaryCheck.read(boundary).matches(inference, profile))
          throw new IOException("Candidate " + entry.id() + " does not reproduce its boundary check within " + BoundaryCheck.TOLERANCE);
      }
      boolean original = entry.kind() == PhiEntryKind.FIXED_CHECKPOINT;
      HandValue value =
          new GrpPhi(
              inference,
              profile,
              driver,
              config.grpBatch(),
              PHI_MEMO,
              original ? PrismConfig.temperatures(entry.temperatures()) : new double[0],
              original && !entry.calibration().isBlank() ? PhiCalibration.load(Path.of(entry.calibration())) : null);
      return new Opened(entry, model, value);
    } catch (IOException | RuntimeException failure) {
      model.close();
      throw failure;
    }
  }

  /**
   * 反復 t で初めて効く候補を拒む: refits.jsonl に REJECTED、phi.jsonl に前の項目を反復 t から効かせる行。refits.jsonl が読めないか書けなくても
   * phi.jsonl の行は必ず書く（歴史と使った Φ が一致していれば系譜は続けられる）。
   */
  private void reject(List<PhiEntry> entries, PhiEntry candidate, int t, Throwable failure) throws IOException {
    PhiEntry previous =
        PhiHistory.effectiveAt(entries, t - 1)
            .orElseThrow(() -> new IOException("Φ entry " + candidate.id() + " cannot be opened and nothing precedes it", failure));
    String detail = "boundary at iteration " + t + ": " + failure;
    RefitRecord rejected =
        new RefitRecord(candidate.id(), RefitStatus.REJECTED, null, null, null, candidate.parent(), "", "", null, detail, 0);
    try {
      RefitRecord.latest(RefitRecord.readAll(files), candidate.id())
          .map(ready -> ready.withStatus(RefitStatus.REJECTED, detail))
          .orElse(rejected)
          .append(files);
    } catch (IOException | RuntimeException unreadable) {
      LOG.warn("t={} could not read or extend {}: {}", t, files.refits(), unreadable.toString());
      try {
        rejected.append(files);
      } catch (IOException | RuntimeException unwritable) {
        LOG.warn("t={} could not write the REJECTED line of {}: {}", t, candidate.id(), unwritable.toString());
      }
    }
    PhiHistory.appendAtIterationStart(files, previous.from(t, "rejected " + candidate.id() + " at iteration " + t));
    LOG.warn("t={} rejected Φ candidate {} and kept {}: {}", t, candidate.id(), previous.id(), failure.toString());
  }

  /** 開いた項目と、その値と GRP（点数だけの Φ は null）。閉じるのは値、GRP の順。 */
  private record Opened(PhiEntry entry, Model model, HandValue value) implements AutoCloseable {
    @Override
    public void close() {
      try (Model grp = model;
          HandValue phi = value) {
        // 値（GrpPhi の評価のスレッド）を先に止めてから GRP を閉じる
      }
    }
  }
}
