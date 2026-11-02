package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * config.json を読むだけの点検（check-config）。jar を差し替える前に、新しい jar で移し替えた config.json の写しを PrismConfig.read と
 * validateTraining で確かめ、Φ を解決する（FIXED_GRP は GRP の正確なディレクトリと sha、SELF_GRP は phi.jsonl を読んで反復 t の項目）。GPU を
 * 使わず、何も書かない。
 *
 * <p>反復 t は --run の state.json の次のコホート（--run がなければ 1）。SELF_GRP で phi.jsonl がまだなければ g000 の出どころ（phiStart）を、
 * あればその g000 が設定と合うか（{@link PhiHistory#requireInitialMatches}）を確かめる。GRP を読む項目は sha を計算し直して項目と照らし、CPU で
 * 網を開いて閉じる。較正の表は読んで形を確かめる。WITH_ORIGINAL なら影の元の GRP も解決する。誤りは書き残して状態 1 を返す（例外は投げない）。
 */
public final class CheckConfigWorkflow implements CommandWorkflow<Integer> {
  private static final Logger LOG = LoggerFactory.getLogger(CheckConfigWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "check-config", WorkflowKind.VALIDATION, "check-config --config FILE [--run DIR]", 2, 4);
  }

  @Override
  public Integer execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path file = options.path("config");
    String run = options.optionalText("run", "");
    options.requireEmpty();
    try {
      PhiEntry entry = check(file, run.isEmpty() ? null : new GrpFiles(Path.of(run)));
      LOG.info("check-config {}: ok, Φ {}", file, entry);
      return 0;
    } catch (IOException | RuntimeException failure) {
      LOG.error("check-config {}: {}", file, failure.toString());
      return 1;
    }
  }

  @Override
  public String summarize(Integer status) {
    return "status=" + status;
  }

  @Override
  public boolean successful(Integer status) {
    return status == 0;
  }

  /**
   * 設定を確かめ、反復 t の Φ の項目を返す。
   *
   * @param files run の置き場（null なら run なし）
   * @throws IOException ファイルが読めないか、GRP・較正の表・phi.jsonl が項目と合わないとき
   * @throws IllegalArgumentException 設定が読めないか、値が検証に落ちるとき
   */
  static PhiEntry check(Path file, GrpFiles files) throws IOException {
    PrismConfig config = PrismConfig.read(Files.readString(file, StandardCharsets.UTF_8));
    config.validateTraining();
    config.phiTemperatureTable();
    if (config.hasPhiCalibration()) PhiCalibration.load(Path.of(config.phiCalibration()));
    int t = files == null ? 1 : PhiHistory.nextIteration(files);
    PhiEntry entry =
        switch (config.phiSource()) {
          case FIXED_GRP -> opened(PhiScorer.original(config));
          case SCORE_RANK -> PhiEntry.scoreRank();
          case SCORE_SMOOTH -> PhiEntry.scoreSmooth();
          case SELF_GRP -> self(config, files, t);
        };
    if (config.phiShadowReference() == PhiShadowReference.WITH_ORIGINAL)
      LOG.info("check-config shadow {}", opened(PhiScorer.original(config)));
    return entry;
  }

  /** SELF_GRP の反復 t の項目（phi.jsonl がなければ g000 の出どころを確かめ、g000 として返す）。 */
  private static PhiEntry self(PrismConfig config, GrpFiles files, int t) throws IOException {
    List<PhiEntry> entries = files == null ? List.of() : PhiHistory.read(files);
    if (entries.isEmpty()) {
      LOG.info("check-config: no phi.jsonl yet, g000 comes from phiStart = {}", config.phiStart());
      if (config.phiStart().kind().scoreBased())
        return new PhiEntry(
            PhiEntry.INITIAL_ID, config.phiStart().kind(), "", "", "", "", 1, "", "phiStart = " + config.phiStart());
      PhiEntry original = opened(PhiScorer.original(config));
      return new PhiEntry(
          PhiEntry.INITIAL_ID,
          original.kind(),
          original.path(),
          original.sha256(),
          original.temperatures(),
          original.calibration(),
          1,
          "",
          "phiStart = " + config.phiStart());
    }
    PhiHistory.requireInitialMatches(files, entries.getFirst(), config);
    Optional<PhiEntry> effective = PhiHistory.effectiveAt(entries, t);
    if (effective.isEmpty()) throw new IOException("No Φ entry is effective at iteration " + t);
    PhiEntry entry = effective.get();
    if (entry.kind().scoreBased()) return entry;
    String sha = EpsilonGrpCheckpointManager.checkpointSha256(Path.of(entry.path()));
    if (!sha.equals(entry.sha256()))
      throw new IOException("Φ entry " + entry.id() + " expects " + entry.sha256() + " but " + entry.path() + " is " + sha);
    return opened(entry);
  }

  /** GRP の項目の網を CPU で開いて閉じる（読めることだけを確かめる）。 */
  private static PhiEntry opened(PhiEntry entry) throws IOException {
    try (Model model = EpsilonGrpCheckpointManager.load(Path.of(entry.path()), Device.cpu())) {
      LOG.info("check-config: opened {} ({})", entry.path(), model.getName());
    }
    return entry;
  }
}
