package com.epsilon.prism.play;

import com.epsilon.client.riichi.RiichiClient;
import com.epsilon.config.settings.RiichiSettings;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.prism.run.RunFiles;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.StandardWorkflowDefinitions;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開した版で RiichiLab のランク戦を打つ（{@link RiichiClient}）。手は版の s の貪欲（{@link GreedyPlayer}）。
 *
 * <p>設定は [epsilon.riichi]（{@link RiichiSettings}）: token は必須、checkpointDir は版のディレクトリ（snapshots/&lt;t&gt;）か run のディレクトリ
 * （その最新の版）、maxGames は完了する対局の数（0 なら止めるまで）。device は epsilon.devices.inference（CPU は FP32 の eager）。接続の前に白紙の
 * 半荘の最初の判断を 1 回採点して、最初の手の遅れを対局の外に出す。
 */
public final class RiichiPlayWorkflow implements CommandWorkflow<RiichiPlayWorkflow.Result> {
  private static final Logger LOG = LoggerFactory.getLogger(RiichiPlayWorkflow.class);

  @Override
  public WorkflowDefinition definition() {
    return StandardWorkflowDefinitions.PLAY_RIICHI;
  }

  @Override
  public Result execute(WorkflowArguments arguments) throws Exception {
    SettingsLoader settings = arguments.settings();
    RiichiSettings riichi = settings.bind(RiichiSettings.class);
    if (riichi.token().isBlank()) {
      throw new IllegalArgumentException("Set epsilon.riichi.token in the settings before play-riichi");
    }
    Path snapshot = snapshot(Path.of(riichi.checkpointDir()));
    try (SnapshotScorer scorer = SnapshotScorer.open(snapshot, settings)) {
      warmUp(scorer);
      LOG.info("RiichiLab snapshot {} ready: maxGames={}", snapshot, riichi.maxGames());
      return new Result(
          snapshot,
          RiichiClient.play(riichi.token(), riichi.maxGames(), () -> new GreedyPlayer(scorer)));
    }
  }

  /** checkpointDir の版: 版のディレクトリ（{@value Snapshot#POLICY} がある）ならそのもの、そうでなければ run のディレクトリとみて最新の版。 */
  static Path snapshot(Path directory) throws IOException {
    if (Files.isRegularFile(directory.resolve(Snapshot.POLICY))) return directory;
    return RunFiles.latestSnapshot(directory);
  }

  /** 白紙の半荘の最初の判断（親の第一打）を 1 回採点する。 */
  static void warmUp(SnapshotScorer scorer) {
    GameEngine engine = new GameEngine(0L);
    var awaiting = (GameStepResult.AwaitingDecisions) engine.stepHanchan();
    EngineDecisionPoint decision = awaiting.decisions().getFirst();
    new GreedyPlayer(scorer).selectAction(engine.getState(), decision.player(), decision.legalActions());
  }

  @Override
  public String summarize(Result result) {
    return "snapshot="
        + result.snapshot()
        + " completedGames="
        + result.play().completedGames()
        + " interruptedGames="
        + result.play().interruptedGames();
  }

  /** 打った版と対局の数。 */
  public record Result(Path snapshot, RiichiClient.Result play) {}
}
