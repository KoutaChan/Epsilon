package com.epsilon.major.ai.decision.duel;

import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.TenhouLogSettings;
import com.epsilon.core.GameState;
import com.epsilon.engine.GameRecorder;
import com.epsilon.engine.Player;
import com.epsilon.engine.SynchronousGameRunner;
import com.epsilon.major.ai.decision.arena.EpsilonDecisionPlayer;
import com.epsilon.major.ai.decision.runtime.EpsilonDecisionEvaluator;
import com.epsilon.major.config.settings.EpsilonSettings;
import com.epsilon.training.TenhouLogWriter;
import com.epsilon.util.FormatUtils;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/** Decision プレイヤー同士の評価対局を実行し、必要に応じて天鳳互換の牌譜を出力する。 */
public final class EpsilonDecisionEvaluationRunner {

  private EpsilonDecisionEvaluationRunner() {}

  /** 全4席を同じ評価器で実行し、必要なら各半荘のTenhou互換ログを保存する。 */
  public static Result evaluate(EpsilonDecisionEvaluator evaluator, int games, Path tenhouLogDir)
      throws IOException {
    return evaluate(evaluator, games, tenhouLogDir, EpsilonSettings.defaults());
  }

  public static Result evaluate(
      EpsilonDecisionEvaluator evaluator, int games, Path tenhouLogDir, SettingsLoader config)
      throws IOException {
    if (games <= 0) {
      throw new IllegalArgumentException("games must be positive");
    }
    int[] lastScores = new int[GameState.NUM_PLAYERS];
    ArrayList<Path> tenhouLogFiles = new ArrayList<>();
    ArrayList<String> viewerUrls = new ArrayList<>();
    if (tenhouLogDir != null) {
      Files.createDirectories(tenhouLogDir);
    }
    for (int game = 0; game < games; game++) {
      TenhouLogWriter recorder =
          tenhouLogDir == null
              ? null
              : new TenhouLogWriter(playerNames(), config.bind(TenhouLogSettings.class));
      lastScores = play(evaluator, 10_000L + game, recorder, config);
      if (recorder != null) {
        Path file = tenhouLogDir.resolve("eval_" + FormatUtils.zeroPad(game + 1, 5) + ".json");
        Files.writeString(file, recorder.toViewerUrlAndJsonLines() + System.lineSeparator());
        tenhouLogFiles.add(file);
        viewerUrls.add(file.getFileName() + "\t" + recorder.toViewerUrl());
      }
    }
    if (tenhouLogDir != null) {
      Files.writeString(
          tenhouLogDir.resolve("viewer-urls.txt"),
          String.join(System.lineSeparator(), viewerUrls)
              + (viewerUrls.isEmpty() ? "" : System.lineSeparator()));
    }
    return new Result(games, lastScores, tenhouLogDir, tenhouLogFiles);
  }

  private static int[] play(
      EpsilonDecisionEvaluator evaluator, long seed, GameRecorder recorder, SettingsLoader config) {
    Player[] players = new Player[GameState.NUM_PLAYERS];
    for (int seat = 0; seat < players.length; seat++) {
      players[seat] = EpsilonDecisionPlayer.builder(evaluator, config).build();
    }
    SynchronousGameRunner runner =
        recorder == null
            ? new SynchronousGameRunner(players, seed)
            : new SynchronousGameRunner(players, seed, recorder);
    return runner.playHanchan();
  }

  private static String[] playerNames() {
    return new String[] {"eval-0", "eval-1", "eval-2", "eval-3"};
  }

  /** ニューラルネットワークによる評価対局の出力。 */
  public record Result(
      int games, int[] lastFinalScores, Path tenhouLogDir, List<Path> tenhouLogFiles) {

    /** 配列と一覧を防御複製する。 */
    public Result {
      if (games <= 0) {
        throw new IllegalArgumentException("evaluation games must be positive");
      }
      if (lastFinalScores.length != GameState.NUM_PLAYERS) {
        throw new IllegalArgumentException(
            "evaluation scores must contain " + GameState.NUM_PLAYERS + " seats");
      }
      lastFinalScores = lastFinalScores.clone();
      tenhouLogFiles = List.copyOf(tenhouLogFiles);
    }

    /** レコード内部のスコア配列を呼出側から変更できないよう、読出し時にも複製する。 */
    @Override
    public int[] lastFinalScores() {
      return lastFinalScores.clone();
    }
  }
}
