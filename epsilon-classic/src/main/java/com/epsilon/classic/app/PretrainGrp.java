package com.epsilon.classic.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Signals;
import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.data.RecordDirectory;
import com.epsilon.classic.learn.GrpPretrain;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.LogGames;
import com.epsilon.grp.train.TrainingEnding;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.function.BooleanSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 牌譜のディレクトリにある半荘で GRP をゼロから学習し（{@link GrpPretrain}）、epsilon-grp の checkpoint（grp.json と weights.f32）を
 * 書く。学び方は全体の設定の {@code grp.training} で決める。ルールの違う牌譜と読めない牌譜は飛ばし、その数を数える。停止の要求（SIGTERM
 * か SIGINT）を受けたら、何も書かずに {@link Exit#STOPPED} で終わる。
 *
 * @param out 書く checkpoint のディレクトリ。既にあれば牌譜を読まず、使い方の誤りとして扱う
 */
@Named(value = "pretrain-grp", doc = "Train a GRP from scratch on the games of the records and write a GRP checkpoint")
public record PretrainGrp(
    @Key(doc = "Records directory (files directly inside, read in name order)", presence = Presence.REQUIRED) Path records,
    @Key(doc = "GRP checkpoint directory to write (must not exist yet)", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Rules", value = "TENHOU_4P") Variant variant,
    @Key(doc = "Torch op threads", value = "1") @Range(min = 1) int threads,
    @Key(doc = "Random seed for shuffling and for the initial weights from scratch", value = "1") long seed)
    implements ClassicCommand {

  private static final Logger LOG = LoggerFactory.getLogger(PretrainGrp.class);

  /** コマンドの結果。GRP の学習の結果、使った半荘の数、飛ばした牌譜の数を持つ。 */
  public record Result(GrpPretrain.Result training, int games, int skipped) {}

  @Override
  public Outcome run(Context<ClassicSettings> context) throws IOException {
    Signals signals = context.signals();
    Result result = train(context.settings().value().grp().training(), signals::requested);
    GrpPretrain.Result training = result.training();
    String summary =
        String.format(
            Locale.ROOT,
            "games=%d skipped=%d rows=%d stop=%d uniform=%.4f frequency=%.4f trained=%.4f half-epochs=%d",
            result.games(), result.skipped(), training.trainRows(), training.stopRows(), training.uniformNll(),
            training.frequencyNll(), training.trained().stopNll(), training.trained().halfEpochs());
    return training.trained().ending() == TrainingEnding.STOPPED
        ? new Outcome(Exit.STOPPED, summary)
        : Outcome.ok(summary);
  }

  /**
   * 牌譜を読んで学習し、最後まで学べたら checkpoint を書く。
   *
   * @param training 学び方
   * @param closing 停止のフラグ。学習の各ステップの前に確かめる
   * @throws SettingsException 書く checkpoint が既にあるとき
   */
  public Result train(GrpTraining training, BooleanSupplier closing) throws IOException {
    Inputs.requireNew(out, "output checkpoint");
    Inputs.requireDirectory(records, "records");
    List<GameRecord> games = new ArrayList<>();
    int skipped = 0;
    for (Path file : RecordDirectory.files(records)) {
      try {
        RecordedGame game = RecordDirectory.read(file);
        if (game.variant() != variant) {
          throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + game.variant());
        }
        games.add(LogGames.of(game));
      } catch (IllegalArgumentException unusable) {
        if (skipped++ == 0) {
          LOG.warn("pretrain-grp skip {}: {}", file, unusable.toString());
        }
      }
    }
    GrpPretrain.Result result = GrpPretrain.train(variant, games, training, seed, threads, closing);
    if (result.trained().ending() != TrainingEnding.STOPPED) {
      GrpCheckpoint.write(result.trained().network(), out);
    }
    return new Result(result, games.size(), skipped);
  }
}
