package com.epsilon.classic.app;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.Signals;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.data.RowChunks;
import com.epsilon.classic.learn.Pretrainer;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Locale;
import java.util.function.BooleanSupplier;

/**
 * 学習の行のディレクトリ（{@link RowChunks}）で Nano・Major を事前学習し（{@link Pretrainer}）、最良のエポックの重みを checkpoint に
 * 書く。checkpoint の学習のステップ数は、初期値に今回のステップ数を足したもの。学習の記録（基準の値とエポックごとの値）は
 * {@code <out>.pretrain.json} に書く。
 *
 * <p>区切って学ぶこともできる。途中の状態のディレクトリを渡すと、停止のフラグ（分の上限、区切りのファイル、SIGTERM か SIGINT）が
 * 立ったときは次のステップに入る前に、またエポックの終わりごとに状態を書く。フラグで止めた呼び出しは、checkpoint を書かずに
 * {@link Exit#STOPPED} で終わる。同じ引数でもう一度呼ぶと続きから学ぶ。続きで変えてよいのはエポック数だけ。状態のディレクトリが
 * 無ければ、フラグが立つと何も書かずに終わる。
 *
 * @param from 始めの重みの checkpoint。{@code architecture} とどちらか一方だけを書く
 * @param architecture ゼロから学習するネットワークのアーキテクチャ。{@code from} とどちらか一方だけを書く
 * @param grp 価値の事前分布に使う GRP（epsilon-grp の checkpoint）。無ければ一様分布
 * @param threads CPU の演算のスレッド数。無ければ変えない
 * @param state 途中の状態のディレクトリ。無ければ区切らない
 * @param minutes 区切るまでの分数。エポックの終わりの検証の時間は数えないので、GPU を使える時間より短めにする
 * @param pauseFile 現れたら区切るファイル
 */
@Named(value = "pretrain", doc = "Train Nano or Major on training rows and write the weights of the best epoch to a checkpoint")
public record Pretrain(
    @Key(doc = "Training rows directory", presence = Presence.REQUIRED) Path rows,
    @Key(doc = "Checkpoint directory to write (must not exist yet)", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Checkpoint of the starting weights (exactly one of this and architecture)", presence = Presence.ABSENT) Path from,
    @Key(doc = "Architecture of a network trained from scratch (exactly one of this and from)", presence = Presence.ABSENT) Architecture architecture,
    @Key(doc = "GRP checkpoint for the value prior (uniform if absent)", presence = Presence.ABSENT) Path grp,
    @Key(doc = "Number of epochs", value = "4") @Range(min = 1) int epochs,
    @Key(doc = "Maximum rows per step", value = "1024") @Range(min = 1) int batch,
    @Key(doc = "Learning rate", value = "2e-4") @Range(min = 0, minInclusive = false) double learningRate,
    @Key(doc = "AdamW weight decay", value = "0.01") @Range(min = 0) double weightDecay,
    @Key(doc = "Weight of the value loss", value = "0.5") @Range(min = 0) double valueWeight,
    @Key(doc = "Loss weight of decisions that can declare riichi and of responses to other players (other decisions 1)", value = "2.0") @Range(min = 0, minInclusive = false)
        double emphasis,
    @Key(doc = "Limit of the global gradient norm", value = "10.0") @Range(min = 0, minInclusive = false) double clipNorm,
    @Key(doc = "Device (cpu or gpuN)", value = "cpu") String device,
    @Key(doc = "Compute type", value = "FLOAT32") Precision precision,
    @Key(doc = "CPU op threads (unchanged if absent)", presence = Presence.ABSENT) @Range(min = 1) Integer threads,
    @Key(doc = "Random seed for the initial weights from scratch and for shuffling rows", value = "1") long seed,
    @Key(doc = "State directory (if given, training runs in segments)", presence = Presence.ABSENT) Path state,
    @Key(doc = "Minutes per segment (needs state)", presence = Presence.ABSENT) @Range(min = 0, minInclusive = false)
        Double minutes,
    @Key(doc = "File whose appearance ends the segment (needs state)", presence = Presence.ABSENT) Path pauseFile)
    implements ClassicCommand {

  /** 学習の記録のファイル名の末尾。 */
  public static final String REPORT_SUFFIX = ".pretrain.json";

  public Pretrain {
    if ((from == null) == (architecture == null)) {
      throw new InvalidSetting("from", "give exactly one of from and architecture");
    }
    if (state == null && (minutes != null || pauseFile != null)) {
      throw new InvalidSetting(minutes != null ? "minutes" : "pauseFile", "minutes and pauseFile need state");
    }
    try {
      NetworkDevices.device(device);
    } catch (IllegalArgumentException invalid) {
      throw new InvalidSetting("device", "expected cpu or gpuN");
    }
  }

  @Override
  public Outcome run(Context<ClassicSettings> context) throws IOException {
    Signals signals = context.signals();
    Pretrainer.Result result = train(signals::requested);
    String summary = summary(result);
    return result.finished() ? Outcome.ok(summary) : new Outcome(Exit.STOPPED, summary);
  }

  /**
   * 事前学習する。全てのエポックを終えたら、checkpoint と学習の記録を書く。
   *
   * @param stop 停止のフラグ。分の上限と区切りのファイルのほかに、ステップごとに確かめる
   * @throws SettingsException 書く checkpoint が既にあるとき
   */
  public Pretrainer.Result train(BooleanSupplier stop) throws IOException {
    Inputs.requireNew(out, "output checkpoint");
    Inputs.requireDirectory(rows, "rows");
    if (from != null) {
      Inputs.requireDirectory(from, "from");
    }
    if (grp != null) {
      Inputs.requireDirectory(grp, "grp");
    }
    Pretrainer.Settings settings =
        new Pretrainer.Settings(epochs, batch, learningRate, weightDecay, valueWeight, emphasis, clipNorm, seed);
    long deadline = minutes != null ? System.nanoTime() + Math.round(minutes * 60e9) : Long.MAX_VALUE;
    BooleanSupplier pause =
        () ->
            stop.getAsBoolean()
                || System.nanoTime() - deadline > 0
                || (pauseFile != null && Files.exists(pauseFile));
    RowChunks loaded = RowChunks.read(rows);
    GrpNetwork prior = grp == null ? null : network(grp);
    Device on = Networks.device(device, threads == null ? 0 : threads);
    try (NDManager manager = NDManager.newBaseManager(on, PtEngine.ENGINE_NAME)) {
      long startStep;
      ClassicNetwork network;
      if (from != null) {
        network = ClassicCheckpoint.read(from, loaded.variant(), manager);
        startStep = ClassicCheckpoint.header(from).step();
      } else {
        network = Networks.create(architecture, loaded.variant(), manager, seed);
        startStep = 0;
      }
      Pretrainer.Result result =
          new Pretrainer(network, manager, precision, loaded, prior, settings).train(state, pause);
      if (result.finished()) {
        ClassicCheckpoint.write(out, network, startStep + result.steps());
        Files.writeString(
            out.resolveSibling(out.getFileName() + REPORT_SUFFIX),
            new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create().toJson(result),
            StandardCharsets.UTF_8);
      }
      return result;
    }
  }

  private static GrpNetwork network(Path directory) throws IOException {
    GrpModel model = GrpCheckpoint.read(directory);
    if (!(model instanceof GrpNetwork network)) {
      throw new IllegalArgumentException("The value prior needs a network GRP, not " + model.getClass().getSimpleName());
    }
    return network;
  }

  private static String summary(Pretrainer.Result result) {
    Pretrainer.Epoch best = result.bestEpoch();
    String head =
        String.format(
            Locale.ROOT,
            "finished=%s rows=%d validation=%d uniform=%.4f rulebot=%.4f steps=%d epochs=%d",
            result.finished(), result.trainRows(), result.validationRows(), result.baselines().uniformAgreement(),
            result.baselines().ruleBotAgreement(), result.steps(), result.epochs().size());
    if (best == null) {
      return head;
    }
    return head
        + String.format(
            Locale.ROOT,
            " best epoch=%d agreement=%.4f nll=%.4f value=%.4f",
            result.best(), best.agreement(), best.nll(), best.valueLoss());
  }
}
