package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.jni.JniUtils;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.Trainer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.ai.grp.EpsilonGrpSinkhorn;
import com.epsilon.ai.grp.EpsilonGrpTrainer;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.ai.network.NetworkFactory;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Comparator;
import java.util.List;
import java.util.Random;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * GRP を半荘の記録で CPU で学習する（trainGames）。損失は 1 半荘の全ての局末の接頭辞の最終順位の NLL をまとめたもの（{@link
 * GrpSequenceForward#gameLogits}）で、接頭辞ごとに足した損失と一致する。半荘の最後の局末（Φ が最終順位の効用そのもの）は学ばない。
 *
 * <p>損失は接頭辞あたり −Σ_席 log p(席, 最終順位)/4 の平均（元の GRP の事前学習と同じ式）。最適化は AdamW（減衰 1e-4、要素ごとの勾配の
 * 切り詰め 0.5。元の GRP と同じ）で、温めでも最適化器の状態は持ち越さない。検証の半荘があれば epoch ごとに NLL を測り、一番良い epoch の重みを
 * 保存し、{@value #PATIENCE} epoch 続けて良くならなければ止める。
 */
public final class GrpGameTrainer {
  private static final Logger log = LoggerFactory.getLogger(GrpGameTrainer.class);
  private static final float WEIGHT_DECAY = 1e-4f, GRAD_CLIP = 0.5f;

  /** 検証の NLL が良くならない epoch がこの数続いたら止める。 */
  static final int PATIENCE = 2;

  private GrpGameTrainer() {}

  /**
   * 学習の結果。
   *
   * @param directory 保存した GRP の正確なディレクトリ（grp-epsilon-v3 の checkpoint）
   * @param sha256 その sha
   * @param epochs 回した epoch の数
   * @param steps 歩数
   * @param trainLoss 最後の epoch の学習の NLL（局末あたり）
   * @param validationLoss 検証の半荘の NLL（局末あたり。検証がなければ NaN）
   * @param gamesPerSecond 学習の歩だけの速さ（半荘/秒。検証と保存の時間は入れない）
   * @param timedOut 締め切りで止めたなら true（directory は書かない）
   */
  public record Trained(
      Path directory,
      String sha256,
      int epochs,
      long steps,
      double trainLoss,
      double validationLoss,
      double gamesPerSecond,
      boolean timedOut) {}

  /** epoch の終わりの通知。 */
  @FunctionalInterface
  public interface EpochListener {
    /** 何もしない。 */
    EpochListener NONE = (epoch, trainLoss, validationLoss, model) -> {};

    /**
     * @param epoch 終わった epoch（1 から）
     * @param trainLoss その epoch の学習の NLL（局末あたり）
     * @param validationLoss 検証の NLL（検証がなければ NaN）
     * @param model 今の重みの GRP（呼び手は書き換えない）
     */
    void epochEnded(int epoch, double trainLoss, double validationLoss, Model model) throws IOException;
  }

  /**
   * 学習して out に保存する（out は無いこと。書きかけの扱いと rename は呼び手）。
   *
   * @param start 温めの始めの GRP の正確なディレクトリ（SCRATCH なら null）
   * @param training 学習の半荘
   * @param validation 早めの打ち切りに使う検証の半荘（空なら epoch の上限まで回す）
   * @throws IOException 始めの GRP が読めないか、保存できないとき
   */
  public static Trained train(
      Path start,
      List<RecordedGame> training,
      List<RecordedGame> validation,
      GrpTrainingPlan plan,
      Path out)
      throws IOException {
    return train(start, training, validation, plan, out, EpochListener.NONE);
  }

  /**
   * {@link #train(Path, List, List, GrpTrainingPlan, Path)} に epoch ごとの通知を足したもの。torch の演算スレッドは学習のあいだだけ
   * plan.threads() にし、終わったら元に戻す。
   */
  public static Trained train(
      Path start,
      List<RecordedGame> training,
      List<RecordedGame> validation,
      GrpTrainingPlan plan,
      Path out,
      EpochListener listener)
      throws IOException {
    if ((start == null) != (plan.init() == GrpInit.SCRATCH))
      throw new IllegalArgumentException("A start GRP is required exactly for WARM: " + plan.init());
    if (Files.exists(out)) throw new IOException("GRP training output already exists: " + out);
    List<RecordedGame> games = continuing(training);
    List<RecordedGame> held = continuing(validation);
    if (games.isEmpty()) throw new IllegalArgumentException("No GRP training game has a continuing hand end");
    Path staging = out.resolveSibling("." + out.getFileName() + ".training");
    deleteTree(staging);
    int threads = JniUtils.getNumThreads();
    JniUtils.setNumThreads(plan.threads());
    try {
      Engine.getEngine("PyTorch").setRandomSeed(Long.hashCode(plan.seed()));
      try (Model model =
              start == null
                  ? NetworkFactory.createGrpModel(
                      Device.cpu(), EpsilonGrpNetwork.DEFAULT_HIDDEN, EpsilonGrpNetwork.DEFAULT_LAYERS)
                  : EpsilonGrpCheckpointManager.load(start, Device.cpu());
          Trainer trainer = model.newTrainer(configuration(model, plan));
          EpsilonGrpTrainer saver =
              new EpsilonGrpTrainer(model, plan.batchGames(), (float) plan.learningRate(), WEIGHT_DECAY, GRAD_CLIP)) {
        return loop(model, trainer, saver, games, held, plan, staging, out, listener);
      }
    } finally {
      JniUtils.setNumThreads(threads);
      deleteTree(staging);
    }
  }

  private static Trained loop(
      Model model,
      Trainer trainer,
      EpsilonGrpTrainer saver,
      List<RecordedGame> games,
      List<RecordedGame> held,
      GrpTrainingPlan plan,
      Path staging,
      Path out,
      EpochListener listener)
      throws IOException {
    EpsilonGrpNetwork network = (EpsilonGrpNetwork) model.getBlock();
    long started = System.nanoTime();
    long deadline = started + plan.deadline().toNanos();
    Random shuffle = new Random(plan.seed());
    List<RecordedGame> order = new ArrayList<>(games);
    double best = Double.POSITIVE_INFINITY, trainLoss = Double.NaN;
    int epochs = 0, stale = 0;
    long steps = 0, seen = 0, stepNanos = 0;
    boolean timedOut = false;
    training:
    for (int epoch = 1; epoch <= plan.maxEpochs(); epoch++) {
      Collections.shuffle(order, shuffle);
      double sum = 0;
      long rows = 0;
      for (int from = 0; from < order.size(); from += plan.batchGames()) {
        if (System.nanoTime() > deadline) {
          timedOut = true;
          break training;
        }
        List<RecordedGame> batch = order.subList(from, Math.min(order.size(), from + plan.batchGames()));
        long stepStarted = System.nanoTime();
        try (NDManager sub = model.getNDManager().newSubManager()) {
          NDArray labels = labels(sub, batch);
          try (GradientCollector collector = trainer.newGradientCollector()) {
            NDArray loss =
                loss(GrpSequenceForward.gameLogits(network, new ParameterStore(sub, false), sub, batch, true), labels);
            collector.backward(loss);
            float value = loss.getFloat();
            if (!Float.isFinite(value)) throw new IllegalStateException("Non-finite GRP training loss " + value);
            sum += (double) value * labels.getShape().get(0);
            rows += labels.getShape().get(0);
          }
          trainer.step();
        }
        stepNanos += System.nanoTime() - stepStarted;
        steps++;
        seen += batch.size();
      }
      epochs = epoch;
      trainLoss = sum / rows;
      double validationLoss = held.isEmpty() ? Double.NaN : nll(model, held, plan.batchGames());
      log.info("GRP train epoch {}: train NLL {} validation NLL {}", epoch, trainLoss, validationLoss);
      listener.epochEnded(epoch, trainLoss, validationLoss, model);
      if (held.isEmpty()) continue;
      if (validationLoss < best) {
        best = validationLoss;
        stale = 0;
        EpsilonGrpCheckpointManager.save(model, saver, staging, (int) steps, 0);
      } else if (++stale >= PATIENCE) {
        break;
      }
    }
    double speed = seen / (stepNanos / 1e9);
    if (timedOut)
      return new Trained(null, "", epochs, steps, trainLoss, held.isEmpty() ? Double.NaN : best, speed, true);
    if (held.isEmpty()) EpsilonGrpCheckpointManager.save(model, saver, staging, (int) steps, 0);
    Files.move(staging, out, StandardCopyOption.ATOMIC_MOVE);
    return new Trained(
        out,
        EpsilonGrpCheckpointManager.checkpointSha256(out),
        epochs,
        steps,
        trainLoss,
        held.isEmpty() ? Double.NaN : best,
        speed,
        false);
  }

  /** 半荘の列の局末あたりの最終順位の NLL（学習の損失と同じ式、勾配なし）。 */
  static double nll(Model model, List<RecordedGame> games, int batchGames) {
    EpsilonGrpNetwork network = (EpsilonGrpNetwork) model.getBlock();
    List<RecordedGame> held = continuing(games);
    double sum = 0;
    long rows = 0;
    for (int from = 0; from < held.size(); from += batchGames) {
      List<RecordedGame> batch = held.subList(from, Math.min(held.size(), from + batchGames));
      try (NDManager sub = model.getNDManager().newSubManager()) {
        NDArray labels = labels(sub, batch);
        NDArray loss =
            loss(GrpSequenceForward.gameLogits(network, new ParameterStore(sub, false), sub, batch, false), labels);
        sum += (double) loss.getFloat() * labels.getShape().get(0);
        rows += labels.getShape().get(0);
      }
    }
    return sum / rows;
  }

  /** 接頭辞あたりの −Σ_席 log p(席, 最終順位)/4 の平均。 */
  static NDArray loss(NDArray logits, NDArray labels) {
    long rows = logits.getShape().get(0);
    return labels
        .mul(EpsilonGrpSinkhorn.logMarginals(logits))
        .reshape(rows, EpsilonGrpRanks.MATRIX_SIZE)
        .sum(new int[] {1})
        .neg()
        .div(EpsilonGrpRanks.SEAT_COUNT)
        .mean();
  }

  /** 半荘が続く全ての局末の最終順位の one-hot（[Σ 局末, 4, 4]、{@link GrpSequenceForward#gameLogits} と同じ順）。 */
  static NDArray labels(NDManager manager, List<RecordedGame> games) {
    int rows = 0;
    for (RecordedGame game : games) rows += game.continuingHandEnds();
    float[] labels = new float[rows * EpsilonGrpRanks.MATRIX_SIZE];
    int row = 0;
    for (RecordedGame game : games) {
      int[] ranks = game.finalRanks();
      for (int k = 0; k < game.continuingHandEnds(); k++, row++)
        for (int seat = 0; seat < 4; seat++)
          labels[row * EpsilonGrpRanks.MATRIX_SIZE + EpsilonGrpRanks.index(seat, ranks[seat])] = 1;
    }
    return manager.create(labels, new Shape(rows, 4, 4));
  }

  private static DefaultTrainingConfig configuration(Model model, GrpTrainingPlan plan) {
    // 損失は手で計算するので、Loss は Trainer を作るためだけの置き物
    return new DefaultTrainingConfig(Loss.l2Loss())
        .optDevices(new Device[] {model.getNDManager().getDevice()})
        .optOptimizer(
            Optimizer.adamW()
                .optLearningRateTracker(Tracker.fixed((float) plan.learningRate()))
                .optWeightDecays(WEIGHT_DECAY)
                .optClipGrad(GRAD_CLIP)
                .build());
  }

  private static List<RecordedGame> continuing(List<RecordedGame> games) {
    return games.stream().filter(game -> game.continuingHandEnds() > 0).toList();
  }

  private static void deleteTree(Path root) throws IOException {
    if (!Files.exists(root)) return;
    try (Stream<Path> paths = Files.walk(root)) {
      for (Path path : paths.sorted(Comparator.reverseOrder()).toList()) Files.delete(path);
    }
  }
}
