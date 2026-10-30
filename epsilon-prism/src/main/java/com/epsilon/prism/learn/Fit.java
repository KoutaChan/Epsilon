package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.nn.Parameter;
import com.epsilon.network.NetworkDevices;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayDeque;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.CompletionException;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 網の当てはめ（refit の single・replay と、系譜の single・filter の生徒）。行を {@link #shuffle}（種は計画の種）で
 * 並べ替え、バッチの行ずつ順に取って（足りなければ巡回）計画の歩数だけ当てる。行の読み出し・微小バッチへの詰め・符号化・教師（{@link Teacher}）は
 * 先読みのスレッドで {@link #PREFETCH} 歩先まで作り（{@link StepFeed}）、学習のスレッドはデバイスの計算だけを回す（{@link Learner}）。
 * デバイスが 2 つ以上なら、詰めた塊の数がデバイスの数の倍数でなかった歩の数を記録する（その歩の最後の塊はデバイスの数の部分に分ける）。
 * 系の性質 {@value FitProfile#PROPERTY} があれば、段の時計の計測の表を当てはめの終わりに記録に出す（{@link FitProfile}）。
 */
public final class Fit {

  /** 記録の名前。 */
  public static final String NAME = "v2";

  /** 先に作っておく歩の数。 */
  static final int PREFETCH = 2;

  /** 進みを記録する歩の間隔（GPU で損失を回収する歩に重なる）。 */
  private static final int PROGRESS_EVERY = 1000;

  /** 並べ替えの塊の長さ。塊の中は元の順（貯蔵の索引の順 = shard の位置の順）のまま。 */
  static final int BLOCK = 64;

  private static final Logger LOG = LoggerFactory.getLogger(Fit.class);

  private Fit() {}

  /**
   * 白紙の網のモデルを作る。初期値は種で決まる（エンジンの乱数の種を決めてから作る）。
   *
   * @param shape 網の形
   * @param device デバイス（学習のデバイスの先頭）
   * @param seed 初期値の種
   * @return モデル（ブロックは {@link PrismNetwork}。閉じるのは呼び手）
   */
  public static Model blank(PrismShape shape, Device device, long seed) {
    // エンジンを初めて取る前に、全モデル共通の実行既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    Model model = Model.newInstance("epsilon-prism-v2", device, "PyTorch");
    try {
      PrismNetwork network = new PrismNetwork(shape);
      model.setBlock(network);
      synchronized (Fit.class) {
        model.getNDManager().getEngine().setRandomSeed(Long.hashCode(seed));
        network.initialize(model.getNDManager());
      }
      return model;
    } catch (RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  /**
   * 保存した網を学習用のモデルに読む（系譜の生徒の温め込み）。読み込んだ重みは勾配を要求しないので、要求させる。
   *
   * @param directory 網のディレクトリ（{@link PrismNetwork#save} の形）
   * @param device デバイス（学習のデバイスの先頭）
   * @return モデル（ブロックは {@link PrismNetwork}。閉じるのは呼び手）
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  public static Model load(Path directory, Device device) throws IOException {
    NetworkRuntime.initialize();
    Model model = Model.newInstance("epsilon-prism-v2", device, "PyTorch");
    try {
      PrismNetwork network = PrismNetwork.load(directory, model.getNDManager());
      for (Parameter parameter : network.getParameters().values()) {
        parameter.freeze(false);
        parameter.getArray().setRequiresGradient(true);
      }
      model.setBlock(network);
      return model;
    } catch (IOException | RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  /**
   * 網を当てる。重みはモデルの網に残る（保存は呼び手）。
   *
   * @param store 行の貯蔵
   * @param rows 学ぶ行（{@link CohortStore#select} で選んだもの）
   * @param model 網のモデル（ブロックは {@link PrismNetwork}、デバイスは devices の先頭）
   * @param plan 当てはめの計画
   * @param teacher 教師の作り方（replay は {@link Teacher#RAW}）
   * @param devices 学習のデバイス（先頭が正本で、残りはデータ並列の複製）
   * @param threads 行の読み出しと符号化のスレッドの数
   * @return 記録（stageSeconds は {歩を作る, 入力と損失の教師の転送, 順伝播と逆伝播, 最適化}）
   */
  public static FitResult run(
      CohortStore store,
      int[] rows,
      Model model,
      FitPlan plan,
      Teacher teacher,
      NetworkDevices devices,
      int threads) {
    return run(store, rows, model, plan, teacher, devices, threads, FitProfile.configured());
  }

  /**
   * 計測を渡して網を当てる（{@link #run(CohortStore, int[], Model, FitPlan, Teacher, NetworkDevices, int)} の本体。計る歩があれば表を
   * 終わりに記録に出す）。
   *
   * @param store 行の貯蔵
   * @param rows 学ぶ行
   * @param model 網のモデル
   * @param plan 当てはめの計画
   * @param teacher 教師の作り方
   * @param devices 学習のデバイス
   * @param threads 行の読み出しと符号化のスレッドの数
   * @param profile 当てはめの計測
   * @return 記録
   */
  static FitResult run(
      CohortStore store,
      int[] rows,
      Model model,
      FitPlan plan,
      Teacher teacher,
      NetworkDevices devices,
      int threads,
      FitProfile profile) {
    long started = System.nanoTime();
    int steps = plan.learner().steps();
    int batchSize = plan.learner().batchSize();
    if (rows.length == 0 || steps <= 0) {
      return FitResult.empty(NAME, rows.length);
    }
    int[] order = shuffle(rows, plan.seed());
    double sum = 0;
    double first = Double.NaN;
    double last = Double.NaN;
    int measured = 0;
    double[] stages;
    int clipped;
    try (StepFeed feed = new StepFeed(store, plan, teacher, devices.size(), threads);
        Learner learner = new Learner(model, devices, plan, teacher, profile)) {
      ArrayDeque<CompletableFuture<StepFeed.Step>> pending = new ArrayDeque<>(PREFETCH);
      int from = 0;
      int queued = 0;
      for (int step = 0; step < steps; step++) {
        while (pending.size() < PREFETCH && queued < steps) {
          pending.add(feed.prepare(batch(order, from, batchSize)));
          from = from + batchSize < order.length ? from + batchSize : 0;
          queued++;
        }
        StepFeed.Step prepared = await(pending.remove());
        double loss;
        try {
          loss = learner.train(prepared);
        } finally {
          feed.release(prepared);
        }
        if (!Double.isNaN(loss)) {
          sum += loss;
          measured++;
          if (Double.isNaN(first)) {
            first = loss;
          }
          last = loss;
        }
        if (step % PROGRESS_EVERY == 0) {
          LOG.info(
              "fit step {}/{} loss {} {}s",
              step + 1,
              steps,
              loss,
              String.format("%.1f", (System.nanoTime() - started) / 1e9));
        }
      }
      double[] learned = learner.stageSeconds();
      stages = new double[] {feed.seconds(), learned[0], learned[1], learned[2]};
      clipped = learner.clippedSteps();
      if (teacher.anchored()) {
        LOG.info("fit TEMPO anchor {}s over all lanes", String.format("%.1f", learner.anchorSeconds()));
      }
      if (devices.size() > 1) {
        LOG.info(
            "fit {} of {} steps packed into a piece count that is not a multiple of {} lanes;"
                + " their last piece was split across the lanes",
            feed.unevenSteps(),
            steps,
            devices.size());
      }
      profile.log();
    }
    // 秒数は学習器を閉じる（更新の完了を待つ）までを含める
    return new FitResult(
        NAME,
        rows.length,
        steps,
        clipped,
        measured > 0 ? sum / measured : Double.NaN,
        first,
        last,
        (System.nanoTime() - started) / 1e9,
        stages);
  }

  /**
   * rows を {@link #BLOCK} 行ずつの塊で seed で並べ替えた写し。塊の中は元の順なので、1 バッチの読み出しが shard の連続した位置の読みになり、
   * 1 行ずつのランダム読みが先読みの効く順次読みになる。塊は同じ半荘の判断を含むので、バッチの独立な半荘の数は
   * 行数 / BLOCK 程度に減る（歩数が多いので平均で薄まる）。
   *
   * @param rows 行（貯蔵の索引の昇順）
   * @param seed 並べ替えの種
   * @return 並べ替えた行
   */
  public static int[] shuffle(int[] rows, long seed) {
    int blocks = Math.ceilDiv(rows.length, BLOCK);
    int[] starts = new int[blocks];
    for (int b = 0; b < blocks; b++) {
      starts[b] = b * BLOCK;
    }
    SplittableRandom random = new SplittableRandom(seed);
    for (int i = blocks - 1; i > 0; i--) {
      int j = random.nextInt(i + 1);
      int swap = starts[i];
      starts[i] = starts[j];
      starts[j] = swap;
    }
    int[] order = new int[rows.length];
    int at = 0;
    for (int start : starts) {
      int end = Math.min(rows.length, start + BLOCK);
      System.arraycopy(rows, start, order, at, end - start);
      at += end - start;
    }
    return order;
  }

  /** order の [from, from + size) の部分（末尾で切る）。 */
  private static int[] batch(int[] order, int from, int size) {
    return Arrays.copyOfRange(order, from, Math.min(order.length, from + size));
  }

  /** 先読みの完了を待つ。先読みのスレッドの例外は包みを外して投げる。 */
  private static <T> T await(CompletableFuture<T> future) {
    try {
      return future.join();
    } catch (CompletionException failure) {
      if (failure.getCause() instanceof RuntimeException cause) {
        throw cause;
      }
      if (failure.getCause() instanceof Error cause) {
        throw cause;
      }
      throw failure;
    }
  }
}
