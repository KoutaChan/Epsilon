package com.epsilon.prism.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.nn.Parameter;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.input.InputFormat;
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
 * ネットワークを学習する（refit の single・replay と、系譜の single・filter の生徒）。行を {@link #shuffle} で並べ替え（乱数の種は
 * 計画の種）、先頭から 1 バッチ分ずつ取って、計画のステップ数だけ学習する。最後まで取ったら先頭に戻る。行の読み出し、マイクロバッチへの
 * 詰め込み、エンコード、教師（{@link Teacher}）は、先読みのスレッドが {@link #PREFETCH} ステップ先まで作る（{@link StepFeed}）。
 * 学習のスレッドは、デバイスの計算だけを回す（{@link Learner}）。デバイスが 2 つ以上なら、詰めたマイクロバッチの数がデバイスの数の
 * 倍数にならなかったステップの数をログに出す（そのステップの最後のマイクロバッチは、全てのレーンに分けて載せる）。
 * システムプロパティ {@value FitProfile#PROPERTY} があれば、段ごとの時間の表を学習の終わりにログに出す（{@link FitProfile}）。
 *
 * <p>系譜では、教師が決まる前（運の精算の当てはめの間）に {@link #prepare} で生徒を読み、学習器をウォームアップしておく
 * （{@link Learner#rehearse}）。ウォームアップは重みにも最適化の状態にもステップ数にも触れないので、ウォームアップを挟んで学習した重みは、
 * {@link #run} だけで学習した重みとビット単位で一致する。学習器のストリームは、{@link #prepare} では呼び出し元の実行のリソースから借りる
 * （系譜は JVM が動いている間、同じリソースを渡す）。{@link #run} では、その場で作って閉じる。
 */
public final class Fit {

  /** 先に作っておくステップの数。 */
  static final int PREFETCH = 2;

  /** 進み具合をログに出すステップの間隔（GPU で損失を回収するステップと重なる）。 */
  private static final int PROGRESS_EVERY = 1000;

  /** 並べ替えの単位の行の数。単位の中は元の順（行のストアの番号の順 = シャードの中の位置の順）のまま。 */
  static final int BLOCK = 64;

  private static final Logger LOG = LoggerFactory.getLogger(Fit.class);

  private Fit() {}

  /**
   * 初期化したままのネットワークのモデルを作る。初期値は乱数の種で決まる（エンジンの乱数の種を決めてから作る）。
   *
   * @param format 読む入力の形式
   * @param shape ネットワークの形
   * @param device デバイス（学習のデバイスの先頭）
   * @param seed 初期値の乱数の種
   * @return モデル（ブロックは {@link PrismNetwork}。閉じるのは呼び出し元）
   */
  public static Model blank(InputFormat format, PrismShape shape, Device device, long seed) {
    // エンジンを初めて取る前に、全てのモデルに共通の実行時の既定値（PyTorch の CPU スレッド数）を決める
    NetworkRuntime.initialize();
    Model model = Model.newInstance("epsilon-prism-v2", device, "PyTorch");
    try {
      PrismNetwork network = new PrismNetwork(format, shape);
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
   * 保存したネットワークを学習用のモデルに読む（系譜の生徒のウォームスタート）。読み込んだ重みは勾配を要求しないので、要求するように
   * 変える。
   *
   * @param directory ネットワークのディレクトリ（{@link PrismNetwork#save} の形）
   * @param device デバイス（学習のデバイスの先頭）
   * @return モデル（ブロックは {@link PrismNetwork}。閉じるのは呼び出し元）
   * @throws IOException 読めないか、形式・フィンガープリント・重みが合わないとき
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
   * ネットワークを学習する。重みはモデルのネットワークに残る（保存するのは呼び出し元）。
   *
   * @param store 行のストア
   * @param rows 学習する行（{@link CohortStore#select} で選んだもの）
   * @param model ネットワークのモデル（ブロックは {@link PrismNetwork}、デバイスは {@code devices} の先頭）
   * @param plan 学習の計画
   * @param teacher 教師の作り方（replay は {@link Teacher#RAW}）
   * @param devices 学習のデバイス（先頭がマスターで、残りはデータ並列のレプリカ）
   * @param threads 行の読み出しとエンコードのスレッドの数
   * @return 記録（{@code stageSeconds} は {ステップを作る, 入力と損失の教師の転送, 順伝播と逆伝播, 最適化}）
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
   * 計測を渡してネットワークを学習する。{@link #run(CohortStore, int[], Model, FitPlan, Teacher, NetworkDevices, int)} の本体で、
   * 計測するステップがあれば、学習の終わりに表をログに出す。
   *
   * @param store 行のストア
   * @param rows 学習する行
   * @param model ネットワークのモデル
   * @param plan 学習の計画
   * @param teacher 教師の作り方
   * @param devices 学習のデバイス
   * @param threads 行の読み出しとエンコードのスレッドの数
   * @param profile 学習の計測
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
    if (rows.length == 0 || plan.learner().steps() <= 0) {
      return FitResult.empty(rows.length);
    }
    try (NetworkRuntime runtime = NetworkRuntime.forLearning()) {
      return new Session(store, rows, model, plan, teacher.gain(), devices, threads, profile, runtime)
          .run(teacher, started);
    }
  }

  /**
   * 生徒を読み、学習器をウォームアップする（{@link Learner#rehearse}）。レーンとレプリカを作ってカーネルを読み込み、最初のステップの行の
   * マイクロバッチを 1 つずつ通して、勾配を捨てる。ウォームアップのマイクロバッチの教師は生の Y にする。補正とアンカーは学習の教師で
   * 決まるが、捨てる勾配の値は何でもよいからである。
   *
   * @param store 行のストア
   * @param rows 学習する行
   * @param start 生徒の始めのネットワークのディレクトリ
   * @param plan 学習の計画（ステップ数は 1 以上）
   * @param gain 補助のヘッドの項に掛けるゲイン K′（学習の教師と同じ値）
   * @param devices 学習のデバイス
   * @param threads 行の読み出しとエンコードのスレッドの数
   * @param runtime 学習器のストリームを借りる実行のリソース（閉じるのは呼び出し元。返した準備を閉じるまで開いておく）
   * @return ウォームアップを済ませた学習の準備（閉じるのは呼び出し元）
   * @throws IOException 生徒のネットワークを読めないとき
   * @throws IllegalArgumentException 行がないか、ステップ数が 0 以下のとき
   */
  public static Prepared prepare(
      CohortStore store,
      int[] rows,
      Path start,
      FitPlan plan,
      float gain,
      NetworkDevices devices,
      int threads,
      NetworkRuntime runtime)
      throws IOException {
    if (rows.length == 0 || plan.learner().steps() <= 0) {
      throw new IllegalArgumentException(
          "a prepared fit needs rows and steps: " + rows.length + ", " + plan.learner().steps());
    }
    long started = System.nanoTime();
    Model model = load(start, devices.primary());
    try {
      Session session =
          new Session(store, rows, model, plan, gain, devices, threads, FitProfile.configured(), runtime);
      try {
        session.rehearse();
      } catch (RuntimeException | Error failure) {
        session.closeQuietly(failure);
        throw failure;
      }
      LOG.info(
          "fit prepared the learner {}s", String.format("%.1f", (System.nanoTime() - started) / 1e9));
      return new Prepared(model, session);
    } catch (RuntimeException | Error failure) {
      model.close();
      throw failure;
    }
  }

  /** ウォームアップを済ませた学習の準備（{@link #prepare} が返す）。生徒のモデルを持ち、閉じるときに一緒に閉じる。 */
  public static final class Prepared implements AutoCloseable {

    private final Model model;
    private final Session session;

    private Prepared(Model model, Session session) {
      this.model = model;
      this.session = session;
    }

    /**
     * 生徒のネットワークを返す（学習した後の重みを保存するのは呼び出し元）。
     *
     * @return ネットワーク
     */
    public PrismNetwork network() {
      return (PrismNetwork) model.getBlock();
    }

    /**
     * ネットワークを学習する（呼べるのは 1 度だけ）。秒数は、この呼び出しから学習器を閉じるまで（読み込みとウォームアップは含めない）。
     *
     * @param teacher 教師の作り方（ゲインはウォームアップしたときと同じ）
     * @return 記録
     * @throws IllegalArgumentException 教師のゲインがウォームアップしたときと違うとき
     */
    public FitResult fit(Teacher teacher) {
      return session.run(teacher, System.nanoTime());
    }

    /** まだ学習していなければ学習器と先読みを閉じ、生徒のモデルを閉じる。 */
    @Override
    public void close() {
      try {
        session.close();
      } finally {
        model.close();
      }
    }
  }

  /** 1 回の学習の先読みと学習器（並べ替えた行の順を持つ）。{@link #run} の終わりか {@link #close} で閉じる。 */
  private static final class Session implements AutoCloseable {

    private final int[] order;
    private final FitPlan plan;
    private final float gain;
    private final int lanes;
    private final FitProfile profile;
    private final StepFeed feed;
    private final Learner learner;
    private boolean closed;

    Session(
        CohortStore store,
        int[] rows,
        Model model,
        FitPlan plan,
        float gain,
        NetworkDevices devices,
        int threads,
        FitProfile profile,
        NetworkRuntime runtime) {
      order = shuffle(rows, plan.seed());
      this.plan = plan;
      this.gain = gain;
      lanes = devices.size();
      this.profile = profile;
      feed = new StepFeed(store, plan, devices.size(), threads);
      try {
        learner = new Learner(model, devices, plan, gain, profile, runtime);
      } catch (RuntimeException | Error failure) {
        feed.close();
        throw failure;
      }
    }

    /** 最初のステップの行で学習器をウォームアップする（教師は生の Y）。 */
    void rehearse() {
      int[] rows = batch(order, 0, plan.learner().batchSize());
      StepFeed.Step step = await(feed.prepare(rows, new Teacher(null, null, 0, gain)));
      try {
        learner.rehearse(step);
      } finally {
        feed.release(step);
      }
    }

    /** 計画のステップ数だけ学習して閉じる。秒数は {@code started} から、学習器を閉じる（更新の完了を待つ）まで。 */
    FitResult run(Teacher teacher, long started) {
      if (closed) {
        throw new IllegalStateException("the fit already ran");
      }
      if (teacher.gain() != gain) {
        throw new IllegalArgumentException(
            "the teacher gain " + teacher.gain() + " differs from the learner gain " + gain);
      }
      int steps = plan.learner().steps();
      int batchSize = plan.learner().batchSize();
      double sum = 0;
      double first = Double.NaN;
      double last = Double.NaN;
      int measured = 0;
      double[] stages;
      int clipped;
      try {
        ArrayDeque<CompletableFuture<StepFeed.Step>> pending = new ArrayDeque<>(PREFETCH);
        int from = 0;
        int queued = 0;
        for (int step = 0; step < steps; step++) {
          while (pending.size() < PREFETCH && queued < steps) {
            pending.add(feed.prepare(batch(order, from, batchSize), teacher));
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
        if (lanes > 1) {
          LOG.info(
              "fit {} of {} steps packed into a piece count that is not a multiple of {} lanes;"
                  + " their last piece was split across the lanes",
              feed.unevenSteps(),
              steps,
              lanes);
        }
        profile.log();
      } catch (RuntimeException | Error failure) {
        closeQuietly(failure);
        throw failure;
      }
      close();
      // 秒数は学習器を閉じる（更新の完了を待つ）までを含める
      return new FitResult(
          order.length,
          steps,
          clipped,
          measured > 0 ? sum / measured : Double.NaN,
          first,
          last,
          (System.nanoTime() - started) / 1e9,
          stages);
    }

    /** 学習器を閉じ（更新の完了を待つ）、先読みを止める。2 度目からは何もしない。 */
    @Override
    public void close() {
      if (closed) {
        return;
      }
      closed = true;
      try {
        learner.close();
      } finally {
        feed.close();
      }
    }

    /** 失敗の後に閉じる。閉じるときに起きた例外は、元の失敗に添える。 */
    void closeQuietly(Throwable failure) {
      try {
        close();
      } catch (RuntimeException | Error problem) {
        failure.addSuppressed(problem);
      }
    }
  }

  /**
   * 行を {@link #BLOCK} 行ずつのまとまりに分け、まとまりの順を乱数で並べ替えたコピーを返す。まとまりの中は元の順なので、1 バッチの
   * 読み出しはシャードの中の続いた位置を読むことになり、1 行ずつのランダムな読みが、先読みの効く順次の読みになる。まとまりは同じ半荘の
   * 判断を含むので、バッチの中の独立な半荘の数は、バッチの行数 / BLOCK 程度に減る（ステップ数が多いので、平均すれば薄まる）。
   *
   * @param rows 行（行のストアの番号の昇順）
   * @param seed 並べ替えの乱数の種
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

  /** {@code order} の [from, from + size) の部分。末尾を越える分は切り捨てる。 */
  private static int[] batch(int[] order, int from, int size) {
    return Arrays.copyOfRange(order, from, Math.min(order.length, from + size));
  }

  /** 先読みの完了を待つ。先読みのスレッドで起きた例外は、包んだ例外から取り出して投げる。 */
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
