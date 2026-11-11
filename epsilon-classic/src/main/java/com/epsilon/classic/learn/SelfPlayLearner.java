package com.epsilon.classic.learn;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.util.Pair;
import com.epsilon.agent.learn.DataParallelGroup;
import com.epsilon.agent.learn.DataParallelGroup.LaneWork;
import com.epsilon.agent.learn.DataParallelGroup.Update;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Key;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicExtras;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicTokenizer;
import com.epsilon.classic.model.ClassicBatch;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.classic.model.HlGauss;
import com.epsilon.classic.model.PolicyGraph;
import com.epsilon.mahjong.util.SeedMixer;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.charset.StandardCharsets;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.stream.IntStream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Nano・Major の自己対局の学習器。1 回の macro（収集と学習を 1 回ずつ行う反復）で集めた軌跡（{@link TrajectoryBatch}）で学習する。方策は
 * 選んだ手だけで取る PPO（{@link PpoLoss}）で、価値は局の中の V-trace の目標（{@link ValueTargets}）で学ぶ。
 *
 * <ol>
 *   <li>学習の前のネットワークで全ての判断を推論し、合法手の対数確率 log π_前 と価値 V を取る。
 *   <li>選んだ手の補正 c、価値の目標、アドバンテージを求める。方策の損失に入れるのは、合法手が 2 つ以上あり、選び替えで局の進みが
 *       変わりえた判断だけ。
 *   <li>判断を乱数の種で並べ替え、{@code steps} 個（判断がそれより少なければ判断の数）のステップに等分する。ステップごとに全ての行の
 *       勾配を溜め、方策と価値を別々の AdamW で 1 回ずつ更新する。
 *       更新の前に測った KL(π_前 ‖ π) の平均が上限を超えたステップでは、方策の更新を捨てる（価値は更新する）。
 *   <li>ステップごとの KL の平均から、次の macro の方策の学習率を決める（{@link KlLearningRate}）。価値の学習率は変えない。
 * </ol>
 *
 * <p>データ並列で学ぶ。勾配はレーン（デバイスごとのネットワーク。0 番は渡されたネットワーク）ごとに溜める。{@link DataParallelGroup} が
 * 方策と価値のパラメーターの組をそれぞれの AdamW で更新し、全てのレーンへ配る。勾配のノルムは組ごとに上限で切る。GPU のレーンは BF16、
 * CPU のレーンは FP32 で計算する。学習器を開いている間は、学習器だけがネットワークの重みを書き換える。
 *
 * <p>入力は判断ごとに O から作り（{@link ClassicTokenizer}）、価値の入力には局の始まりの GRP の周辺確率を書く（周辺確率を持たない半荘は
 * 一様）。バッチは推論の区分（{@link Buckets#inference}）ごとにまとめる。バッチの行の数は、設定の行の上限と、合法手 × 遷移 × 行がセルの
 * 上限に収まる数の小さい方。
 *
 * <p>途中の状態（{@link #save}）として書くのは、方策と価値の AdamW の状態、学習率の制御の状態、macro とステップの数。重みは checkpoint
 * に書く。
 */
public final class SelfPlayLearner implements AutoCloseable {

  private static final Logger LOG = LoggerFactory.getLogger(SelfPlayLearner.class);

  private static final ClassicExtras EXTRAS = ClassicExtras.empty();

  private static final long SHUFFLE_SALT = 0x6A09_E667_F3BC_C908L;

  private static final String STATE = "learner.json";
  private static final String ACTOR_STATE = "actor.optimizer";
  private static final String VALUE_STATE = "value.optimizer";
  private static final String TEMPORARY = ".tmp";

  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();

  private static final PpoLoss.Sum[] SUMS = PpoLoss.Sum.values();

  /**
   * 学習の設定。classic の設定（{@code ClassicSettings}）のセクションから作る。
   *
   * @param ppo 方策の学び方
   * @param actorRate 方策の学習率の制御
   * @param valueRate 価値の学び方
   * @param vtrace 価値の目標
   * @param batches バッチの大きさ
   */
  public record Settings(
      Ppo ppo, KlLearningRate.Settings actorRate, ValueRate valueRate, VTrace vtrace, Batches batches) {

    public Settings {
      Objects.requireNonNull(ppo, "ppo");
      Objects.requireNonNull(actorRate, "actorRate");
      Objects.requireNonNull(valueRate, "valueRate");
      Objects.requireNonNull(vtrace, "vtrace");
      Objects.requireNonNull(batches, "batches");
    }
  }

  /**
   * 方策の学び方。既定は Nano の自己対局の値。
   *
   * @param steps 1 回の macro のステップ数（macro の判断を等分する）
   * @param clip PPO のクリップの幅（0 より大きく 1 未満）
   * @param credit 選んだ手の補正の下限（0 より大きく 1 以下）。価値の目標の補正にも使う
   * @param entropy エントロピーの係数（0 以上）
   * @param klLimit 更新の前に測る KL の平均の上限（超えたステップでは方策の更新を捨てる）
   * @param weightDecay AdamW の重み減衰
   * @param gradientClip 方策の勾配のノルムの上限
   */
  public record Ppo(
      @Key(doc = "1 回の macro のステップ数（macro の判断を等分する）", value = "32") int steps,
      @Key(doc = "PPO のクリップの幅（0 より大きく 1 未満）", value = "0.2") double clip,
      @Key(doc = "選んだ手の補正の下限（0 より大きく 1 以下）。価値の目標の補正にも使う", value = "0.1") double credit,
      @Key(doc = "エントロピーの係数（0 以上）", value = "0.001") double entropy,
      @Key(doc = "更新の前に測る KL の平均の上限（超えたステップでは方策の更新を捨てる）", value = "0.01") double klLimit,
      @Key(doc = "AdamW の重み減衰", value = "1e-4") double weightDecay,
      @Key(doc = "方策の勾配のノルムの上限", value = "0.5") double gradientClip) {

    public Ppo {
      if (steps < 1 || !(clip > 0 && clip < 1) || !(credit > 0 && credit <= 1) || !(entropy >= 0) || !(klLimit > 0)
          || !(weightDecay >= 0) || !(gradientClip > 0)) {
        throw new IllegalArgumentException("Bad PPO settings: " + this);
      }
    }
  }

  /**
   * 価値の学び方。既定は Nano の自己対局の値。
   *
   * @param rate 学習率（学習の間は変えない）
   * @param weightDecay AdamW の重み減衰
   * @param gradientClip 価値の勾配のノルムの上限
   */
  public record ValueRate(
      @Key(doc = "学習率（学習の間は変えない）", value = "1.25e-6") double rate,
      @Key(doc = "AdamW の重み減衰", value = "1e-4") double weightDecay,
      @Key(doc = "価値の勾配のノルムの上限", value = "0.5") double gradientClip) {

    public ValueRate {
      if (!(rate > 0) || !(weightDecay >= 0) || !(gradientClip > 0)) {
        throw new IllegalArgumentException("Bad value settings: " + this);
      }
    }
  }

  /**
   * 価値の目標。
   *
   * @param lambda 局の中の V-trace で、次の判断の価値の代わりにその判断の価値の目標を使う割合（0 以上 1 以下）
   */
  public record VTrace(@Key(doc = "局の中の V-trace で、次の判断の価値の代わりにその判断の価値の目標を使う割合（0 以上 1 以下）", value = "0.95") double lambda) {

    public VTrace {
      if (!(lambda >= 0 && lambda <= 1)) {
        throw new IllegalArgumentException("Lambda must be in [0, 1]: " + lambda);
      }
    }
  }

  /**
   * バッチの大きさ。
   *
   * @param rows 1 つのバッチの行の上限
   * @param cells 1 つのバッチの合法手 × 遷移 × 行の上限
   */
  public record Batches(
      @Key(doc = "1 つのバッチの行の上限", value = "1024") int rows,
      @Key(doc = "1 つのバッチの合法手 × 遷移 × 行の上限", value = "16384") int cells) {

    public Batches {
      if (rows < 1 || cells < 1) {
        throw new IllegalArgumentException("Batch rows and cells must be positive: " + rows + ", " + cells);
      }
    }
  }

  /**
   * 1 ステップの記録。
   *
   * @param index macro の中でのステップの番号（0 始まり）
   * @param rows ステップの行の数
   * @param actorRows ステップの中で、方策の損失に入れた行の数
   * @param kl 更新の前に測った KL(π_前 ‖ π) の、方策の行での平均（方策の行が無ければ 0）
   * @param actorUpdated 方策を更新したなら true（KL が上限を超えたステップと、方策の行が無いステップは false）
   * @param surrogate 方策の行での、補正を掛けたサロゲート目的関数の平均
   * @param entropy 方策の行でのエントロピーの平均
   * @param valueLoss 価値の交差エントロピーの平均
   * @param clipFraction 比がクリップの幅の外だった方策の行の割合
   * @param actorNorm 切る前の方策の勾配のノルム（更新しなければ NaN）
   * @param valueNorm 切る前の価値の勾配のノルム
   */
  public record Step(
      int index,
      int rows,
      int actorRows,
      double kl,
      boolean actorUpdated,
      double surrogate,
      double entropy,
      double valueLoss,
      double clipFraction,
      double actorNorm,
      double valueNorm) {}

  /**
   * 1 回の macro の学習の記録。
   *
   * @param games 半荘の数
   * @param decisions 判断の数
   * @param actorRows 方策の損失に入れた判断の数
   * @param steps ステップごとの記録
   * @param kl 方策の行があるステップでの KL の平均（学習率の制御に渡した値）
   * @param credit 方策の行での補正の平均
   * @param advantage 方策の行でのアドバンテージの平均
   * @param rate 方策の学習率の調整
   * @param evaluateSeconds 学習の前の推論と、目標を作るのにかかった秒数
   * @param stepSeconds 全てのステップにかかった秒数
   */
  public record Macro(
      int games,
      int decisions,
      int actorRows,
      List<Step> steps,
      double kl,
      double credit,
      double advantage,
      KlLearningRate.Adjustment rate,
      double evaluateSeconds,
      double stepSeconds) {

    public Macro {
      steps = List.copyOf(steps);
    }

    /** 方策を更新したステップの数。 */
    public int actorSteps() {
      return (int) steps.stream().filter(Step::actorUpdated).count();
    }
  }

  /** 書き出した状態（learner.json）。要素の数は、組ごとのパラメーターの要素数の和。読み戻すときにネットワークと照合する。 */
  private record Saved(
      String architecture,
      long actorElements,
      long valueElements,
      long macros,
      long actorSteps,
      long valueSteps,
      KlLearningRate.State actorRate) {}

  private final ClassicNetwork network;
  private final NDManager manager;
  private final Settings settings;
  private final Optimizer actorOptimizer;
  private final Optimizer valueOptimizer;
  private final ParallelOptimizer group;
  private KlLearningRate actorRate;

  /** 今の macro の方策の学習率（AdamW の更新の中で読む）。 */
  private volatile float actorLearningRate;

  private long macros;
  private long actorSteps;
  private long valueSteps;

  /**
   * 最初から学ぶ。AdamW の状態は 0 で、学習率の制御はウォームアップから始める。
   *
   * @param canonical 学習するネットワークのモデル（ブロックは {@link ClassicNetwork}、デバイスは学習のデバイスの先頭）
   * @param devices 学習のデバイス。先頭が主のデバイスで、2 つ目からはネットワークのレプリカを置くレーン
   * @param runtime GPU のレーンが計算のストリームを借りるランタイム（CPU だけなら null でよい。閉じるのは呼び出し元）
   * @param settings 学習の設定
   */
  public SelfPlayLearner(Model canonical, NetworkDevices devices, NetworkRuntime runtime, Settings settings) {
    if (!(canonical.getBlock() instanceof ClassicNetwork block)) {
      throw new IllegalArgumentException("The model does not hold a classic network");
    }
    if (!canonical.getNDManager().getDevice().equals(devices.primary())) {
      throw new IllegalArgumentException("The model is not on the primary learner device " + devices.primary());
    }
    network = block;
    manager = canonical.getNDManager();
    this.settings = Objects.requireNonNull(settings, "settings");
    actorRate = new KlLearningRate(settings.actorRate());
    actorLearningRate = (float) actorRate.rate();
    actorOptimizer =
        Optimizer.adamW()
            .optLearningRateTracker(
                new Tracker() {
                  @Override
                  public float getNewValue(int numUpdate) {
                    return actorLearningRate;
                  }
                })
            .optWeightDecays((float) settings.ppo().weightDecay())
            .build();
    valueOptimizer =
        Optimizer.adamW()
            .optLearningRateTracker(Tracker.fixed((float) settings.valueRate().rate()))
            .optWeightDecays((float) settings.valueRate().weightDecay())
            .build();
    group = lanes(canonical, devices, runtime);
  }

  /**
   * 書き出した状態から続ける。
   *
   * @param canonical 学習するネットワークのモデル（コンストラクターと同じ）
   * @param devices 学習のデバイス（コンストラクターと同じ）
   * @param runtime GPU のレーンが計算のストリームを借りるランタイム（コンストラクターと同じ）
   * @param settings 学習の設定
   * @param state {@link #save} が書いたディレクトリ
   * @return 状態を読み込んだ学習器
   * @throws IllegalArgumentException 状態がほかの形のネットワークか、ほかの設定で作られた
   */
  public static SelfPlayLearner resume(
      Model canonical, NetworkDevices devices, NetworkRuntime runtime, Settings settings, Path state)
      throws IOException {
    SelfPlayLearner learner = new SelfPlayLearner(canonical, devices, runtime, settings);
    try {
      learner.load(state);
      return learner;
    } catch (IOException | RuntimeException | Error failure) {
      try {
        learner.close();
      } catch (RuntimeException closing) {
        failure.addSuppressed(closing);
      }
      throw failure;
    }
  }

  public Settings settings() {
    return settings;
  }

  /** 方策の学習率の制御（次の macro の学習率と状態）。 */
  public KlLearningRate actorRate() {
    return actorRate;
  }

  /** 学んだ macro の数。 */
  public long macros() {
    return macros;
  }

  /** 方策を更新したステップの累計。 */
  public long actorSteps() {
    return actorSteps;
  }

  /** 価値を更新したステップの累計。 */
  public long valueSteps() {
    return valueSteps;
  }

  /**
   * 1 回の macro の軌跡で学習する。判断の並べ替えは乱数の種だけで決まる。
   *
   * @param batch 1 回の macro で集めた軌跡
   * @param seed この macro の並べ替えに使う乱数の種
   * @return この macro の学習の記録
   * @throws IllegalArgumentException ルールの違う軌跡
   * @throws IllegalStateException 勾配が有限でない（どの組も更新せずに勾配を捨てる）
   */
  public Macro learn(TrajectoryBatch batch, long seed) throws IOException {
    if (batch.variant() != network.variant()) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + network.variant() + " but found " + batch.variant());
    }
    long started = System.nanoTime();
    Index index = Index.of(batch);
    int size = batch.size();
    float[] rolloutLog = new float[index.offsets[size]];
    float[] values = new float[size];
    evaluate(batch, index, rolloutLog, values);
    float[] credits = new float[size];
    double creditSum = 0;
    for (int decision = 0; decision < size; decision++) {
      if (index.actor[decision]) {
        GameTrajectory game = batch.games().get(index.game[decision]);
        int local = index.local[decision];
        double rollout = Math.exp(rolloutLog[index.offsets[decision] + game.chosen(local)]);
        credits[decision] = (float) ValueTargets.credit(settings.ppo().credit(), rollout, game.behaviour(local));
        creditSum += credits[decision];
      } else {
        credits[decision] = 1f;
      }
    }
    float[] targets = new float[size];
    float[] advantages = new float[size];
    ValueTargets.compute(
        batch, network.utility(), values, credits, settings.vtrace().lambda(), targets, advantages);
    double advantageSum = 0;
    for (int decision = 0; decision < size; decision++) {
      advantageSum += index.actor[decision] ? advantages[decision] : 0;
    }
    long evaluated = System.nanoTime();
    Labels labels = new Labels(batch, index, rolloutLog, credits, targets, advantages);
    int[] order = shuffled(size, seed);
    int steps = Math.min(settings.ppo().steps(), size);
    List<Step> records = new ArrayList<>(steps);
    for (int step = 0; step < steps; step++) {
      int from = (int) ((long) size * step / steps);
      int to = (int) ((long) size * (step + 1) / steps);
      records.add(step(step, labels, order, from, to));
    }
    double klSum = 0;
    int measured = 0;
    int updated = 0;
    for (Step record : records) {
      if (record.actorRows() > 0) {
        klSum += record.kl();
        measured++;
      }
      updated += record.actorUpdated() ? 1 : 0;
    }
    double kl = measured == 0 ? 0 : klSum / measured;
    KlLearningRate.Adjustment adjustment = updated > 0 ? actorRate.accept(kl, updated) : actorRate.reject(kl);
    actorLearningRate = (float) actorRate.rate();
    macros++;
    long finished = System.nanoTime();
    Macro macro =
        new Macro(
            batch.games().size(),
            size,
            index.actorRows,
            records,
            kl,
            index.actorRows == 0 ? Double.NaN : creditSum / index.actorRows,
            index.actorRows == 0 ? Double.NaN : advantageSum / index.actorRows,
            adjustment,
            (evaluated - started) / 1e9,
            (finished - evaluated) / 1e9);
    LOG.info(
        "classic learn macro {} games={} decisions={} actorRows={} actorSteps={}/{} kl={} credit={} advantage={}"
            + " rate {} {} -> {} evaluate={}s steps={}s",
        macros, macro.games(), size, index.actorRows, updated, steps, kl, macro.credit(), macro.advantage(),
        adjustment.action(), adjustment.applied(), adjustment.next(), macro.evaluateSeconds(), macro.stepSeconds());
    return macro;
  }

  /**
   * 途中の状態を新しいディレクトリに書く。&lt;dir&gt;.tmp に全てを書いてから名前を変えるので、&lt;dir&gt; があれば中身はそろっている。
   *
   * @param directory 書き出すディレクトリ（まだ無いこと）
   * @throws FileAlreadyExistsException ディレクトリが既にある
   */
  public void save(Path directory) throws IOException {
    if (Files.exists(directory)) {
      throw new FileAlreadyExistsException(directory.toString(), null, "Learner state directory already exists");
    }
    group.awaitParameterUpdates();
    Path temporary = directory.resolveSibling(directory.getFileName() + TEMPORARY);
    if (Files.exists(temporary)) {
      DurableFiles.delete(temporary);
    }
    try {
      Files.createDirectories(temporary);
      actorOptimizer.saveState(temporary.resolve(ACTOR_STATE));
      valueOptimizer.saveState(temporary.resolve(VALUE_STATE));
      sync(temporary.resolve(ACTOR_STATE));
      sync(temporary.resolve(VALUE_STATE));
      Saved saved =
          new Saved(
              network.architecture().name(), elements(Parameters.ACTOR), elements(Parameters.VALUE), macros,
              actorSteps, valueSteps, actorRate.state());
      DurableFiles.write(temporary.resolve(STATE), GSON.toJson(saved).getBytes(StandardCharsets.UTF_8));
      Files.move(temporary, directory, StandardCopyOption.ATOMIC_MOVE);
    } catch (IOException | RuntimeException failure) {
      try {
        if (Files.exists(temporary)) {
          DurableFiles.delete(temporary);
        }
      } catch (IOException cleanup) {
        failure.addSuppressed(cleanup);
      }
      throw failure;
    }
  }

  /** 重みの更新が終わるのを待ち、レプリカのレーンを閉じて、方策と価値の AdamW の状態を捨てる。重みは渡されたネットワークに残る。 */
  @Override
  public void close() {
    try {
      group.awaitParameterUpdates();
    } finally {
      try (actorOptimizer;
          valueOptimizer) {
        group.close();
      }
    }
  }

  /** 書き出した状態を読み込む。まだ macro を学んでいない学習器で呼ぶ。 */
  private void load(Path directory) throws IOException {
    Saved saved;
    try {
      saved = GSON.fromJson(Files.readString(directory.resolve(STATE), StandardCharsets.UTF_8), Saved.class);
    } catch (RuntimeException invalid) {
      throw new IOException("Invalid learner state " + directory, invalid);
    }
    if (saved == null) {
      throw new IOException("Empty learner state " + directory);
    }
    if (!saved.architecture().equals(network.architecture().name())
        || saved.actorElements() != elements(Parameters.ACTOR)
        || saved.valueElements() != elements(Parameters.VALUE)) {
      throw new IllegalArgumentException(
          "The learner state " + directory + " was saved for another network: " + saved.architecture() + " "
              + saved.actorElements() + "/" + saved.valueElements());
    }
    actorRate = new KlLearningRate(settings.actorRate(), saved.actorRate());
    actorLearningRate = (float) actorRate.rate();
    actorOptimizer.loadState(manager, directory.resolve(ACTOR_STATE));
    valueOptimizer.loadState(manager, directory.resolve(VALUE_STATE));
    macros = saved.macros();
    actorSteps = saved.actorSteps();
    valueSteps = saved.valueSteps();
  }

  /** 学習の前のネットワークで全ての判断を推論し、合法手の対数確率と価値を書く。 */
  private void evaluate(TrajectoryBatch batch, Index index, float[] rolloutLog, float[] values) throws IOException {
    int size = batch.size();
    if (size == 0) {
      return;
    }
    int[] all = new int[size];
    for (int decision = 0; decision < size; decision++) {
      all[decision] = decision;
    }
    List<List<int[]>> assigned = assign(parts(all, 0, size, index), index);
    List<LaneWork<NetworkLane, Boolean>> work = new ArrayList<>();
    int[] lanes = active(assigned);
    for (int lane : lanes) {
      List<int[]> parts = assigned.get(lane);
      work.add(worker -> worker.evaluate(batch, index, parts, rolloutLog, values));
    }
    group.execute(lanes, work);
  }

  /** 1 ステップ学習する。並べ替えた判断のうち [from, to) の勾配を溜めて更新する。 */
  private Step step(int number, Labels labels, int[] order, int from, int to) throws IOException {
    Index index = labels.index();
    int actorRows = 0;
    for (int position = from; position < to; position++) {
      actorRows += index.actor[order[position]] ? 1 : 0;
    }
    List<List<int[]>> assigned = assign(parts(order, from, to, index), index);
    int[] lanes = active(assigned);
    PpoLoss.Coefficients coefficients =
        new PpoLoss.Coefficients(settings.ppo().clip(), settings.ppo().entropy(), actorRows, to - from);
    List<LaneWork<NetworkLane, float[]>> work = new ArrayList<>();
    for (int lane : lanes) {
      List<int[]> parts = assigned.get(lane);
      work.add(worker -> worker.train(labels, parts, coefficients));
    }
    double[] sums = new double[SUMS.length];
    boolean actorUpdated;
    try {
      for (float[] laneSums : group.execute(lanes, work)) {
        for (int sum = 0; sum < sums.length; sum++) {
          sums[sum] += laneSums[sum];
        }
      }
      double kl = actorRows == 0 ? 0 : sums[PpoLoss.Sum.KL.ordinal()] / actorRows;
      actorUpdated = actorRows > 0 && kl <= settings.ppo().klLimit();
      group.forgetNorms();
      List<Update> updates = new ArrayList<>(2);
      if (actorUpdated) {
        updates.add(new Update(Parameters.ACTOR, actorOptimizer, 1f));
      }
      updates.add(new Update(Parameters.VALUE, valueOptimizer, 1f));
      group.step(lanes, updates);
      if (!actorUpdated) {
        group.clearGradients();
      }
    } catch (IOException | RuntimeException | Error failure) {
      try {
        group.clearGradients();
      } catch (RuntimeException | Error cleanup) {
        failure.addSuppressed(cleanup);
      }
      throw failure;
    }
    actorSteps += actorUpdated ? 1 : 0;
    valueSteps++;
    double actors = Math.max(1, actorRows);
    Step record =
        new Step(
            number,
            to - from,
            actorRows,
            actorRows == 0 ? 0 : sums[PpoLoss.Sum.KL.ordinal()] / actorRows,
            actorUpdated,
            sums[PpoLoss.Sum.SURROGATE.ordinal()] / actors,
            sums[PpoLoss.Sum.ENTROPY.ordinal()] / actors,
            sums[PpoLoss.Sum.VALUE.ordinal()] / (to - from),
            sums[PpoLoss.Sum.CLIPPED.ordinal()] / actors,
            group.norm(Parameters.ACTOR),
            group.norm(Parameters.VALUE));
    LOG.info(
        "classic learn step {} rows={} actorRows={} kl={} actorUpdated={} surrogate={} entropy={} value={} clipped={}"
            + " norms={}/{}",
        number, record.rows(), actorRows, record.kl(), actorUpdated, record.surrogate(), record.entropy(),
        record.valueLoss(), record.clipFraction(), record.actorNorm(), record.valueNorm());
    return record;
  }

  /** 判断を推論の区分ごとにまとめ、区分の行の上限で切ってバッチにする。バッチは区分の順に並び、区分の中の判断は与えた順のまま。 */
  private List<int[]> parts(int[] decisions, int from, int to, Index index) {
    int[] counts = new int[Buckets.count()];
    for (int position = from; position < to; position++) {
      counts[index.bucket[decisions[position]]]++;
    }
    int[][] grouped = new int[Buckets.count()][];
    for (int bucket = 0; bucket < grouped.length; bucket++) {
      grouped[bucket] = new int[counts[bucket]];
    }
    int[] filled = new int[Buckets.count()];
    for (int position = from; position < to; position++) {
      int decision = decisions[position];
      grouped[index.bucket[decision]][filled[index.bucket[decision]]++] = decision;
    }
    List<int[]> parts = new ArrayList<>();
    for (int bucket = 0; bucket < grouped.length; bucket++) {
      int limit = rowLimit(Buckets.of(bucket), settings.batches());
      for (int start = 0; start < grouped[bucket].length; start += limit) {
        parts.add(Arrays.copyOfRange(grouped[bucket], start, Math.min(grouped[bucket].length, start + limit)));
      }
    }
    return parts;
  }

  /**
   * バッチをレーンに配る。セル（行 × 合法手 × 遷移）の多いバッチから順に、それまでに配ったセルの和が最も小さいレーンへ入れる。セルの
   * 数が同じなら、バッチもレーンも番号の小さい方を先にする。
   */
  private List<List<int[]>> assign(List<int[]> parts, Index index) {
    int lanes = group.laneCount();
    List<List<int[]>> assigned = new ArrayList<>(lanes);
    for (int lane = 0; lane < lanes; lane++) {
      assigned.add(new ArrayList<>());
    }
    if (lanes == 1) {
      assigned.getFirst().addAll(parts);
      return assigned;
    }
    long[] cells = new long[parts.size()];
    Integer[] sorted = new Integer[parts.size()];
    for (int part = 0; part < sorted.length; part++) {
      int[] decisions = parts.get(part);
      Bucket bucket = Buckets.of(index.bucket[decisions[0]]);
      sorted[part] = part;
      cells[part] = (long) decisions.length * bucket.actions() * bucket.transitions();
    }
    Arrays.sort(sorted, (left, right) -> cells[left] != cells[right] ? Long.compare(cells[right], cells[left]) : left - right);
    long[] load = new long[lanes];
    for (int part : sorted) {
      int lightest = 0;
      for (int lane = 1; lane < lanes; lane++) {
        if (load[lane] < load[lightest]) {
          lightest = lane;
        }
      }
      load[lightest] += cells[part];
      assigned.get(lightest).add(parts.get(part));
    }
    return assigned;
  }

  /** バッチを配ったレーンの番号。 */
  private static int[] active(List<List<int[]>> assigned) {
    return IntStream.range(0, assigned.size()).filter(lane -> !assigned.get(lane).isEmpty()).toArray();
  }

  private static int rowLimit(Bucket bucket, Batches batches) {
    return Math.max(1, Math.min(batches.rows(), batches.cells() / (bucket.actions() * bucket.transitions())));
  }

  /** 0 以上 {@code size} 未満の番号を、乱数の種で並べ替えた列。 */
  private static int[] shuffled(int size, long seed) {
    int[] order = new int[size];
    for (int index = 0; index < size; index++) {
      order[index] = index;
    }
    SplittableRandom random = new SplittableRandom(SeedMixer.mix64(seed ^ SHUFFLE_SALT));
    for (int index = size - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int held = order[index];
      order[index] = order[other];
      order[other] = held;
    }
    return order;
  }

  /** 組に入るパラメーターの要素数の和。 */
  private long elements(Parameters parameters) {
    long elements = 0;
    for (Pair<String, Parameter> pair : network.getParameters()) {
      if (parameters.includes(pair.getKey())) {
        elements += pair.getValue().getArray().size();
      }
    }
    return elements;
  }

  private static void sync(Path file) throws IOException {
    try (FileChannel channel = FileChannel.open(file, StandardOpenOption.WRITE)) {
      channel.force(true);
    }
  }

  /** レーンを作る。0 番は渡されたネットワークで、ほかは同じ形のネットワークに重みを写したレプリカ。 */
  private ParallelOptimizer lanes(Model canonical, NetworkDevices devices, NetworkRuntime runtime) {
    List<NetworkLane> lanes = new ArrayList<>(devices.size());
    try {
      requireGradients(network);
      lanes.add(new NetworkLane(canonical, network, false, runtime, settings.batches()));
      for (int index = 1; index < devices.size(); index++) {
        Device device = devices.get(index);
        Model replica = Model.newInstance("classic-learner-" + index, device, "PyTorch");
        try {
          ClassicNetwork copy = new ClassicNetwork(network.architecture(), network.variant(), network.utility());
          replica.setBlock(copy);
          copy.initialize(replica.getNDManager());
          DataParallelGroup.copyParameters(canonical, replica);
          requireGradients(copy);
          lanes.add(new NetworkLane(replica, copy, true, runtime, settings.batches()));
        } catch (RuntimeException | Error failure) {
          replica.close();
          throw failure;
        }
      }
      return new ParallelOptimizer(
          lanes, (float) settings.ppo().gradientClip(), (float) settings.valueRate().gradientClip());
    } catch (RuntimeException | Error failure) {
      for (NetworkLane lane : lanes) {
        try {
          lane.close();
        } catch (RuntimeException closing) {
          failure.addSuppressed(closing);
        }
      }
      throw failure;
    }
  }

  private static void requireGradients(ClassicNetwork network) {
    for (Pair<String, Parameter> pair : network.getParameters()) {
      pair.getValue().getArray().setRequiresGradient(true);
    }
  }

  /** 方策の損失と価値の損失がそれぞれ勾配を流すパラメーターの組。 */
  private enum Parameters implements DataParallelGroup.ParameterGroup {
    ACTOR,
    VALUE;

    @Override
    public boolean includes(String parameter) {
      return ClassicNetwork.valueParameter(parameter) == (this == VALUE);
    }

    @Override
    public boolean allowsMissingGradients() {
      return true;
    }

    @Override
    public boolean allowsZeroGradient() {
      return false;
    }
  }

  /**
   * 学習で引く、判断ごとの値。どの配列も判断の通しの番号の順。
   *
   * @param game 半荘の番号
   * @param local 半荘の中の判断の番号
   * @param bucket 推論の区分の通し番号
   * @param actor 方策の損失に入れるなら true
   * @param offsets 合法手の対数確率を並べた配列での、判断ごとの先頭の位置（末尾の要素は合法手の総数）
   * @param actorRows 方策の損失に入れる判断の数
   */
  private record Index(int[] game, int[] local, byte[] bucket, boolean[] actor, int[] offsets, int actorRows) {

    /** 全ての判断を読んで作る（判断を等分し、並列に読む）。 */
    static Index of(TrajectoryBatch batch) {
      int size = batch.size();
      int[] game = new int[size];
      int[] local = new int[size];
      byte[] bucket = new byte[size];
      boolean[] actor = new boolean[size];
      byte[] legal = new byte[size];
      for (int at = 0; at < batch.games().size(); at++) {
        int start = batch.start(at);
        for (int decision = 0; decision < batch.games().get(at).size(); decision++) {
          game[start + decision] = at;
          local[start + decision] = decision;
        }
      }
      Ranges.split(size, (from, to) -> {
        Observation reader = new Observation(batch.variant());
        for (int decision = from; decision < to; decision++) {
          GameTrajectory trajectory = batch.games().get(game[decision]);
          Observation o = trajectory.bind(local[decision], reader);
          bucket[decision] = (byte) Buckets.ordinal(Buckets.inference(o));
          legal[decision] = (byte) o.legalActions();
          actor[decision] = trajectory.causal(local[decision]) && o.legalActions() > 1;
        }
        return null;
      });
      int[] offsets = new int[size + 1];
      int actorRows = 0;
      for (int decision = 0; decision < size; decision++) {
        offsets[decision + 1] = Math.addExact(offsets[decision], legal[decision]);
        actorRows += actor[decision] ? 1 : 0;
      }
      return new Index(game, local, bucket, actor, offsets, actorRows);
    }

    int legal(int decision) {
      return offsets[decision + 1] - offsets[decision];
    }
  }

  /** ステップで使うバッチの教師を作るための値。軌跡、判断ごとの値、学習の前の対数確率、補正、価値の目標、アドバンテージ。 */
  private record Labels(
      TrajectoryBatch batch, Index index, float[] rolloutLog, float[] credits, float[] targets, float[] advantages) {}

  /**
   * 1 つのデバイスのネットワークと、入力と教師を書く作業領域。レーンの処理は、1 度に 1 つのスレッドだけが走らせる。
   */
  private static final class NetworkLane extends DataParallelGroup.Lane {

    private final ClassicNetwork network;
    private final Precision precision;
    private final ClassicSchema schema;
    private final ClassicTokenizer tokenizer;
    private final Observation reader;
    private final Batches batches;

    /** 入力と教師を転送するための、使い回すページ固定のバッファ。レーンの処理を回すスレッドが、1 度に 1 つだけ使う。 */
    private final PinnedUploads uploads;

    /** [区分の通し番号] → 入力（初めて使うときに作る）。 */
    private final ClassicInputs[] inputs = new ClassicInputs[Buckets.count()];

    /** 教師のバッファ。行 × 合法手の大きさまで伸ばして使い回す。 */
    private float[] legalMask = new float[0];

    private float[] chosenMask = new float[0];
    private float[] rolloutRows = new float[0];
    private float[] advantage = new float[0];
    private float[] credit = new float[0];
    private float[] actor = new float[0];
    private float[] target = new float[0];

    NetworkLane(Model model, ClassicNetwork network, boolean ownsModel, NetworkRuntime runtime, Batches batches) {
      super(model, ownsModel, runtime);
      this.network = network;
      this.batches = batches;
      precision = model.getNDManager().getDevice().isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
      schema = ClassicSchema.of(network.architecture());
      tokenizer = new ClassicTokenizer(network.architecture(), network.variant());
      reader = new Observation(network.variant());
      uploads = PinnedUploads.on(manager());
    }

    /** バッチの判断を推論し、合法手の対数確率と価値を書く。 */
    Boolean evaluate(TrajectoryBatch batch, Index index, List<int[]> parts, float[] rolloutLog, float[] values) {
      HlGauss support = network.support();
      for (int[] part : parts) {
        ClassicInputs into = encode(batch, index, part);
        int actions = into.bucket().actions();
        try (NDManager working = manager().newSubManager();
            InferenceMode ignored = manager().getEngine().newInferenceMode()) {
          ClassicBatch input = ClassicBatch.send(working, uploads, into, part.length);
          ClassicNetwork.Output output =
              network.forward(new ParameterStore(working, false), input, precision, false, ClassicNetwork.Heads.BOTH);
          float[] log = PolicyGraph.logProbabilities(output.policy(), input).toFloatArray();
          float[] logits = output.value().toFloatArray();
          for (int row = 0; row < part.length; row++) {
            int decision = part[row];
            System.arraycopy(log, row * actions, rolloutLog, index.offsets[decision], index.legal(decision));
            values[decision] = support.decode(logits, row * HlGauss.BINS);
          }
        }
      }
      return Boolean.TRUE;
    }

    /**
     * バッチの判断の勾配を溜める。
     *
     * @return 指標の和（{@link PpoLoss.Sum} の順）
     */
    float[] train(Labels labels, List<int[]> parts, PpoLoss.Coefficients coefficients) {
      try (NDArray totals = manager().zeros(new Shape(SUMS.length))) {
        for (int[] part : parts) {
          ClassicInputs into = encode(labels.batch(), labels.index(), part);
          int rows = part.length;
          int actions = into.bucket().actions();
          fill(labels, part, actions);
          try (NDManager working = manager().newSubManager()) {
            ClassicBatch input = ClassicBatch.send(working, uploads, into, rows);
            Shape cells = new Shape(rows, actions);
            NDArray legal = create(working, legalMask, cells);
            NDArray chosen = create(working, chosenMask, cells);
            NDArray rollout = create(working, rolloutRows, cells);
            NDArray rowAdvantage = create(working, advantage, new Shape(rows));
            NDArray rowCredit = create(working, credit, new Shape(rows));
            NDArray rowActor = create(working, actor, new Shape(rows));
            NDArray rowTarget = create(working, target, new Shape(rows));
            try (GradientCollector collector = newGradientCollector()) {
              ClassicNetwork.Output output =
                  network.forward(new ParameterStore(working, false), input, precision, true, ClassicNetwork.Heads.BOTH);
              NDArray logProbabilities = PolicyGraph.logProbabilities(output.policy(), input);
              PpoLoss.Terms terms =
                  PpoLoss.compute(
                      logProbabilities, legal, chosen, rollout, rowAdvantage, rowCredit, rowActor, output.value(),
                      network.support().targets(rowTarget), coefficients);
              collector.backward(terms.objective());
              totals.addi(terms.sums());
            }
          }
        }
        return totals.toFloatArray();
      }
    }

    /** 判断を、その区分の入力に先頭の行から書く。 */
    private ClassicInputs encode(TrajectoryBatch batch, Index index, int[] part) {
      int ordinal = index.bucket[part[0]];
      ClassicInputs into = inputs[ordinal];
      if (into == null) {
        Bucket bucket = Buckets.of(ordinal);
        into = new ClassicInputs(schema, bucket, rowLimit(bucket, batches));
        inputs[ordinal] = into;
      }
      for (int row = 0; row < part.length; row++) {
        int decision = part[row];
        GameTrajectory game = batch.games().get(index.game[decision]);
        int local = index.local[decision];
        Observation o = game.bind(local, reader);
        float[] prior = game.prior(game.hand(local));
        if (prior == null) {
          tokenizer.write(o, EXTRAS, into, row);
        } else {
          tokenizer.write(o, EXTRAS, prior, into, row);
        }
      }
      return into;
    }

    /** バッチの教師をバッファに書く。 */
    private void fill(Labels labels, int[] part, int actions) {
      int rows = part.length;
      int cells = rows * actions;
      if (legalMask.length < cells) {
        legalMask = new float[cells];
        chosenMask = new float[cells];
        rolloutRows = new float[cells];
      }
      if (advantage.length < rows) {
        advantage = new float[rows];
        credit = new float[rows];
        actor = new float[rows];
        target = new float[rows];
      }
      Arrays.fill(legalMask, 0, cells, 0f);
      Arrays.fill(chosenMask, 0, cells, 0f);
      Arrays.fill(rolloutRows, 0, cells, 0f);
      Index index = labels.index();
      for (int row = 0; row < rows; row++) {
        int decision = part[row];
        int legal = index.legal(decision);
        int base = row * actions;
        Arrays.fill(legalMask, base, base + legal, 1f);
        System.arraycopy(labels.rolloutLog(), index.offsets[decision], rolloutRows, base, legal);
        chosenMask[base + labels.batch().games().get(index.game[decision]).chosen(index.local[decision])] = 1f;
        advantage[row] = labels.advantages()[decision];
        credit[row] = labels.credits()[decision];
        actor[row] = index.actor[decision] ? 1f : 0f;
        target[row] = labels.targets()[decision];
      }
    }

    private NDArray create(NDManager manager, float[] values, Shape shape) {
      int size = Math.toIntExact(shape.size());
      return uploads.send(manager, DataType.FLOAT32, shape, buffer -> buffer.asFloatBuffer().put(values, 0, size));
    }

    @Override
    protected void closeRuntime() {
      uploads.close();
    }
  }

  /** データ並列のグループ。パラメーターの組ごとに勾配のノルムを測り、上限で切る。測ったノルムは次のステップまで残す。 */
  private static final class ParallelOptimizer extends DataParallelGroup<NetworkLane> {

    private final float actorClip;
    private final float valueClip;
    private final double[] norms = new double[Parameters.values().length];

    ParallelOptimizer(List<NetworkLane> lanes, float actorClip, float valueClip) {
      super(lanes);
      this.actorClip = actorClip;
      this.valueClip = valueClip;
      forgetNorms();
    }

    void forgetNorms() {
      Arrays.fill(norms, Double.NaN);
    }

    /** このステップで測った、組の勾配のノルム（更新しなかった組は NaN）。 */
    double norm(Parameters parameters) {
      return norms[parameters.ordinal()];
    }

    @Override
    protected void prepareGradient(ParameterGroup group, NDArray gradient) {
      Parameters parameters = (Parameters) group;
      float clip = parameters == Parameters.ACTOR ? actorClip : valueClip;
      try (NDManager scratch = gradient.getManager().newSubManager()) {
        NDArray squared = gradient.square();
        squared.attach(scratch);
        NDArray sum = squared.sum();
        sum.attach(scratch);
        double norm = Math.sqrt(sum.getFloat());
        if (!Double.isFinite(norm)) {
          throw new IllegalStateException("Non-finite " + parameters + " gradient norm " + norm);
        }
        norms[parameters.ordinal()] = norm;
        if (norm > clip) {
          gradient.muli((float) (clip / norm));
        }
      }
    }
  }
}
