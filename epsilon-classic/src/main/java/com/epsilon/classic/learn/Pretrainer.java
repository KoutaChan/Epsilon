package com.epsilon.classic.learn;

import ai.djl.engine.InferenceMode;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Parameter;
import ai.djl.training.GradientCollector;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.classic.data.ClassicRows;
import com.epsilon.classic.data.RowChunks;
import com.epsilon.classic.input.Buckets;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicExtras;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicTokenizer;
import com.epsilon.classic.model.ClassicBatch;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.classic.model.PolicyGraph;
import com.epsilon.classic.rulebot.RuleBot;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.RankPermutations;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.util.SeedMixer;
import java.io.IOException;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;
import java.util.function.BooleanSupplier;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 牌譜の行で Nano・Major のネットワークを学習する（方策の模倣と価値の事前学習）。損失は 2 つの項の和。1 つは、方策のグラフで求めた
 * 記録の手の NLL の重み付き平均で、立直を選べる判断と他家への応答の判断は重みを大きくする。もう 1 つは、価値の HL-Gauss の交差
 * エントロピーに重みを掛けたもの。価値の目標は、その席が終局で取った順位の効用で、効用はネットワークが持つものを使う。ネットワークの
 * 作りにより、価値の勾配は共有の符号化の層へは流れない。
 *
 * <p>入力は行ごとに O から作り（{@link ClassicTokenizer}）、価値の入力には局の始まりの GRP の周辺確率を書く（GRP が無ければ一様）。行は推論の
 * 区分（{@link Buckets#inference}）ごとにまとめてバッチにする。バッチの行の数は、設定の行の上限と、合法手 × 遷移 × 行が {@link #MAX_CELLS}
 * に収まる数の小さい方。ステップごとの入力と教師は、使い回すページ固定のバッファ（{@link PinnedUploads}）から同期で転送する。最適化は
 * AdamW（重み減衰は分離型。{@link AdamW}）で、勾配全体のノルムを設定の上限で抑える。エポックごとの並べ替えは、乱数の種とエポックの番号で
 * 決まる。
 *
 * <p>エポックごとに、検証用の半荘の行で 3 つの値を測る。方策のグラフで貪欲に選んだ手が記録の手と一致する割合、記録の手の NLL、
 * 価値の損失。学習の終わりに、NLL が最良のエポックの重みをネットワークに戻す。基準として、一様に選んだときの一致率の期待値と NLL、
 * ルールベースの打ち手（{@link RuleBot}）の手が記録の手と一致する割合も測る。
 *
 * <p>中断して続きから学べる。状態のディレクトリを渡すと、中断のフラグが立ったステップの前と各エポックの終わりに、重み・AdamW の状態・
 * 最良の重み・進み具合を書く（{@link PretrainState}）。次に同じ設定と行で呼ぶと、その続きから学ぶ。並べ替えはエポックごとに決まるので、
 * 中断してもしなくても同じ順に学ぶ。
 */
public final class Pretrainer {

  private static final Logger LOG = LoggerFactory.getLogger(Pretrainer.class);

  /** 1 つのバッチの合法手 × 遷移 × 行の上限。 */
  public static final int MAX_CELLS = 16_384;

  private static final ClassicExtras EXTRAS = ClassicExtras.empty();

  private static final long SHUFFLE_SALT = 0x3C6E_F372_FE94_F82BL;

  /** 途中の損失をログに書くステップの間隔。 */
  private static final int PROGRESS_STEPS = 200;

  /**
   * 学習の設定。
   *
   * @param epochs 学ぶエポックの数
   * @param batchRows 1 ステップの行の上限
   * @param learningRate 学習率
   * @param weightDecay AdamW の重み減衰
   * @param valueWeight 価値の損失の重み
   * @param emphasis 立直を選べる判断と応答の判断の重み（ほかは 1）
   * @param clipNorm 勾配全体のノルムの上限
   * @param seed 並べ替えの乱数の種
   */
  public record Settings(
      int epochs,
      int batchRows,
      double learningRate,
      double weightDecay,
      double valueWeight,
      double emphasis,
      double clipNorm,
      long seed) {

    public Settings {
      if (epochs < 1 || batchRows < 1) {
        throw new IllegalArgumentException("Epochs and batch rows must be positive: " + epochs + ", " + batchRows);
      }
      if (!(learningRate > 0) || !(weightDecay >= 0) || !(valueWeight >= 0) || !(emphasis > 0) || !(clipNorm > 0)) {
        throw new IllegalArgumentException("Bad pretraining settings: " + this);
      }
    }

    /** エポックの数だけを替えた設定。 */
    public Settings withEpochs(int epochs) {
      return new Settings(epochs, batchRows, learningRate, weightDecay, valueWeight, emphasis, clipNorm, seed);
    }
  }

  /**
   * 検証用の行で測った基準。
   *
   * @param uniformAgreement 合法手から一様に選んだときの一致率の期待値
   * @param uniformNll 一様な分布の NLL（合法手の数の対数の平均）
   * @param ruleBotAgreement ルールベースの打ち手の手が記録の手と一致する割合
   */
  public record Baselines(double uniformAgreement, double uniformNll, double ruleBotAgreement) {}

  /**
   * 1 エポックの記録。
   *
   * @param epoch エポックの番号（0 始まり）
   * @param steps このエポックを終えた時点のステップの総数
   * @param trainLoss 学習のステップの損失の平均
   * @param agreement 検証用の行で、貪欲に選んだ手が記録の手と一致する割合
   * @param nll 検証用の行での、記録の手の NLL の平均（重みなし）
   * @param valueLoss 検証用の行での、価値の交差エントロピーの平均
   * @param seconds このエポックを終えたプロセスが、そのエポックに使った秒数（中断を挟んだエポックは、最後に再開してからの分だけ）
   */
  public record Epoch(int epoch, long steps, double trainLoss, double agreement, double nll, double valueLoss, double seconds) {}

  /**
   * 学習の結果。全てのエポックを終えたとき（状態のディレクトリを渡さずに中断したときも）、ネットワークにはそれまでで最良のエポックの
   * 重みが入っている。
   *
   * @param trainRows 学習用の行の数
   * @param validationRows 検証用の行の数
   * @param baselines 検証用の行で測った基準
   * @param epochs 終えたエポックの記録
   * @param best 検証の NLL が最良のエポック（0 始まり。終えたエポックが無ければ −1）
   * @param steps ステップの総数（中断の前の分を含む）
   * @param finished 全てのエポックを終えたなら true。中断のフラグで止めたなら false
   */
  public record Result(
      int trainRows, int validationRows, Baselines baselines, List<Epoch> epochs, int best, long steps, boolean finished) {

    /** 最良のエポックの記録（終えたエポックが無ければ null）。 */
    public Epoch bestEpoch() {
      return best < 0 ? null : epochs.get(best);
    }
  }

  private final ClassicNetwork network;
  private final NDManager manager;
  private final Precision precision;
  private final RowChunks rows;
  private final Settings settings;
  private final Variant variant;
  private final ClassicSchema schema;
  private final ClassicTokenizer tokenizer;
  private final Observation reader;

  /** [半荘][局] → 局の始まりの GRP の周辺確率（GRP が無ければ null）。 */
  private final float[][][] priors;

  /** 行ごとの値。推論の区分の通し番号、損失の重み、価値の目標にする効用の 3 つ。 */
  private final byte[] buckets;

  private final float[] weights;
  private final float[] utilities;

  /** [区分の通し番号] → バッチの入力（初めて使うときに作る）。 */
  private final ClassicInputs[] inputs = new ClassicInputs[Buckets.count()];

  /** ステップごとの入力と教師を転送するバッファ。学習するスレッドだけが使い、パラメーターのマネージャーに付ける。 */
  private final PinnedUploads uploads;

  private final PolicyGraph.Greedy greedy = new PolicyGraph.Greedy();
  private final List<Pair<String, Parameter>> parameters;
  private final AdamW optimizer;

  /** このプロセスで入力を作った時間と、順伝播・逆伝播・更新の時間（ナノ秒）。 */
  private long encodeNanos;

  private long computeNanos;

  /**
   * @param network 学習するネットワーク（パラメーターはマネージャーが持ち、学習で書き換える）
   * @param manager パラメーターのマネージャー（このマネージャーのデバイスで学習する）
   * @param precision 計算の型
   * @param rows 牌譜の行（学習用と検証用）
   * @param grp 局の始まりの周辺確率を求める GRP（null なら価値の入力に書く事前分布は一様）
   * @param settings 学習の設定
   */
  public Pretrainer(
      ClassicNetwork network,
      NDManager manager,
      Precision precision,
      RowChunks rows,
      GrpNetwork grp,
      Settings settings) {
    this.network = Objects.requireNonNull(network, "network");
    this.manager = Objects.requireNonNull(manager, "manager");
    this.precision = Objects.requireNonNull(precision, "precision");
    this.rows = Objects.requireNonNull(rows, "rows");
    this.settings = Objects.requireNonNull(settings, "settings");
    variant = network.variant();
    if (rows.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + rows.variant());
    }
    if (grp != null && grp.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + grp.variant());
    }
    schema = ClassicSchema.of(network.architecture());
    tokenizer = new ClassicTokenizer(network.architecture(), variant);
    reader = new Observation(variant);
    uploads = PinnedUploads.on(manager);
    priors = grp == null ? null : priors(grp);
    buckets = new byte[rows.size()];
    weights = new float[rows.size()];
    utilities = new float[rows.size()];
    Ranges.split(rows.size(), (from, to) -> {
      Observation scanner = new Observation(variant);
      for (int row = from; row < to; row++) {
        Observation o = rows.bind(row, scanner);
        buckets[row] = (byte) Buckets.ordinal(Buckets.inference(o));
        weights[row] = emphasized(o) ? (float) settings.emphasis() : 1f;
        utilities[row] = (float) network.utility().value(variant, rows.game(row).finalRank(o.observer()));
      }
      return null;
    });
    parameters = new ArrayList<>();
    List<NDArray> arrays = new ArrayList<>();
    for (Pair<String, Parameter> pair : network.getParameters()) {
      NDArray array = pair.getValue().getArray();
      array.setRequiresGradient(true);
      parameters.add(pair);
      arrays.add(array);
    }
    optimizer = new AdamW(arrays, settings.learningRate(), settings.weightDecay());
  }

  /** 立直を選べる判断か、他家への応答の判断（ロン・鳴き・見送り）なら true。 */
  static boolean emphasized(Observation o) {
    EngineDecisionKind kind = o.decisionKind();
    if (kind == EngineDecisionKind.RESPONSE || kind == EngineDecisionKind.CHANKAN
        || kind == EngineDecisionKind.NUKI_RESPONSE) {
      return true;
    }
    for (int action = 0; action < o.legalActions(); action++) {
      if (o.action(action).type() == Action.Type.RIICHI_DAHAI) {
        return true;
      }
    }
    return false;
  }

  /**
   * エポックの数だけ学び、最良のエポックの重みをネットワークに戻す。状態のディレクトリに進み具合があれば、その続きから学ぶ。
   *
   * @param state 状態のディレクトリ（null なら状態を読み書きしない）
   * @param pause 中断のフラグ（ステップごとに見る）。立つと、状態のディレクトリがあれば状態を書き、今の重みのまま終える。無ければ今の
   *     エポックは測らずに、それまでで最良の重みを戻して終える
   * @return 学習の結果
   * @throws IllegalArgumentException 学習用か検証用の行が無い。または状態が、ほかの設定・ネットワークの形・行で作られた
   * @throws IOException 状態を読めないか書けない
   */
  public Result train(Path state, BooleanSupplier pause) throws IOException {
    int[] train = select(false);
    int[] validation = select(true);
    if (train.length == 0 || validation.length == 0) {
      throw new IllegalArgumentException(
          "Pretraining needs training and validation rows: " + train.length + ", " + validation.length);
    }
    PretrainState.Progress progress = state == null ? null : PretrainState.progress(state);
    Baselines baselines;
    List<Epoch> epochs = new ArrayList<>();
    float[][] best = null;
    int bestEpoch = -1;
    int firstEpoch = 0;
    int firstBatch = 0;
    double resumedLoss = 0;
    long resumedSteps = 0;
    long steps = 0;
    int segment = 0;
    if (progress == null) {
      baselines = baselines(validation);
    } else {
      requireSame(progress, train.length, validation.length);
      PretrainState.Tensors tensors = PretrainState.tensors(state, progress);
      restore(tensors);
      baselines = progress.baselines();
      epochs.addAll(progress.epochs());
      bestEpoch = progress.best();
      best = tensors.best();
      firstEpoch = progress.epoch();
      firstBatch = progress.batch();
      resumedLoss = progress.lossSum();
      resumedSteps = progress.epochSteps();
      steps = progress.steps();
      segment = progress.segment() + 1;
      LOG.info("pretrain resume epoch {} batch {} steps {} from {}", firstEpoch, firstBatch, steps, state);
    }
    LOG.info(
        "pretrain {} rows={} validation={} uniform agreement={} nll={} rulebot agreement={}",
        network.architecture(), train.length, validation.length, baselines.uniformAgreement(),
        baselines.uniformNll(), baselines.ruleBotAgreement());
    boolean paused = false;
    long started = System.nanoTime();
    long trained = 0;
    for (int epoch = firstEpoch; epoch < settings.epochs() && !paused; epoch++) {
      long epochStarted = System.nanoTime();
      boolean resumed = epoch == firstEpoch;
      double lossSum = resumed ? resumedLoss : 0;
      long epochSteps = resumed ? resumedSteps : 0;
      List<int[]> batches = batches(train, epochRandom(epoch));
      for (int batch = resumed ? firstBatch : 0; batch < batches.size(); batch++) {
        if (pause.getAsBoolean()) {
          paused = true;
          if (state != null) {
            save(state, new PretrainState.Progress(
                settings, network.architecture().name(), train.length, validation.length, baselines, epochs,
                bestEpoch, epoch, batch, lossSum, epochSteps, steps, segment++), best);
            LOG.info("pretrain paused at epoch {} batch {} steps {}", epoch, batch, steps);
          }
          break;
        }
        int[] rowsOfBatch = batches.get(batch);
        lossSum += step(rowsOfBatch);
        epochSteps++;
        steps++;
        trained += rowsOfBatch.length;
        if (epochSteps % PROGRESS_STEPS == 0) {
          LOG.info(
              "pretrain epoch {} batch {}/{} train loss={} rows/s={} encode={}s compute={}s",
              epoch, batch + 1, batches.size(), lossSum / epochSteps,
              Math.round(trained / ((System.nanoTime() - started) / 1e9)), encodeNanos / 1e9, computeNanos / 1e9);
        }
      }
      if (paused) {
        break;
      }
      double[] measured = measure(validation);
      Epoch record =
          new Epoch(epoch, steps, lossSum / epochSteps, measured[0], measured[1], measured[2],
              (System.nanoTime() - epochStarted) / 1e9);
      epochs.add(record);
      LOG.info(
          "pretrain epoch {} steps={} train loss={} agreement={} nll={} value loss={} seconds={}",
          epoch, steps, record.trainLoss(), record.agreement(), record.nll(), record.valueLoss(), record.seconds());
      if (bestEpoch < 0 || record.nll() < epochs.get(bestEpoch).nll()) {
        bestEpoch = epoch;
        best = snapshot();
      }
      if (state != null) {
        save(state, new PretrainState.Progress(
            settings, network.architecture().name(), train.length, validation.length, baselines, epochs, bestEpoch,
            epoch + 1, 0, 0, 0, steps, segment++), best);
      }
    }
    if (best != null && (!paused || state == null)) {
      restoreWeights(best);
    }
    return new Result(train.length, validation.length, baselines, List.copyOf(epochs), bestEpoch, steps, !paused);
  }

  /** 状態が、同じ設定（エポックの数は除く）、同じネットワークの形、同じ行の数で作られたことを確かめる。 */
  private void requireSame(PretrainState.Progress progress, int trainRows, int validationRows) {
    if (!progress.settings().withEpochs(settings.epochs()).equals(settings)
        || !progress.architecture().equals(network.architecture().name())
        || progress.trainRows() != trainRows
        || progress.validationRows() != validationRows) {
      throw new IllegalArgumentException(
          "The pretraining state was made with other settings, network or rows: " + progress.settings() + " "
              + progress.architecture() + " rows " + progress.trainRows() + "/" + progress.validationRows() + " vs "
              + settings + " " + network.architecture() + " rows " + trainRows + "/" + validationRows);
    }
  }

  private void save(Path state, PretrainState.Progress progress, float[][] best) throws IOException {
    PretrainState.write(
        state, progress,
        new PretrainState.Tensors(
            names(), optimizer.steps(), snapshot(), optimizer.means(), optimizer.variances(), best));
  }

  private void restore(PretrainState.Tensors tensors) {
    if (!tensors.names().equals(names())) {
      throw new IllegalArgumentException("The pretraining state holds other parameters: " + tensors.names());
    }
    restoreWeights(tensors.weights());
    optimizer.restore(tensors.adamSteps(), tensors.means(), tensors.variances());
  }

  private List<String> names() {
    return parameters.stream().map(Pair::getKey).toList();
  }

  /** エポックの並べ替えに使う乱数（乱数の種とエポックの番号で決まる）。 */
  private SplittableRandom epochRandom(int epoch) {
    return new SplittableRandom(SeedMixer.mix64((settings.seed() ^ SHUFFLE_SALT) + epoch));
  }

  /** 検証用の行（引数が true のとき）か、学習用の行の番号を返す。 */
  private int[] select(boolean validation) {
    int count = 0;
    for (int row = 0; row < rows.size(); row++) {
      count += rows.validation(row) == validation ? 1 : 0;
    }
    int[] selected = new int[count];
    int next = 0;
    for (int row = 0; row < rows.size(); row++) {
      if (rows.validation(row) == validation) {
        selected[next++] = row;
      }
    }
    return selected;
  }

  /** 行を区分ごとに並べ替えてバッチに切り分け、バッチの順もさらに並べ替えて返す。 */
  private List<int[]> batches(int[] selected, SplittableRandom random) {
    List<int[]> batches = new ArrayList<>();
    for (int[] group : groups(selected)) {
      shuffle(group, random);
      int limit = rowLimit(Buckets.of(buckets[group[0]]));
      for (int from = 0; from < group.length; from += limit) {
        batches.add(Arrays.copyOfRange(group, from, Math.min(group.length, from + limit)));
      }
    }
    for (int index = batches.size() - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int[] held = batches.get(index);
      batches.set(index, batches.get(other));
      batches.set(other, held);
    }
    return batches;
  }

  /** 区分ごとの行の番号を、区分の通し番号の順に返す（行の無い区分は除く）。 */
  private List<int[]> groups(int[] selected) {
    int[] counts = new int[Buckets.count()];
    for (int row : selected) {
      counts[buckets[row]]++;
    }
    int[][] groups = new int[Buckets.count()][];
    for (int bucket = 0; bucket < groups.length; bucket++) {
      groups[bucket] = new int[counts[bucket]];
    }
    int[] filled = new int[Buckets.count()];
    for (int row : selected) {
      groups[buckets[row]][filled[buckets[row]]++] = row;
    }
    List<int[]> nonEmpty = new ArrayList<>();
    for (int[] group : groups) {
      if (group.length > 0) {
        nonEmpty.add(group);
      }
    }
    return nonEmpty;
  }

  private int rowLimit(Bucket bucket) {
    return Math.max(1, Math.min(settings.batchRows(), MAX_CELLS / (bucket.actions() * bucket.transitions())));
  }

  /** バッチの行を、その区分の入力に書いて返す。 */
  private ClassicInputs encode(int[] batch) {
    long started = System.nanoTime();
    int ordinal = buckets[batch[0]];
    ClassicInputs into = inputs[ordinal];
    if (into == null) {
      into = new ClassicInputs(schema, Buckets.of(ordinal), rowLimit(Buckets.of(ordinal)));
      inputs[ordinal] = into;
    }
    for (int index = 0; index < batch.length; index++) {
      int row = batch[index];
      Observation o = rows.bind(row, reader);
      if (priors == null) {
        tokenizer.write(o, EXTRAS, into, index);
      } else {
        tokenizer.write(o, EXTRAS, priors[rows.gameIndex(row)][rows.round(row)], into, index);
      }
    }
    encodeNanos += System.nanoTime() - started;
    return into;
  }

  /** 1 ステップ学習し、更新の前の損失を返す。 */
  private double step(int[] batch) {
    ClassicInputs into = encode(batch);
    long started = System.nanoTime();
    int count = batch.length;
    int actions = into.bucket().actions();
    double weightSum = 0;
    for (int row : batch) {
      weightSum += weights[row];
    }
    double loss;
    try (NDManager working = manager.newSubManager()) {
      ClassicBatch input = ClassicBatch.send(working, uploads, into, count);
      NDArray chosen =
          uploads.send(
              working,
              DataType.FLOAT32,
              new Shape(count, actions),
              buffer -> oneHot(buffer.asFloatBuffer(), batch, actions));
      NDArray rowWeight = rowValues(working, weights, batch);
      NDArray rowUtility = rowValues(working, utilities, batch);
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        ClassicNetwork.Output output =
            network.forward(new ParameterStore(working, false), input, precision, true, ClassicNetwork.Heads.BOTH);
        NDArray nll = PolicyGraph.logProbabilities(output.policy(), input).mul(chosen).sum(new int[] {1}).neg();
        NDArray policyLoss = nll.mul(rowWeight).sum().div((float) weightSum);
        NDArray valueLoss = valueLoss(output.value(), rowUtility).mean();
        NDArray objective = policyLoss.add(valueLoss.mul((float) settings.valueWeight()));
        collector.backward(objective);
        loss = objective.getFloat();
      }
    }
    if (!Double.isFinite(loss)) {
      throw new IllegalStateException("Non-finite pretraining loss " + loss);
    }
    update();
    computeNanos += System.nanoTime() - started;
    return loss;
  }

  /** 行ごとの価値の交差エントロピー [行]。 */
  private NDArray valueLoss(NDArray logits, NDArray rowUtility) {
    return network.support().targets(rowUtility).mul(logits.logSoftmax(1)).sum(new int[] {1}).neg();
  }

  /**
   * 勾配全体のノルムを上限で抑えてから、AdamW で更新する。ノルムはデバイスの上で足し合わせ、値は 1 度だけ読む。勾配は AdamW の更新が 0 に
   * 戻す。
   */
  private void update() {
    NDArray[] gradients = new NDArray[parameters.size()];
    try (NDManager scratch = manager.newSubManager()) {
      NDList squares = new NDList(gradients.length);
      for (int index = 0; index < gradients.length; index++) {
        gradients[index] = parameters.get(index).getValue().getArray().getGradient();
        try (NDArray square = gradients[index].square()) {
          NDArray sum = square.sum();
          sum.attach(scratch);
          squares.add(sum);
        }
      }
      NDArray stacked = NDArrays.stack(squares);
      stacked.attach(scratch);
      NDArray total = stacked.sum();
      total.attach(scratch);
      double norm = Math.sqrt(total.getFloat());
      if (!Double.isFinite(norm)) {
        throw new IllegalStateException("Non-finite gradient norm " + norm);
      }
      if (norm > settings.clipNorm()) {
        float scale = (float) (settings.clipNorm() / norm);
        for (NDArray gradient : gradients) {
          gradient.muli(scale);
        }
      }
      optimizer.step(gradients);
    } finally {
      for (NDArray gradient : gradients) {
        if (gradient != null) {
          gradient.close();
        }
      }
    }
  }

  /** 検証用の行で測った {一致率, NLL の平均, 価値の損失の平均} を返す。 */
  private double[] measure(int[] validation) {
    long agreed = 0;
    double nll = 0;
    double value = 0;
    for (int[] batch : batches(validation, new SplittableRandom(0))) {
      ClassicInputs into = encode(batch);
      int count = batch.length;
      int actions = into.bucket().actions();
      try (NDManager working = manager.newSubManager()) {
        // 転送のバッファは推論モードの外で取る（推論モードの中で取ると、推論用のテンソルになる）
        ClassicBatch input = ClassicBatch.send(working, uploads, into, count);
        NDArray rowUtility = rowValues(working, utilities, batch);
        try (InferenceMode ignored = manager.getEngine().newInferenceMode()) {
          ClassicNetwork.Output output =
              network.forward(new ParameterStore(working, false), input, precision, false, ClassicNetwork.Heads.BOTH);
          NDArray logProbabilities = PolicyGraph.logProbabilities(output.policy(), input);
          float[] log = logProbabilities.toFloatArray();
          float[] probabilities = PolicyGraph.probabilities(logProbabilities, input).toFloatArray();
          for (int index = 0; index < count; index++) {
            int chosen = rows.chosen(batch[index]);
            agreed += greedy.select(into, index, probabilities, index * actions) == chosen ? 1 : 0;
            nll -= log[index * actions + chosen];
          }
          value += valueLoss(output.value(), rowUtility).sum().getFloat();
        }
      }
    }
    return new double[] {agreed / (double) validation.length, nll / validation.length, value / validation.length};
  }

  /**
   * 行ごとの値をバッチの行の順に並べ、バッファからデバイスへ転送する。
   *
   * @param working 配列を持つマネージャー
   * @param values 行の番号で引く値
   * @param batch バッチの行の番号
   * @return [行]（FP32）
   */
  private NDArray rowValues(NDManager working, float[] values, int[] batch) {
    return uploads.send(working, DataType.FLOAT32, new Shape(batch.length), buffer -> {
      FloatBuffer into = buffer.asFloatBuffer();
      for (int row : batch) {
        into.put(values[row]);
      }
    });
  }

  /** 検証用の行で基準を測る（行を等分し、並列に測る）。 */
  private Baselines baselines(int[] validation) {
    List<double[]> parts = Ranges.split(validation.length, (from, to) -> {
      double uniformAgreement = 0;
      double uniformNll = 0;
      long ruleBot = 0;
      Observation scanner = new Observation(variant);
      try (RuleBot bot = new RuleBot(variant)) {
        for (int index = from; index < to; index++) {
          int row = validation[index];
          Observation o = rows.bind(row, scanner);
          uniformAgreement += 1.0 / o.legalActions();
          uniformNll += Math.log(o.legalActions());
          ruleBot += bot.act(o) == rows.chosen(row) ? 1 : 0;
        }
      }
      return new double[] {uniformAgreement, uniformNll, ruleBot};
    });
    double[] total = new double[3];
    for (double[] part : parts) {
      for (int index = 0; index < total.length; index++) {
        total[index] += part[index];
      }
    }
    return new Baselines(total[0] / validation.length, total[1] / validation.length, total[2] / validation.length);
  }

  /** 記録の手の one-hot [行, 合法手] を書く（記録の手で 1、ほかは 0）。 */
  private void oneHot(FloatBuffer into, int[] batch, int actions) {
    for (int row : batch) {
      int chosen = rows.chosen(row);
      for (int action = 0; action < actions; action++) {
        into.put(action == chosen ? 1f : 0f);
      }
    }
  }

  private float[][] snapshot() {
    float[][] values = new float[parameters.size()][];
    for (int index = 0; index < values.length; index++) {
      values[index] = parameters.get(index).getValue().getArray().toFloatArray();
    }
    return values;
  }

  private void restoreWeights(float[][] values) {
    if (values.length != parameters.size()) {
      throw new IllegalArgumentException("Weights of " + values.length + " tensors for " + parameters.size());
    }
    for (int index = 0; index < values.length; index++) {
      parameters.get(index).getValue().getArray().set(values[index]);
    }
  }

  /** [半荘][局] ごとに、局の始まりの GRP の周辺確率（絶対席 × 順位）を求める。 */
  private float[][][] priors(GrpNetwork grp) {
    GrpWorkspace workspace = GrpWorkspace.of(grp.shape());
    RankPermutations permutations = RankPermutations.of(variant);
    double[] probabilities = new double[permutations.count()];
    double[] marginals = new double[variant.seats() * variant.seats()];
    List<ClassicRows.Game> games = rows.games();
    float[][][] priors = new float[games.size()][][];
    for (int game = 0; game < games.size(); game++) {
      List<RoundStart> rounds = games.get(game).rounds();
      priors[game] = new float[rounds.size()][];
      for (int round = 0; round < rounds.size(); round++) {
        RoundStart start = rounds.get(round);
        GrpState state = new GrpState(variant, start.round(), start.honba(), start.kyotaku(), start.scores());
        grp.permutations(state, workspace, probabilities);
        permutations.marginals(probabilities, state.dealer(), marginals);
        float[] prior = new float[marginals.length];
        for (int index = 0; index < prior.length; index++) {
          prior[index] = (float) marginals[index];
        }
        priors[game][round] = prior;
      }
    }
    return priors;
  }

  private static void shuffle(int[] values, SplittableRandom random) {
    for (int index = values.length - 1; index > 0; index--) {
      int other = random.nextInt(index + 1);
      int held = values[index];
      values[index] = values[other];
      values[other] = held;
    }
  }
}
