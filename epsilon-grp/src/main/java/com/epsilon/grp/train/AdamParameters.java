package com.epsilon.grp.train;

import ai.djl.Device;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.GradientCollector;
import java.util.Arrays;
import java.util.SplittableRandom;
import java.util.function.IntFunction;

/**
 * DJL（PyTorch、CPU、float32）で学習する重みのテンソルをまとめて持ち、Adam で更新する（β1 0.9、β2 0.999、ε 1e-8）。重み減衰は
 * AdamW と同じく勾配とは切り離し、更新の前に掛ける。GRP、局の結果、prism の Sinkhorn の 3 つのネットワークを学習するのに使う。
 *
 * <p>テンソルごとに配列を 2 つ持つ。1 つは勾配を持つ配列で、学習の順伝播に使う。もう 1 つは同じメモリを指す勾配を持たない配列で、
 * 更新と、勾配の要らない順伝播に使う。更新は勾配を持たない側に書くので、学習するスレッドで torch の grad mode が有効でも無効でも
 * 結果は同じ。勾配は 1 ステップごとに 0 に戻す。配列はこのクラスの NDManager が持ち、{@link #close()} で解放する。
 */
public final class AdamParameters implements AutoCloseable {

  /** DJL のエンジンの名前。 */
  public static final String ENGINE = "PyTorch";

  private static final float BETA1 = 0.9f;
  private static final float BETA2 = 0.999f;
  private static final float EPSILON = 1e-8f;

  /** 順伝播や損失の式。テンソルの番号から重みの配列を返す関数を受け取り、渡されたマネージャーの上に結果の配列を作る。 */
  @FunctionalInterface
  public interface Forward {
    NDArray apply(NDManager scope, IntFunction<NDArray> weights);
  }

  private final NDManager manager;

  /** 重みの列の中で、各テンソルが始まる位置。最後の要素は重みの総数。 */
  private final int[] starts;

  /** テンソルごとに、隠れ層の重みなら true。重み減衰は隠れ層の重みにだけ掛ける。 */
  private final boolean[] hidden;

  private final NDArray[] trainable;
  private final NDArray[] values;
  private final NDArray[] firstMoments;
  private final NDArray[] secondMoments;
  private long steps;

  /**
   * 重みの列をテンソルごとに区切り、それぞれを配列にする。
   *
   * @param weights 全てのテンソルの重みを順に並べた列
   * @param starts テンソルごとの開始位置。最後の要素は重みの総数
   * @param forms テンソルごとの形
   * @param hidden テンソルごとに、隠れ層の重みなら true
   * @throws IllegalArgumentException 配列の長さが合わないとき
   */
  public AdamParameters(float[] weights, int[] starts, Shape[] forms, boolean[] hidden) {
    int blocks = forms.length;
    if (starts.length != blocks + 1 || hidden.length != blocks || weights.length != starts[blocks]) {
      throw new IllegalArgumentException(
          blocks + " blocks over " + starts.length + " starts and " + hidden.length + " hidden flags, got "
              + weights.length + " weights");
    }
    this.starts = starts.clone();
    this.hidden = hidden.clone();
    manager = NDManager.newBaseManager(Device.cpu(), ENGINE);
    trainable = new NDArray[blocks];
    values = new NDArray[blocks];
    firstMoments = new NDArray[blocks];
    secondMoments = new NDArray[blocks];
    for (int block = 0; block < blocks; block++) {
      trainable[block] = manager.create(Arrays.copyOfRange(weights, starts[block], starts[block + 1]), forms[block]);
      trainable[block].setRequiresGradient(true);
      values[block] = trainable[block].stopGradient();
      firstMoments[block] = manager.zeros(forms[block]);
      secondMoments[block] = manager.zeros(forms[block]);
    }
  }

  /**
   * ゼロから学習するときの初期の重みを作る。隠れ層の重みは He の初期値で、平均 0、分散 2 / fan-in の正規分布からテンソルの順に
   * 引く（fan-in は形の最初の次元）。ほかの重みは 0 にする。
   */
  public static float[] scratch(int[] starts, Shape[] forms, boolean[] hidden, SplittableRandom random) {
    float[] weights = new float[starts[forms.length]];
    for (int block = 0; block < forms.length; block++) {
      if (hidden[block]) {
        double deviation = Math.sqrt(2.0 / forms[block].get(0));
        for (int index = starts[block]; index < starts[block + 1]; index++) {
          weights[index] = (float) (random.nextGaussian() * deviation);
        }
      }
    }
    return weights;
  }

  /** バッチの列（容量の行数ぶん確保したもの）の先頭から、指定した形のぶんを配列にしてマネージャーの上に作る。 */
  public static NDArray rows(NDManager scope, float[] values, Shape shape) {
    int size = Math.toIntExact(shape.size());
    return scope.create(size == values.length ? values : Arrays.copyOf(values, size), shape);
  }

  /** バッチの列（容量の行数ぶん確保したもの）の先頭から、指定した形のぶんを配列にしてマネージャーの上に作る。 */
  public static NDArray rows(NDManager scope, long[] values, Shape shape) {
    int size = Math.toIntExact(shape.size());
    return scope.create(size == values.length ? values : Arrays.copyOf(values, size), shape);
  }

  /**
   * 行ごとの損失 w_i·NLL_i の和を行数で割った値を目的にして、それが下がる向きに 1 ステップ更新する。
   *
   * @param loss 行ごとの損失を作る式
   * @param rows 損失の和を割る行数
   * @param learningRate 学習率
   * @param hiddenDecay 隠れ層の重みに掛ける重み減衰
   * @param clipNorm 勾配全体のノルムの上限。超えたら上限まで縮める
   * @return 更新する前の重みでの損失と、勾配を縮めたかどうか
   * @throws IllegalStateException 損失か勾配のノルムが有限でないとき
   */
  public TrainingStep step(Forward loss, int rows, double learningRate, double hiddenDecay, double clipNorm) {
    double value;
    try (NDManager scope = manager.newSubManager()) {
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        NDArray objective = loss.apply(scope, this::trainable).sum().div(rows);
        collector.backward(objective);
        value = objective.getFloat();
      }
    }
    if (!Double.isFinite(value)) {
      throw new IllegalStateException("Non-finite training loss " + value);
    }
    return new TrainingStep(value, update(learningRate, hiddenDecay, clipNorm));
  }

  /**
   * 行ごとの損失 w_i·NLL_i の和を行数で割った値について、今の重みでの勾配を求める。重みと Adam のモーメントは変えない。求めた後、
   * テンソルの勾配は 0 に戻す。
   *
   * @param loss 行ごとの損失を作る式
   * @param rows 損失の和を割る行数
   * @return テンソルの順に並べた勾配
   */
  public float[] gradient(Forward loss, int rows) {
    try (NDManager scope = manager.newSubManager()) {
      try (GradientCollector collector = manager.getEngine().newGradientCollector()) {
        collector.backward(loss.apply(scope, this::trainable).sum().div(rows));
      }
    }
    float[] gradient = new float[starts[trainable.length]];
    for (int block = 0; block < trainable.length; block++) {
      try (NDArray part = trainable[block].getGradient()) {
        float[] values = part.toFloatArray();
        System.arraycopy(values, 0, gradient, starts[block], values.length);
        part.muli(0);
      }
    }
    return gradient;
  }

  /** 行ごとの損失 w_i·NLL_i の和を、勾配を作らずに求める。 */
  public double weightedNll(Forward loss) {
    try (NDManager scope = manager.newSubManager()) {
      return loss.apply(scope, this::value).sum().getFloat();
    }
  }

  /** 式を今の重みで計算し、行優先の float の列で返す。勾配は作らない。 */
  public float[] evaluate(Forward output) {
    try (NDManager scope = manager.newSubManager()) {
      return output.apply(scope, this::value).toFloatArray();
    }
  }

  /** 今の重みを、テンソルの順に並べて返す。 */
  public float[] weights() {
    float[] weights = new float[starts[values.length]];
    for (int block = 0; block < values.length; block++) {
      float[] part = values[block].toFloatArray();
      System.arraycopy(part, 0, weights, starts[block], part.length);
    }
    return weights;
  }

  @Override
  public void close() {
    manager.close();
  }

  private NDArray trainable(int block) {
    return trainable[block];
  }

  private NDArray value(int block) {
    return values[block];
  }

  /** 逆伝播の後に 1 ステップ更新し、勾配を 0 に戻す。勾配のノルムを上限まで縮めたら true を返す。 */
  private boolean update(double learningRate, double hiddenDecay, double clipNorm) {
    NDArray[] gradients = new NDArray[trainable.length];
    try {
      double squares = 0;
      for (int block = 0; block < trainable.length; block++) {
        gradients[block] = trainable[block].getGradient();
        try (NDArray square = gradients[block].square();
            NDArray sum = square.sum()) {
          squares += sum.getFloat();
        }
      }
      double norm = Math.sqrt(squares);
      if (!Double.isFinite(norm)) {
        throw new IllegalStateException("Non-finite gradient norm " + norm);
      }
      boolean clipped = norm > clipNorm;
      float scale = clipped ? (float) (clipNorm / norm) : 1f;
      steps++;
      for (int block = 0; block < trainable.length; block++) {
        step(block, gradients[block], scale, learningRate, hidden[block] ? hiddenDecay : 0);
      }
      return clipped;
    } finally {
      for (NDArray gradient : gradients) {
        if (gradient != null) {
          gradient.muli(0);
          gradient.close();
        }
      }
    }
  }

  /** PyTorch の Adam と同じ更新。重み減衰は AdamW と同じく、更新の前に掛ける。 */
  private void step(int block, NDArray gradient, float scale, double learningRate, double decay) {
    NDArray first = firstMoments[block];
    NDArray second = secondMoments[block];
    NDArray value = values[block];
    try (NDArray scaled = gradient.mul(scale)) {
      try (NDArray part = scaled.mul(1 - BETA1)) {
        first.muli(BETA1).addi(part);
      }
      try (NDArray square = scaled.square()) {
        square.muli(1 - BETA2);
        second.muli(BETA2).addi(square);
      }
    }
    double firstCorrection = 1 - Math.pow(BETA1, steps);
    double secondCorrection = 1 - Math.pow(BETA2, steps);
    if (decay > 0) {
      value.muli(1 - learningRate * decay);
    }
    try (NDArray denominator = second.sqrt()) {
      denominator.divi(Math.sqrt(secondCorrection)).addi(EPSILON);
      try (NDArray change = first.div(denominator)) {
        change.muli(learningRate / firstCorrection);
        value.subi(change);
      }
    }
  }
}
