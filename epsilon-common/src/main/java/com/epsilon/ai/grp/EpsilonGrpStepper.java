package com.epsilon.ai.grp;

import ai.djl.nn.Block;
import ai.djl.nn.Parameter;

/**
 * GRP の系列の隠れ状態と、そこから1局境界だけ進めた順位周辺確率を CPU で求める。重みは作ったときに写し、以後は変えない（スレッド安全）。
 *
 * <p>同じ系列の先に候補の局境界を何通りも足して評価するときに、系列の前半を毎回通し直さないためのもの。式は DJL（PyTorch エンジン）の GRU と
 * 同じで、ゲートの順は r・z・n、{@code n = tanh(W_in x + b_in + r ⊙ (W_hn h + b_hn))}、{@code h' = (1 − z) ⊙ n + z ⊙ h}。
 * GRU の出力の後は {@link EpsilonGrpNetwork} と同じく投影・ReLU・ロジット・Sinkhorn を通す。
 */
public final class EpsilonGrpStepper {
  private final int hidden;
  private final int layers;
  private final float[][] inputWeights;
  private final float[][] stateWeights;
  private final float[][] inputBiases;
  private final float[][] stateBiases;
  private final float[] projectionWeight;
  private final float[] projectionBias;
  private final float[] logitsWeight;
  private final float[] logitsBias;

  EpsilonGrpStepper(EpsilonGrpNetwork network) {
    hidden = network.hiddenSize();
    layers = network.layers();
    inputWeights = new float[layers][];
    stateWeights = new float[layers][];
    inputBiases = new float[layers][];
    stateBiases = new float[layers][];
    Block gru = network.gru();
    for (int layer = 0; layer < layers; layer++) {
      inputWeights[layer] = values(gru, "l_" + layer + "_i2h_WEIGHT");
      stateWeights[layer] = values(gru, "l_" + layer + "_h2h_WEIGHT");
      inputBiases[layer] = values(gru, "l_" + layer + "_i2h_BIAS");
      stateBiases[layer] = values(gru, "l_" + layer + "_h2h_BIAS");
    }
    projectionWeight = values(network.projection(), "weight");
    projectionBias = values(network.projection(), "bias");
    logitsWeight = values(network.logitsHead(), "weight");
    logitsBias = values(network.logitsHead(), "bias");
  }

  private static float[] values(Block block, String name) {
    Parameter parameter = block.getDirectParameters().get(name);
    if (parameter == null) throw new IllegalStateException("No parameter " + name + " in " + block);
    return parameter.getArray().toFloatArray();
  }

  /** 隠れ状態の長さ（層の数 × 隠れ表現の幅）。 */
  public int stateSize() {
    return layers * hidden;
  }

  /**
   * 系列全体を通した後の各層の隠れ状態。
   *
   * @param prefix 一次元化済みの GRP 特徴量系列（1局境界以上）
   * @return 層 0 から順に並べた隠れ状態（長さ {@link #stateSize()}）
   */
  public float[] hidden(float[] prefix) {
    int steps = EpsilonGrpFeature.steps(prefix);
    if (steps <= 0) throw new IllegalArgumentException("GRP sequence is empty");
    float[] state = new float[stateSize()];
    float[] next = new float[stateSize()];
    float[] input = new float[EpsilonGrpFeature.FEATURE_SIZE];
    for (int step = 0; step < steps; step++) {
      System.arraycopy(prefix, step * input.length, input, 0, input.length);
      advance(state, input, next);
      float[] swap = state;
      state = next;
      next = swap;
    }
    return state;
  }

  /**
   * {@link #hidden} の状態から局境界を1つ進めた順位周辺確率を、候補ごとに求める。
   *
   * @param state {@link #hidden} の結果。書き換えない
   * @param next 候補の局境界の特徴量（各長さ {@link EpsilonGrpFeature#FEATURE_SIZE}）
   * @param out 候補ごとの16要素の周辺確率（席ごとに4順位）を書く先
   */
  public void marginalsAfter(float[] state, float[][] next, float[][] out) {
    int count = next.length;
    float[] stepped = new float[stateSize()];
    float[] projected = new float[hidden];
    float[] logits = new float[count * EpsilonGrpRanks.MATRIX_SIZE];
    int top = (layers - 1) * hidden;
    for (int candidate = 0; candidate < count; candidate++) {
      advance(state, next[candidate], stepped);
      for (int unit = 0; unit < hidden; unit++) {
        float sum = projectionBias[unit];
        int row = unit * hidden;
        for (int k = 0; k < hidden; k++) sum += projectionWeight[row + k] * stepped[top + k];
        projected[unit] = Math.max(0f, sum);
      }
      int base = candidate * EpsilonGrpRanks.MATRIX_SIZE;
      for (int unit = 0; unit < EpsilonGrpRanks.MATRIX_SIZE; unit++) {
        float sum = logitsBias[unit];
        int row = unit * hidden;
        for (int k = 0; k < hidden; k++) sum += logitsWeight[row + k] * projected[k];
        logits[base + unit] = sum;
      }
    }
    float[] marginals = EpsilonGrpSinkhorn.marginalsBatch(logits);
    for (int candidate = 0; candidate < count; candidate++)
      System.arraycopy(
          marginals,
          candidate * EpsilonGrpRanks.MATRIX_SIZE,
          out[candidate],
          0,
          EpsilonGrpRanks.MATRIX_SIZE);
  }

  /** 全層を1歩進める。state と out は別の配列。 */
  private void advance(float[] state, float[] input, float[] out) {
    float[] layerInput = input;
    int inputOffset = 0;
    for (int layer = 0; layer < layers; layer++) {
      int width = layer == 0 ? input.length : hidden;
      float[] wi = inputWeights[layer], wh = stateWeights[layer];
      float[] bi = inputBiases[layer], bh = stateBiases[layer];
      int offset = layer * hidden;
      for (int unit = 0; unit < hidden; unit++) {
        float ir = bi[unit], iz = bi[hidden + unit], in = bi[2 * hidden + unit];
        float hr = bh[unit], hz = bh[hidden + unit], hn = bh[2 * hidden + unit];
        int r = unit * width, z = (hidden + unit) * width, n = (2 * hidden + unit) * width;
        for (int k = 0; k < width; k++) {
          float x = layerInput[inputOffset + k];
          ir += wi[r + k] * x;
          iz += wi[z + k] * x;
          in += wi[n + k] * x;
        }
        int rh = unit * hidden, zh = (hidden + unit) * hidden, nh = (2 * hidden + unit) * hidden;
        for (int k = 0; k < hidden; k++) {
          float h = state[offset + k];
          hr += wh[rh + k] * h;
          hz += wh[zh + k] * h;
          hn += wh[nh + k] * h;
        }
        float reset = sigmoid(ir + hr);
        float update = sigmoid(iz + hz);
        float candidate = (float) Math.tanh(in + reset * hn);
        out[offset + unit] = (1f - update) * candidate + update * state[offset + unit];
      }
      layerInput = out;
      inputOffset = offset;
    }
  }

  private static float sigmoid(float x) {
    return (float) (1.0 / (1.0 + Math.exp(-x)));
  }
}
