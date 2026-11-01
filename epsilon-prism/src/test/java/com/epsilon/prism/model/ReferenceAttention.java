package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import com.epsilon.prism.hip.RmsNormKernels;
import com.epsilon.prism.hip.TrainingKernels;

/**
 * {@link TrainingKernels} の参照の実装（試験だけ）。配列をホストに写し、式をそのまま書いた繰り返しで計算して、配列の型で書き戻す（CPU の
 * FP32 でも GPU の BF16 でも使える）。{@link AttentionTape} のつなぎを自前の HIP の演算と切り離して確かめるために使う。注意の softmax は
 * 最大を引く 2 回の繰り返し、逆伝播は ds_ij = p_ij·(do_i·v_j − δ_i)。RMSNorm は {@link RmsNormKernels} の式を FP64 で計算し（r は
 * 逆伝播でも入力から計算し直す）、利得の勾配は同じ塊の部分和にする。
 */
final class ReferenceAttention implements TrainingKernels {

  @Override
  public void forward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray out,
      NDArray outFloat,
      NDArray logSumExp) {
    Inputs in = new Inputs(d, query, keyValue, mask, codes, table);
    int width = d.width();
    float[] o = new float[d.rows() * d.queries() * width];
    float[] lse = new float[d.rows() * d.heads() * d.queries()];
    for (int r = 0; r < d.rows(); r++) {
      for (int h = 0; h < d.heads(); h++) {
        for (int i = 0; i < d.queries(); i++) {
          double[] p = in.probabilities(r, h, i);
          double sum = p[d.keys()];
          for (int f = 0; f < FEATURES; f++) {
            double value = 0;
            for (int j = 0; j < d.keys(); j++) {
              value += p[j] * in.value(r, j, h, f);
            }
            o[(r * d.queries() + i) * width + h * FEATURES + f] = (float) value;
          }
          lse[(r * d.heads() + h) * d.queries() + i] = (float) sum;
        }
      }
    }
    write(out, o);
    write(outFloat, o);
    write(logSumExp, lse);
  }

  @Override
  public void backward(
      Dimensions d,
      NDArray query,
      NDArray keyValue,
      NDArray mask,
      NDArray codes,
      NDArray table,
      NDArray outFloat,
      NDArray logSumExp,
      NDArray gradOut,
      NDArray gradQuery,
      NDArray gradKeyValue,
      NDArray gradTable) {
    Inputs in = new Inputs(d, query, keyValue, mask, codes, table);
    int width = d.width();
    float[] o = outFloat.toFloatArray();
    float[] dout = gradOut.toType(DataType.FLOAT32, false).toFloatArray();
    float[] dq = new float[d.rows() * d.queries() * width];
    float[] dkv = new float[d.rows() * d.keys() * 2 * width];
    float[] drho = new float[d.rows() * d.heads() * d.tableRows()];
    for (int r = 0; r < d.rows(); r++) {
      for (int h = 0; h < d.heads(); h++) {
        for (int i = 0; i < d.queries(); i++) {
          int at = (r * d.queries() + i) * width + h * FEATURES;
          double delta = 0;
          for (int f = 0; f < FEATURES; f++) {
            delta += (double) dout[at + f] * o[at + f];
          }
          double[] p = in.probabilities(r, h, i);
          for (int j = 0; j < d.keys(); j++) {
            if (!in.present(r, j)) {
              continue;
            }
            double dp = 0;
            for (int f = 0; f < FEATURES; f++) {
              dp += dout[at + f] * in.value(r, j, h, f);
            }
            double ds = p[j] * (dp - delta);
            int kv = (r * d.keys() + j) * 2 * width + h * FEATURES;
            for (int f = 0; f < FEATURES; f++) {
              dq[at + f] += (float) (ds * in.key(r, j, h, f) * d.scale());
              dkv[kv + f] += (float) (ds * in.query(r, i, h, f) * d.scale());
              dkv[kv + width + f] += (float) (p[j] * dout[at + f]);
            }
            for (int s = 0; s < d.segments(); s++) {
              drho[(r * d.heads() + h) * d.tableRows() + s * d.entries() + in.code(r, i, j, s)] += (float) ds;
            }
          }
        }
      }
    }
    write(gradQuery, dq);
    write(gradKeyValue, dkv);
    write(gradTable, drho);
  }

  @Override
  public void forward(NormShape d, NDArray input, NDArray gain, NDArray out, NDArray inverseRms) {
    float[] x = input.toType(DataType.FLOAT32, false).toFloatArray();
    float[] g = gain.toFloatArray();
    float[] y = new float[x.length];
    float[] r = new float[d.rows()];
    for (int i = 0; i < d.rows(); i++) {
      double inverse = inverseRms(d, x, i);
      r[i] = (float) inverse;
      for (int j = 0; j < d.width(); j++) {
        int at = i * d.width() + j;
        y[at] = (float) (x[at] * inverse * g[j]);
      }
    }
    write(out, y);
    write(inverseRms, r);
  }

  @Override
  public void backward(
      NormShape d,
      NDArray input,
      NDArray gain,
      NDArray inverseRms,
      NDArray gradOut,
      NDArray gradInput,
      NDArray gradGain) {
    float[] x = input.toType(DataType.FLOAT32, false).toFloatArray();
    float[] g = gain.toFloatArray();
    float[] dy = gradOut.toType(DataType.FLOAT32, false).toFloatArray();
    float[] dx = new float[x.length];
    double[] sums = new double[d.partials() * d.width()];
    for (int i = 0; i < d.rows(); i++) {
      double inverse = inverseRms(d, x, i);
      int first = i * d.width();
      double m = 0;
      for (int j = 0; j < d.width(); j++) {
        m += (double) dy[first + j] * g[j] * x[first + j] * inverse;
      }
      m /= d.width();
      int block = i / RmsNormKernels.GAIN_ROWS * d.width();
      for (int j = 0; j < d.width(); j++) {
        double normalized = x[first + j] * inverse;
        dx[first + j] = (float) (inverse * ((double) dy[first + j] * g[j] - normalized * m));
        sums[block + j] += dy[first + j] * normalized;
      }
    }
    float[] partials = new float[sums.length];
    for (int k = 0; k < sums.length; k++) {
      partials[k] = (float) sums[k];
    }
    write(gradInput, dx);
    write(gradGain, partials);
  }

  /** 行 i の 1/rms（FP64）。 */
  private static double inverseRms(NormShape d, float[] x, int i) {
    double sum = 0;
    for (int j = 0; j < d.width(); j++) {
      double value = x[i * d.width() + j];
      sum += value * value;
    }
    return 1 / Math.sqrt(sum / d.width() + d.epsilon());
  }

  /** 値を配列の型にして書く（GPU の BF16 の配列にも書ける）。 */
  private static void write(NDArray target, float[] values) {
    try (NDArray source = target.getManager().create(values, target.getShape())) {
      if (target.getDataType() == DataType.FLOAT32) {
        source.copyTo(target);
      } else {
        try (NDArray cast = source.toType(target.getDataType(), false)) {
          cast.copyTo(target);
        }
      }
    }
  }

  /** 入力をホストの配列にしたもの。 */
  private static final class Inputs {
    private final Dimensions d;
    private final float[] q;
    private final float[] kv;
    private final float[] mask;
    private final int[] codes;
    private final float[] table;

    Inputs(Dimensions d, NDArray query, NDArray keyValue, NDArray mask, NDArray codes, NDArray table) {
      this.d = d;
      q = query.toType(DataType.FLOAT32, false).toFloatArray();
      kv = keyValue.toType(DataType.FLOAT32, false).toFloatArray();
      this.mask = mask.toType(DataType.FLOAT32, false).toFloatArray();
      this.codes = codes.toType(DataType.INT32, false).toIntArray();
      this.table = table.toFloatArray();
    }

    boolean present(int r, int j) {
      return mask[r * d.keys() + j] != 0f;
    }

    double query(int r, int i, int h, int f) {
      return q[(r * d.queries() + i) * d.width() + h * FEATURES + f];
    }

    double key(int r, int j, int h, int f) {
      return kv[(r * d.keys() + j) * 2 * d.width() + h * FEATURES + f];
    }

    double value(int r, int j, int h, int f) {
      return kv[(r * d.keys() + j) * 2 * d.width() + d.width() + h * FEATURES + f];
    }

    int code(int r, int i, int j, int s) {
      int word = codes[((r * d.codeQueries() + i) * d.keys() + j) * d.words() + s / 4] & 0xffff;
      return (word >> ((s % 4) * 4)) & 15;
    }

    /** 問い合わせ (r, h, i) の確率（在る鍵だけ）。末尾に log Σ exp を置く。 */
    double[] probabilities(int r, int h, int i) {
      double[] p = new double[d.keys() + 1];
      double maximum = Double.NEGATIVE_INFINITY;
      for (int j = 0; j < d.keys(); j++) {
        if (!present(r, j)) {
          continue;
        }
        double score = 0;
        for (int f = 0; f < FEATURES; f++) {
          score += query(r, i, h, f) * key(r, j, h, f);
        }
        score *= d.scale();
        for (int s = 0; s < d.segments(); s++) {
          score += table[(s * d.entries() + code(r, i, j, s)) * d.heads() + h];
        }
        p[j] = score;
        maximum = Math.max(maximum, score);
      }
      double sum = 0;
      for (int j = 0; j < d.keys(); j++) {
        p[j] = present(r, j) ? Math.exp(p[j] - maximum) : 0;
        sum += p[j];
      }
      for (int j = 0; j < d.keys(); j++) {
        p[j] = sum > 0 ? p[j] / sum : 0;
      }
      p[d.keys()] = sum > 0 ? maximum + Math.log(sum) : 0;
      return p;
    }
  }
}
