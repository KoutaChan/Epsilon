package com.epsilon.prism.hip;

/**
 * K9 の CPU の参照。K9 と同じ式を同じ所で丸め、積和は素朴に計算する（行列は FP32 で K の順に足し、Cross-Attention の softmax は FP64）。
 *
 * <pre>
 * q2 = ⌊⌊N_q(z1)⌉·W_qᵀ + b_q⌉
 * o2 = ⌊Σ_j softmax_j((q2·K_j + q2·E_k[c_j])·scale)·(V_j + E_v[c_j])⌉   j は行の局・席・牌とイベントのキー
 * z  = ⌊z1 + ⌊o2·W_Oᵀ + b_O⌉⌉
 * B  = Σ_g silu(⌊⌊N_B(z)⌉·W_Bᵀ + c_B⌉)·wB_g + Σ_k x^B_k·uBx_k
 * </pre>
 *
 * <p>キーのコード c_j は、局・席・牌のキーでは遷移のコードをそのまま使う。イベントのキーでは、τ とイベントの牌の両方があれば
 * 「遷移のコードの数 + 1 + 牌の関係の種類」、どちらかが無ければ「遷移のコードの数」。
 */
final class RereadReference {

  /**
   * 参照の出力。
   *
   * @param transitions z [遷移, H]（BF16 のビット）
   * @param values B [遷移]
   */
  record Result(short[] transitions, float[] values) {}

  private RereadReference() {}

  /**
   * 全ての遷移の z と B を計算する。
   *
   * @param c 入力
   * @return z と B
   */
  static Result run(RereadCase c) {
    HipArchitecture architecture = c.architecture;
    HipCounts counts = architecture.counts();
    HipPack.Header header = c.header;
    RereadCase.Weights w = c.weights;
    int h = architecture.hidden();
    int a = architecture.attention();
    int g = architecture.gate();
    int fixed = counts.fixed();
    int tiles = counts.tiles();
    float epsilon = architecture.epsilon();
    int categoryWidth = header.sectionWidth(HipSection.TRANSITION_CATEGORY);
    int callDiscard = header.schema(HipSchema.TRANSITION_CALL_DISCARD);
    int mainTile = header.schema(HipSchema.TRANSITION_MAIN_TILE);
    int exactWidth = header.sectionWidth(HipSection.DISCARD_EXACT);
    float[] query = values(w.query());
    float[] output = values(w.output());
    float[] hidden = values(w.discardHidden());
    short[] z = new short[c.transitions * h];
    float[] b = new float[c.transitions];
    for (int t = 0; t < c.transitions; t++) {
      int row = c.transitionRow[t];
      int afterCall = c.category[t * categoryWidth + callDiscard] & 0xff;
      int tau = afterCall > 0 ? afterCall : c.category[t * categoryWidth + mainTile] & 0xff;
      float[] start = new float[h];
      for (int i = 0; i < h; i++) {
        start[i] = MlpReference.value(c.start[t * h + i]);
      }
      float[] n = normalize(start, w.queryNorm(), epsilon);
      float[] q = new float[a];
      for (int o = 0; o < a; o++) {
        float acc = 0f;
        for (int k = 0; k < h; k++) {
          acc += n[k] * query[o * h + k];
        }
        q[o] = MlpReference.round(acc + w.queryBias()[o]);
      }
      int events = c.rows.get(row).events();
      int keys = fixed + events;
      int[] memoryRow = new int[keys];
      int[] code = new int[keys];
      for (int j = 0; j < keys; j++) {
        if (j < fixed) {
          memoryRow[j] = row * fixed + j;
          code[j] = c.codes[t * fixed + j] & 0xff;
        } else {
          int event = c.firstEvent[row] + j - fixed;
          memoryRow[j] = c.rows.size() * fixed + event;
          int tile = c.eventTiles[event] & 0xff;
          code[j] =
              tau > 0 && tile > 0
                  ? counts.keyCodes() + 1 + (w.tileRelation()[(tau - 1) * tiles + tile - 1] & 0xff)
                  : counts.keyCodes();
        }
      }
      float[] attended = new float[a];
      for (int head = 0; head < a / HipArchitecture.HEAD_WIDTH; head++) {
        int lane = head * HipArchitecture.HEAD_WIDTH;
        double[] logits = new double[keys];
        double maximum = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < keys; j++) {
          double dot = 0;
          for (int d = 0; d < HipArchitecture.HEAD_WIDTH; d++) {
            double key =
                (double) MlpReference.value(c.memory[memoryRow[j] * 2 * a + lane + d]) + w.codeKey()[code[j] * a + lane + d];
            dot += q[lane + d] * key;
          }
          logits[j] = dot * architecture.scale();
          maximum = Math.max(maximum, logits[j]);
        }
        double total = 0;
        for (int j = 0; j < keys; j++) {
          logits[j] = Math.exp(logits[j] - maximum);
          total += logits[j];
        }
        double[] value = new double[HipArchitecture.HEAD_WIDTH];
        for (int j = 0; j < keys; j++) {
          for (int d = 0; d < HipArchitecture.HEAD_WIDTH; d++) {
            double v =
                (double) MlpReference.value(c.memory[memoryRow[j] * 2 * a + a + lane + d])
                    + w.codeValue()[code[j] * a + lane + d];
            value[d] += logits[j] / total * v;
          }
        }
        for (int d = 0; d < HipArchitecture.HEAD_WIDTH; d++) {
          attended[lane + d] = MlpReference.round((float) value[d]);
        }
      }
      float[] next = new float[h];
      for (int i = 0; i < h; i++) {
        float acc = 0f;
        for (int k = 0; k < a; k++) {
          acc += attended[k] * output[i * a + k];
        }
        short bits = MlpReference.bf16(start[i] + MlpReference.round(acc + w.outputBias()[i]));
        z[t * h + i] = bits;
        next[i] = MlpReference.value(bits);
      }
      float[] nb = normalize(next, w.discardNorm(), epsilon);
      float sum = 0f;
      for (int unit = 0; unit < g; unit++) {
        float acc = 0f;
        for (int k = 0; k < h; k++) {
          acc += nb[k] * hidden[unit * h + k];
        }
        float u = MlpReference.round(acc + w.discardHiddenBias()[unit]);
        sum += u / (1f + (float) Math.exp(-u)) * w.discardOutput()[unit];
      }
      float direct = 0f;
      for (int k = 0; k < exactWidth; k++) {
        direct += c.exact[t * exactWidth + k] * w.discardExact()[k];
      }
      b[t] = sum + direct;
    }
    return new Result(z, b);
  }

  /** ⌊(v·r)·g⌉ を FP32 で計算する（r = 1/√(mean(v²) + ε)）。 */
  private static float[] normalize(float[] source, float[] gain, float epsilon) {
    float sum = 0f;
    for (float v : source) {
      sum += v * v;
    }
    float r = (float) (1.0 / Math.sqrt(sum / source.length + epsilon));
    float[] out = new float[source.length];
    for (int k = 0; k < source.length; k++) {
      out[k] = MlpReference.round((source[k] * r) * gain[k]);
    }
    return out;
  }

  private static float[] values(short[] bf16) {
    float[] values = new float[bf16.length];
    for (int i = 0; i < bf16.length; i++) {
      values[i] = MlpReference.value(bf16[i]);
    }
    return values;
  }
}
