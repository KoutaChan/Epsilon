package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipComposition;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;

/**
 * {@link HipReference} と同じ式の FP64 の順伝播（丸めなし）。重みは書いた WeightPack の値（BF16・FP32 の値をそのまま）、入力は K1 が読む
 * 値（EXACT の特徴は K1 と同じく BF16 に丸めた値）なので、丸めの位置だけが違う。層 0 は畳む前の式（L0-FOLD の精度の物差し）。重み・頭・
 * 関係の符号は {@link HipReference} の道具を使う。
 */
final class ExactReference {

  private final HipReference r;
  private final int hidden;
  private final int attention;

  private double[] residual;
  private double[] attentionOut;
  private double[] state;
  private double[] memory;
  private double[] statics;
  private double[] transitions;
  private double[] values;
  private double[] scores;

  ExactReference(HipReference reference) {
    r = reference;
    hidden = reference.hidden;
    attention = reference.attention;
  }

  private double w(HipWeight weight, int layer, int index) {
    return r.weight(weight, layer, index);
  }

  /**
   * 1 塊の順伝播。
   *
   * @param batch 封じた塊
   * @return s（塊の手の順）
   */
  double[] forward(HostBatch batch) {
    HipReference.Input in = new HipReference.Input(batch);
    int tokens = r.fixed * in.rows + in.events;
    residual = new double[tokens * hidden];
    attentionOut = new double[tokens * attention];
    state = new double[r.fixed * in.rows * hidden];
    memory = new double[r.fixed * in.rows * 2 * attention];
    statics = new double[in.transitions * hidden];
    transitions = new double[in.transitions * hidden];
    values = new double[in.transitions];
    scores = new double[in.actions];
    embed(in);
    for (int layer = 0; layer + 1 < r.depth; layer++) {
      attention(in, layer, false);
      for (int t = 0; t < tokens; t++) {
        double[] x = mlpStage(slice(residual, t * hidden, hidden), slice(attentionOut, t * attention, attention), layer, false);
        System.arraycopy(x, 0, residual, t * hidden, hidden);
      }
    }
    attention(in, r.depth - 1, true);
    mlpKv(in, r.depth - 1);
    candidate(in);
    valueGraph(in);
    return scores.clone();
  }

  private void embed(HipReference.Input in) {
    PackView pack = r.pack;
    for (HipEmbedding.Slot slot : HipEmbedding.Slot.values()) {
      PackView.Spec spec = pack.embedding(slot);
      int count = r.slotCount(slot, in);
      int start = r.slotStart(slot, in);
      double[] out = slot == HipEmbedding.Slot.TRANSITION ? statics : residual;
      Tensor category = HipReference.tensor(spec.category());
      Tensor scalar = spec.scalar() < 0 ? null : HipReference.tensor(spec.scalar());
      Tensor exact = spec.exact() < 0 ? null : HipReference.tensor(spec.exact());
      for (int t = 0; t < count; t++) {
        double[] acc = new double[hidden];
        for (int table = 0; table < spec.tableOffset().length; table++) {
          int entry = 0;
          for (int m = 0; m < spec.columns()[table].length; m++) {
            entry += in.u8(category, t, spec.columns()[table][m]) * spec.strides()[table][m];
          }
          int base = spec.tableOffset()[table] + entry * hidden * Float.BYTES;
          for (int h = 0; h < hidden; h++) {
            acc[h] += pack.bytes.getFloat(base + h * Float.BYTES);
          }
        }
        for (int k = 0; k < spec.source().length; k++) {
          int source = spec.source()[k] >> 16;
          int index = spec.source()[k] & 0xFFFF;
          double feature =
              switch (HipEmbedding.Source.values()[source]) {
                case FLAG -> in.u8(category, t, index);
                case SCALAR -> in.bf16(scalar, t, index);
                case EXACT -> HipReference.bf(in.f32(exact, t, index));
              };
          for (int h = 0; h < hidden; h++) {
            double weight =
                Float.intBitsToFloat(
                    (pack.bytes.getShort(spec.denseOffset() + (k * hidden + h) * Short.BYTES) & 0xFFFF) << 16);
            acc[h] += weight * feature;
          }
        }
        for (int h = 0; h < hidden; h++) {
          out[(start + t) * hidden + h] = acc[h] + pack.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
        }
      }
    }
  }

  private void attention(HipReference.Input in, int layer, boolean last) {
    PackView pack = r.pack;
    int relationWidth = Tensor.RELATION.width();
    int columnTile = pack.schema(HipSchema.RELATION_TILE);
    int columnOwner = pack.schema(HipSchema.RELATION_OWNER);
    int columnEvent = pack.schema(HipSchema.RELATION_EVENT);
    int columnRiichi = pack.schema(HipSchema.RELATION_RIICHI);
    int columnDiscard = pack.schema(HipSchema.RELATION_LAST_DISCARD);
    int columnLink = pack.schema(HipSchema.RELATION_LINK);
    int columnSafety = pack.schema(HipSchema.RELATION_SAFETY);
    int safeties = pack.schema(HipSchema.RELATION_SAFETIES);
    int heads = r.heads;
    for (int row = 0; row < in.rows; row++) {
      int firstEvent = in.row(row, RowColumn.FIRST_EVENT);
      int tokens = r.fixed + in.row(row, RowColumn.EVENTS);
      int queries = last ? r.fixed : tokens;
      int[] global = new int[tokens];
      int[][] attributes = new int[tokens][relationWidth];
      for (int p = 0; p < tokens; p++) {
        global[p] = r.tokenIndex(in.rows, row, firstEvent, p);
        for (int c = 0; c < relationWidth; c++) {
          attributes[p][c] = in.u8(Tensor.RELATION, global[p], c);
        }
      }
      double[][] qkv = new double[tokens][3 * attention];
      for (int p = 0; p < tokens; p++) {
        double[] normalized = normalize(residual, global[p] * hidden, HipWeight.ATTENTION_NORM, layer);
        for (int col = 0; col < 3 * attention; col++) {
          if (col < attention && p >= queries) {
            continue;
          }
          double acc = 0;
          for (int k = 0; k < hidden; k++) {
            acc += normalized[k] * w(HipWeight.QKV, layer, col * hidden + k);
          }
          qkv[p][col] = acc + w(HipWeight.QKV_BIAS, layer, col);
        }
      }
      int now = attributes[0][columnEvent];
      for (int i = 0; i < queries; i++) {
        int typeQ = r.type(i);
        int[][] pairs = new int[tokens][];
        for (int j = 0; j < tokens; j++) {
          pairs[j] =
              r.segments(typeQ, r.type(j), attributes[i], attributes[j], now, columnTile, columnOwner, columnEvent,
                  columnRiichi, columnDiscard, columnLink, columnSafety, safeties);
        }
        for (int h = 0; h < heads; h++) {
          double[] logits = new double[tokens];
          double maximum = Double.NEGATIVE_INFINITY;
          for (int j = 0; j < tokens; j++) {
            double dot = 0;
            for (int d = 0; d < 16; d++) {
              dot += qkv[i][16 * h + d] * qkv[j][attention + 16 * h + d];
            }
            double score = dot * r.scale;
            for (int s = 0; s < 7; s++) {
              score += w(HipWeight.RELATION_BIAS, layer, (16 * s + pairs[j][s]) * heads + h);
            }
            logits[j] = score;
            maximum = Math.max(maximum, score);
          }
          double total = 0;
          for (int j = 0; j < tokens; j++) {
            logits[j] = Math.exp(logits[j] - maximum);
            total += logits[j];
          }
          for (int d = 0; d < 16; d++) {
            double value = 0;
            for (int j = 0; j < tokens; j++) {
              value += logits[j] / total * qkv[j][2 * attention + 16 * h + d];
            }
            attentionOut[global[i] * attention + 16 * h + d] = value;
          }
        }
      }
    }
  }

  private void mlpKv(HipReference.Input in, int layer) {
    int tokens = r.fixed * in.rows;
    for (int t = 0; t < tokens; t++) {
      double[] x = mlpStage(slice(residual, t * hidden, hidden), slice(attentionOut, t * attention, attention), layer, false);
      double[] y = normalize(x, 0, HipWeight.FINAL_NORM, 0);
      System.arraycopy(y, 0, state, t * hidden, hidden);
      int[] located = r.locate(in.rows, t);
      int base = (located[0] * r.fixed + located[1]) * 2 * attention;
      for (int c = 0; c < 2 * attention; c++) {
        double acc = 0;
        for (int k = 0; k < hidden; k++) {
          acc += y[k] * w(HipWeight.KEY_VALUE, 0, c * hidden + k);
        }
        memory[base + c] = acc + w(HipWeight.KEY_VALUE_BIAS, 0, c);
      }
    }
  }

  private double[] mlpStage(double[] x, double[] o, int layer, boolean candidate) {
    HipWeight outWeight = candidate ? HipWeight.CANDIDATE_OUTPUT : HipWeight.OUTPUT;
    HipWeight outBias = candidate ? HipWeight.CANDIDATE_OUTPUT_BIAS : HipWeight.OUTPUT_BIAS;
    HipWeight normGain = candidate ? HipWeight.CANDIDATE_FFN_NORM : HipWeight.FFN_NORM;
    HipWeight up = candidate ? HipWeight.CANDIDATE_UP : HipWeight.UP;
    HipWeight upBias = candidate ? HipWeight.CANDIDATE_UP_BIAS : HipWeight.UP_BIAS;
    HipWeight down = candidate ? HipWeight.CANDIDATE_DOWN : HipWeight.DOWN;
    HipWeight downBias = candidate ? HipWeight.CANDIDATE_DOWN_BIAS : HipWeight.DOWN_BIAS;
    int ffn = r.ffn;
    double[] x1 = new double[hidden];
    for (int i = 0; i < hidden; i++) {
      double acc = 0;
      for (int k = 0; k < attention; k++) {
        acc += o[k] * w(outWeight, layer, i * attention + k);
      }
      x1[i] = x[i] + (acc + w(outBias, layer, i));
    }
    double[] n = normalize(x1, 0, normGain, layer);
    double[] h = new double[ffn];
    for (int j = 0; j < ffn; j++) {
      double acc = 0;
      for (int k = 0; k < hidden; k++) {
        acc += n[k] * w(up, layer, j * hidden + k);
      }
      h[j] = silu(acc + w(upBias, layer, j));
    }
    double[] out = new double[hidden];
    for (int i = 0; i < hidden; i++) {
      double acc = 0;
      for (int j = 0; j < ffn; j++) {
        acc += h[j] * w(down, layer, i * ffn + j);
      }
      out[i] = x1[i] + (acc + w(downBias, layer, i));
    }
    return out;
  }

  private void candidate(HipReference.Input in) {
    int keys = r.fixed;
    int heads = r.heads;
    int categoryCallDiscard = r.pack.schema(HipSchema.TRANSITION_CALL_DISCARD);
    int categoryMainTile = r.pack.schema(HipSchema.TRANSITION_MAIN_TILE);
    int[] rowOf = HipReference.transitionRows(in);
    for (int t = 0; t < in.transitions; t++) {
      int row = rowOf[t];
      int callDiscard = in.u8(Tensor.TRANSITION_CATEGORY, t, categoryCallDiscard);
      int mainTile = in.u8(Tensor.TRANSITION_CATEGORY, t, categoryMainTile);
      int tileId = callDiscard > 0 ? callDiscard : mainTile;
      double[] start = new double[hidden];
      for (int h = 0; h < hidden; h++) {
        double acc = 0;
        if (tileId > 0) {
          int token = (1 + r.seats) * in.rows + r.tiles * row + tileId - 1;
          for (int k = 0; k < hidden; k++) {
            acc += state[token * hidden + k] * w(HipWeight.TILE_ENTRY, 0, h * hidden + k);
          }
        }
        start[h] = statics[t * hidden + h] + acc;
      }
      double[] normalized = normalize(start, 0, HipWeight.QUERY_NORM, 0);
      double[] query = new double[attention];
      for (int c = 0; c < attention; c++) {
        double acc = 0;
        for (int k = 0; k < hidden; k++) {
          acc += normalized[k] * w(HipWeight.QUERY, 0, c * hidden + k);
        }
        query[c] = acc + w(HipWeight.QUERY_BIAS, 0, c);
      }
      double[] attended = new double[attention];
      for (int h = 0; h < heads; h++) {
        double[] logits = new double[keys];
        double maximum = Double.NEGATIVE_INFINITY;
        for (int j = 0; j < keys; j++) {
          int code = in.u8(Tensor.TRANSITION_CODE, t, j);
          int base = (row * keys + j) * 2 * attention + 16 * h;
          double dot = 0;
          for (int d = 0; d < 16; d++) {
            dot += query[16 * h + d] * (memory[base + d] + w(HipWeight.CODE_KEY, 0, code * attention + 16 * h + d));
          }
          logits[j] = dot * r.scale;
          maximum = Math.max(maximum, logits[j]);
        }
        double total = 0;
        for (int j = 0; j < keys; j++) {
          logits[j] = Math.exp(logits[j] - maximum);
          total += logits[j];
        }
        for (int j = 0; j < keys; j++) {
          double p = logits[j] / total;
          int code = in.u8(Tensor.TRANSITION_CODE, t, j);
          int base = (row * keys + j) * 2 * attention + attention + 16 * h;
          for (int d = 0; d < 16; d++) {
            attended[16 * h + d] +=
                p * (memory[base + d] + w(HipWeight.CODE_VALUE, 0, code * attention + 16 * h + d));
          }
        }
      }
      double[] z = mlpStage(start, attended, 0, true);
      System.arraycopy(z, 0, transitions, t * hidden, hidden);
      double[] n = normalize(z, 0, HipWeight.DISCARD_NORM, 0);
      double sum = 0;
      for (int g = 0; g < r.gate; g++) {
        double acc = 0;
        for (int k = 0; k < hidden; k++) {
          acc += n[k] * w(HipWeight.DISCARD_HIDDEN, 0, g * hidden + k);
        }
        sum += silu(acc + w(HipWeight.DISCARD_HIDDEN_BIAS, 0, g)) * w(HipWeight.DISCARD_OUTPUT, 0, g);
      }
      double exact = 0;
      for (int k = 0; k < Tensor.DISCARD_EXACT.width(); k++) {
        exact += in.f32(Tensor.DISCARD_EXACT, t, k) * w(HipWeight.DISCARD_EXACT, 0, k);
      }
      values[t] = sum + exact;
    }
  }

  private void valueGraph(HipReference.Input in) {
    PackView pack = r.pack;
    int exactWidth = Tensor.ACTION_EXACT.width();
    double[] delta = new double[in.actions];
    for (int a = 0; a < in.actions; a++) {
      int role = in.action(a, ActionColumn.ROLE);
      int kind = pack.gate(role);
      if (kind < 0) {
        continue;
      }
      HipComposition composition = pack.composition(role);
      int representative = in.action(a, ActionColumn.REPRESENTATIVE);
      double[] chosen =
          aggregate(in.action(representative, ActionColumn.FIRST_TRANSITION),
              in.action(representative, ActionColumn.TRANSITIONS));
      int row = in.action(a, ActionColumn.ROW);
      double[] base;
      if (composition == HipComposition.TURN_GATE) {
        base = bar(in, row);
      } else {
        int baseAction = in.action(a, ActionColumn.BASE);
        int baseRepresentative = in.action(baseAction, ActionColumn.REPRESENTATIVE);
        base = slice(transitions, in.action(baseRepresentative, ActionColumn.FIRST_TRANSITION) * hidden, hidden);
      }
      double[] input = new double[3 * hidden];
      System.arraycopy(normalize(chosen, 0, HipWeight.GATE_NORM, 0), 0, input, 0, hidden);
      System.arraycopy(normalize(base, 0, HipWeight.GATE_NORM, 0), 0, input, hidden, hidden);
      System.arraycopy(state, row * hidden, input, 2 * hidden, hidden);
      double sum = 0;
      for (int g = 0; g < r.gate; g++) {
        double acc = 0;
        for (int k = 0; k < 3 * hidden; k++) {
          acc += input[k] * w(HipWeight.GATE_HIDDEN, 0, g * 3 * hidden + k);
        }
        double projected = 0;
        for (int x = 0; x < exactWidth; x++) {
          projected +=
              w(HipWeight.GATE_EXACT_PROJECTION, 0, (kind * r.gate + g) * exactWidth + x) * in.f32(Tensor.ACTION_EXACT, a, x);
        }
        double u =
            acc + w(HipWeight.GATE_HIDDEN_BIAS, 0, g) + w(HipWeight.GATE_KIND_OFFSET, 0, kind * r.gate + g) + projected;
        sum += silu(u) * w(HipWeight.GATE_OUTPUT, 0, kind * r.gate + g);
      }
      double direct = 0;
      for (int x = 0; x < exactWidth; x++) {
        direct += in.f32(Tensor.ACTION_EXACT, a, x) * w(HipWeight.GATE_EXACT_OUTPUT, 0, kind * exactWidth + x);
      }
      delta[a] = sum + w(HipWeight.GATE_OUTPUT_BIAS, 0, kind) + direct;
    }
    for (int row = 0; row < in.rows; row++) {
      int first = in.row(row, RowColumn.FIRST_ACTION);
      int count = in.row(row, RowColumn.ACTIONS);
      double pivot = 0;
      boolean havePivot = false;
      for (int a = first; a < first + count; a++) {
        HipComposition composition = pack.composition(in.action(a, ActionColumn.ROLE));
        double score;
        switch (composition) {
          case DISCARD_VALUE -> {
            int representative = in.action(a, ActionColumn.REPRESENTATIVE);
            score = values[in.action(representative, ActionColumn.FIRST_TRANSITION)];
          }
          case RIICHI_GATE -> {
            int representative = in.action(in.action(a, ActionColumn.BASE), ActionColumn.REPRESENTATIVE);
            score = values[in.action(representative, ActionColumn.FIRST_TRANSITION)] + delta[a];
          }
          case RESPONSE_GATE -> score = delta[a];
          case TURN_GATE -> {
            continue;
          }
          default -> score = 0;
        }
        scores[a] = score;
        if (composition == HipComposition.DISCARD_VALUE || composition == HipComposition.RIICHI_GATE) {
          pivot = havePivot ? Math.max(pivot, score) : score;
          havePivot = true;
        }
      }
      for (int a = first; a < first + count; a++) {
        if (pack.composition(in.action(a, ActionColumn.ROLE)) == HipComposition.TURN_GATE) {
          scores[a] = (havePivot ? pivot : 0) + delta[a];
        }
      }
    }
  }

  /** softmax(B/τ) で z を集める。 */
  private double[] aggregate(int first, int count) {
    if (count == 1) {
      return slice(transitions, first * hidden, hidden);
    }
    double maximum = Double.NEGATIVE_INFINITY;
    for (int i = 0; i < count; i++) {
      maximum = Math.max(maximum, values[first + i] / r.tau);
    }
    double total = 0;
    for (int i = 0; i < count; i++) {
      total += Math.exp(values[first + i] / r.tau - maximum);
    }
    double[] e = new double[hidden];
    for (int i = 0; i < count; i++) {
      double weight = Math.exp(values[first + i] / r.tau - maximum) / total;
      for (int h = 0; h < hidden; h++) {
        e[h] += weight * transitions[(first + i) * hidden + h];
      }
    }
    return e;
  }

  /** ē: 行の代表の打牌の softmax(B/τ) で z を集める。 */
  private double[] bar(HipReference.Input in, int row) {
    int first = in.row(row, RowColumn.FIRST_ACTION);
    int count = in.row(row, RowColumn.ACTIONS);
    double maximum = Double.NEGATIVE_INFINITY;
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        maximum = Math.max(maximum, values[in.action(a, ActionColumn.FIRST_TRANSITION)] / r.tau);
      }
    }
    double total = 0;
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        total += Math.exp(values[in.action(a, ActionColumn.FIRST_TRANSITION)] / r.tau - maximum);
      }
    }
    double[] e = new double[hidden];
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        int t = in.action(a, ActionColumn.FIRST_TRANSITION);
        double weight = Math.exp(values[t] / r.tau - maximum) / total;
        for (int h = 0; h < hidden; h++) {
          e[h] += weight * transitions[t * hidden + h];
        }
      }
    }
    return e;
  }

  private boolean representativeDiscard(HipReference.Input in, int action) {
    return r.pack.composition(in.action(action, ActionColumn.ROLE)) == HipComposition.DISCARD_VALUE
        && in.action(action, ActionColumn.REPRESENTATIVE) == action;
  }

  /** (v·r)·g（r = 1/√(mean(v²) + ε)）。 */
  private double[] normalize(double[] source, int at, HipWeight gain, int layer) {
    double sum = 0;
    for (int k = 0; k < hidden; k++) {
      sum += source[at + k] * source[at + k];
    }
    double inverse = 1.0 / Math.sqrt(sum / hidden + r.epsilon);
    double[] out = new double[hidden];
    for (int k = 0; k < hidden; k++) {
      out[k] = source[at + k] * inverse * w(gain, layer, k);
    }
    return out;
  }

  private static double[] slice(double[] values, int at, int length) {
    double[] out = new double[length];
    System.arraycopy(values, at, out, 0, length);
    return out;
  }

  private static double silu(double value) {
    return value / (1 + Math.exp(-value));
  }
}
