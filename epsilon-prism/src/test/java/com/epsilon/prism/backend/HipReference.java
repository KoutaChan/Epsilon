package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipComposition;
import com.epsilon.prism.hip.HipDimensions;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionColumn;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.RowColumn;
import com.epsilon.prism.input.InputSchema.Tensor;
import com.epsilon.prism.model.PrismShape;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

/**
 * HIP の K1〜K7 の CPU の参照。カーネルと同じ式・同じ丸めの所を、書いた WeightPack（頭を native の並びで読み直す）と HostBatch の blob
 * から素朴な FP32 の積和で計算する。和の順は行列積（WMMA）と 1/rms でカーネルと違い、ほかの所（K1・注意の softmax と p·v・候補の
 * 交差注意・集約・合成）はカーネルと同じ順にする。作業域の値は HIP の作業域と同じ並びの float（BF16 の区画は BF16 の値）。行ごとに閉じた計算
 * なので、同じ行の s は塊の組み方に依らずビットで同じ。方策の試験は GPU の計画の代わりに使う（{@link #forward} だけを公開）。
 */
public final class HipReference {

  private final PackView pack;
  private final int hidden;
  private final int attention;
  private final int heads;
  private final int ffn;
  private final int gate;
  private final int depth;
  private final float epsilon;
  private final int seats;
  private final int tiles;
  private final int fixed;
  private final float scale;
  private final float tau;

  /** 置き場ごと・層ごとの値（BF16・UINT8 は float にした値）。 */
  private final float[][][] cache = new float[HipWeight.values().length][][];

  float[] residual;
  float[] attentionOut;
  float[] state;
  float[] memory;
  float[] statics;
  float[] transitions;
  float[] values;
  float[] scores;

  /** K6 の交差注意を FP64 で計算し、o を BF16 に 1 回だけ丸める（カーネルの FP32 の順の精度を測る物差し。既定はカーネルと同じ順の FP32）。 */
  boolean exactCrossAttention;

  /**
   * @param weights 重みの束（頭を native の並びで読み直す）
   */
  public HipReference(WeightPack weights) {
    pack = new PackView(weights.buffer());
    PrismShape shape = weights.layout().shape();
    hidden = shape.hidden();
    attention = shape.attention();
    heads = shape.heads();
    ffn = shape.ffn();
    gate = shape.gate();
    depth = shape.depth();
    epsilon = (float) shape.epsilon();
    seats = pack.schema(HipSchema.SEATS);
    tiles = pack.schema(HipSchema.TILES);
    fixed = 1 + seats + tiles;
    scale = Float.intBitsToFloat(pack.schema(HipSchema.SCALE_BITS));
    tau = Float.intBitsToFloat(pack.schema(HipSchema.TAU_BITS));
    HipDimensions dimensions = weights.layout().dimensions();
    for (HipWeight weight : HipWeight.values()) {
      int layers = weight.layered() ? depth : 1;
      cache[weight.ordinal()] = new float[layers][];
      for (int layer = 0; layer < layers; layer++) {
        int elements = Math.toIntExact(weight.elements(dimensions));
        float[] values = new float[elements];
        for (int i = 0; i < elements; i++) {
          values[i] =
              switch (weight.element()) {
                case U8 -> pack.u8(weight, layer, i);
                case BF16 -> pack.bf16(weight, layer, i);
                case FP32 -> pack.f32(weight, layer, i);
                case I32 -> throw new IllegalStateException("no int32 weights");
              };
        }
        cache[weight.ordinal()][layer] = values;
      }
    }
  }

  /** 置き場の値（層を持たない置き場は層 0）。 */
  private float weight(HipWeight weight, int layer, int index) {
    return cache[weight.ordinal()][weight.layered() ? layer : 0][index];
  }

  /** 塊の入力の読み手（blob の区画）。 */
  static final class Input {
    final ByteBuffer blob;
    final BatchPlan plan;
    final int rows;
    final int events;
    final int transitions;
    final int actions;

    Input(HostBatch batch) {
      blob = batch.blob().duplicate().order(ByteOrder.LITTLE_ENDIAN);
      plan = batch.plan();
      rows = plan.rows();
      events = plan.count(Axis.EVENT);
      transitions = plan.count(Axis.TRANSITION);
      actions = plan.count(Axis.ACTION);
    }

    int u8(Tensor tensor, int row, int column) {
      return blob.get(plan.offset(tensor) + row * tensor.width() + column) & 0xFF;
    }

    float bf16(Tensor tensor, int row, int column) {
      return Float.intBitsToFloat(
          (blob.getShort(plan.offset(tensor) + (row * tensor.width() + column) * Short.BYTES) & 0xFFFF) << 16);
    }

    float f32(Tensor tensor, int row, int column) {
      return blob.getFloat(plan.offset(tensor) + (row * tensor.width() + column) * Float.BYTES);
    }

    int i32(Tensor tensor, int row, int column) {
      return blob.getInt(plan.offset(tensor) + (row * tensor.width() + column) * Integer.BYTES);
    }

    int row(int row, RowColumn column) {
      return i32(Tensor.ROW_COLUMN, row, column.ordinal());
    }

    int action(int action, ActionColumn column) {
      return i32(Tensor.ACTION_COLUMN, action, column.ordinal());
    }
  }

  /**
   * 1 塊の順伝播（K1 → 層 → K6 → K7）。
   *
   * @param batch 封じた塊
   * @return s（塊の手の順）
   */
  public float[] forward(HostBatch batch) {
    Input in = new Input(batch);
    allocate(in);
    embed(in);
    for (int layer = 0; layer + 1 < depth; layer++) {
      attention(in, layer, false);
      mlp(in, layer);
    }
    attention(in, depth - 1, true);
    mlpKv(in, depth - 1);
    candidate(in);
    valueGraph(in);
    return scores.clone();
  }

  /** 作業域を塊の大きさで取る。 */
  void allocate(Input in) {
    int tokens = fixed * in.rows + in.events;
    residual = new float[tokens * hidden];
    attentionOut = new float[tokens * attention];
    state = new float[fixed * in.rows * hidden];
    memory = new float[fixed * in.rows * 2 * attention];
    statics = new float[in.transitions * hidden];
    transitions = new float[in.transitions * hidden];
    values = new float[in.transitions];
    scores = new float[in.actions];
  }

  void embed(Input in) {
    for (HipEmbedding.Slot slot : HipEmbedding.Slot.values()) {
      PackView.Spec spec = pack.embedding(slot);
      int count =
          switch (slot) {
            case ROUND -> in.rows;
            case SEAT -> seats * in.rows;
            case TILE -> tiles * in.rows;
            case EVENT -> in.events;
            case TRANSITION -> in.transitions;
          };
      int start =
          switch (slot) {
            case ROUND, TRANSITION -> 0;
            case SEAT -> in.rows;
            case TILE -> (1 + seats) * in.rows;
            case EVENT -> fixed * in.rows;
          };
      float[] out = slot == HipEmbedding.Slot.TRANSITION ? statics : residual;
      Tensor category = tensor(spec.category());
      Tensor scalar = spec.scalar() < 0 ? null : tensor(spec.scalar());
      Tensor exact = spec.exact() < 0 ? null : tensor(spec.exact());
      for (int t = 0; t < count; t++) {
        float[] acc = new float[hidden];
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
          float feature =
              switch (HipEmbedding.Source.values()[source]) {
                case FLAG -> in.u8(category, t, index);
                case SCALAR -> in.bf16(scalar, t, index);
                case EXACT -> bf(in.f32(exact, t, index));
              };
          for (int h = 0; h < hidden; h++) {
            float weight =
                Float.intBitsToFloat(
                    (pack.bytes.getShort(spec.denseOffset() + (k * hidden + h) * Short.BYTES) & 0xFFFF) << 16);
            acc[h] = Math.fma(weight, feature, acc[h]);
          }
        }
        for (int h = 0; h < hidden; h++) {
          acc[h] += pack.bytes.getFloat(spec.biasOffset() + h * Float.BYTES);
          out[(start + t) * hidden + h] = bf(acc[h]);
        }
      }
    }
  }

  /** 背骨の層 layer の注意の入力 n = ⌊(x·r)·g⌉ を x の全てのトークンに（NORMALIZED の区画と同じ並び）。 */
  float[] attentionInput(float[] x, int layer) {
    float[] out = new float[x.length];
    for (int t = 0; t < x.length / hidden; t++) {
      System.arraycopy(normalize(x, t * hidden, HipWeight.ATTENTION_NORM, layer), 0, out, t * hidden, hidden);
    }
    return out;
  }

  void attention(Input in, int layer, boolean last) {
    int relationWidth = Tensor.RELATION.width();
    int columnTile = pack.schema(HipSchema.RELATION_TILE);
    int columnOwner = pack.schema(HipSchema.RELATION_OWNER);
    int columnEvent = pack.schema(HipSchema.RELATION_EVENT);
    int columnRiichi = pack.schema(HipSchema.RELATION_RIICHI);
    int columnDiscard = pack.schema(HipSchema.RELATION_LAST_DISCARD);
    int columnLink = pack.schema(HipSchema.RELATION_LINK);
    int columnSafety = pack.schema(HipSchema.RELATION_SAFETY);
    int safeties = pack.schema(HipSchema.RELATION_SAFETIES);
    for (int r = 0; r < in.rows; r++) {
      int firstEvent = in.row(r, RowColumn.FIRST_EVENT);
      int tokens = fixed + in.row(r, RowColumn.EVENTS);
      int queries = last ? fixed : tokens;
      int[] global = new int[tokens];
      int[][] attributes = new int[tokens][relationWidth];
      for (int p = 0; p < tokens; p++) {
        global[p] = tokenIndex(in.rows, r, firstEvent, p);
        for (int c = 0; c < relationWidth; c++) {
          attributes[p][c] = in.u8(Tensor.RELATION, global[p], c);
        }
      }
      float[][] qkv = new float[tokens][3 * attention];
      for (int p = 0; p < tokens; p++) {
        float[] normalized = normalize(residual, global[p] * hidden, HipWeight.ATTENTION_NORM, layer);
        for (int col = 0; col < 3 * attention; col++) {
          if (col < attention && p >= queries) {
            continue;
          }
          float acc = 0f;
          for (int k = 0; k < hidden; k++) {
            acc += normalized[k] * weight(HipWeight.QKV, layer, col * hidden + k);
          }
          qkv[p][col] = bf(acc + weight(HipWeight.QKV_BIAS, layer, col));
        }
      }
      int now = attributes[0][columnEvent];
      for (int i = 0; i < queries; i++) {
        int typeQ = type(i);
        int[][] pairs = new int[tokens][];
        for (int j = 0; j < tokens; j++) {
          pairs[j] =
              segments(typeQ, type(j), attributes[i], attributes[j], now, columnTile, columnOwner, columnEvent,
                  columnRiichi, columnDiscard, columnLink, columnSafety, safeties);
        }
        for (int h = 0; h < heads; h++) {
          float[] logits = new float[tokens];
          float maximum = Float.NEGATIVE_INFINITY;
          for (int j = 0; j < tokens; j++) {
            int[] segments = pairs[j];
            float dot = 0f;
            for (int d = 0; d < 16; d++) {
              dot = Math.fma(qkv[i][16 * h + d], qkv[j][attention + 16 * h + d], dot);
            }
            float score = dot * scale;
            for (int s = 0; s < 7; s++) {
              score += weight(HipWeight.RELATION_BIAS, layer, (16 * s + segments[s]) * heads + h);
            }
            logits[j] = score;
            maximum = Math.max(maximum, score);
          }
          float[] exp = new float[tokens];
          for (int j = 0; j < tokens; j++) {
            exp[j] = expf(logits[j] - maximum);
          }
          float total = tree64(exp);
          for (int d = 0; d < 16; d++) {
            float value = 0f;
            for (int j = 0; j < tokens; j++) {
              value = Math.fma(exp[j] / total, qkv[j][2 * attention + 16 * h + d], value);
            }
            attentionOut[global[i] * attention + 16 * h + d] = bf(value);
          }
        }
      }
    }
  }

  /** eager の packed 注意の GPU のカーネルの和: 64 個の部分和（鍵 v, v+64, …、順に）の 2 分木。 */
  static float tree64(float[] values) {
    float[] partial = new float[64];
    for (int v = 0; v < 64; v++) {
      float sum = 0f;
      for (int key = v; key < values.length; key += 64) {
        sum += values[key];
      }
      partial[v] = sum;
    }
    for (int stride = 32; stride > 0; stride >>= 1) {
      for (int v = 0; v < stride; v++) {
        partial[v] += partial[v + stride];
      }
    }
    return partial[0];
  }

  /** RelationClass と同じ 7 区画の符号。 */
  private int[] segments(
      int typeQ,
      int typeK,
      int[] q,
      int[] k,
      int now,
      int columnTile,
      int columnOwner,
      int columnEvent,
      int columnRiichi,
      int columnDiscard,
      int columnLink,
      int columnSafety,
      int safeties) {
    int[] segment = new int[7];
    segment[0] = 4 * typeQ + typeK;
    int tileQ = q[columnTile];
    int tileK = k[columnTile];
    boolean bothTiles = tileQ > 0 && tileK > 0;
    int relation =
        (int) weight(HipWeight.TILE_RELATION, 0, Math.max(tileQ - 1, 0) * tiles + Math.max(tileK - 1, 0));
    segment[1] = bothTiles && relation < 15 ? relation + 1 : 0;
    segment[2] = bothTiles && relation >= 15 ? relation - 14 : 0;
    int ownerQ = q[columnOwner];
    int ownerK = k[columnOwner];
    if (ownerQ > 0 && ownerK > 0) {
      segment[3] = Math.floorMod(ownerK - ownerQ + 4, 4) + 1;
    } else if (ownerK > 0) {
      segment[3] = ownerK + 4;
    } else if (ownerQ > 0) {
      segment[3] = ownerQ + 8;
    }
    int eventQ = q[columnEvent];
    int eventK = k[columnEvent];
    boolean keyEvent = typeK == 3;
    boolean queryEvent = typeQ == 3;
    int lastDiscardQ = q[columnDiscard];
    boolean afterDiscard = lastDiscardQ > 0 && eventK > lastDiscardQ;
    int riichiK = k[columnRiichi];
    segment[4] =
        (keyEvent ? phase(eventK, riichiK) + 1 + (afterDiscard ? 4 : 0) : 0)
            + (queryEvent && typeK == 1 ? phase(eventQ, riichiK) + 9 : 0);
    boolean bothEvents = queryEvent && keyEvent;
    int delta = eventK - eventQ;
    int magnitude = Math.abs(delta);
    int step = (magnitude >= 1 ? 1 : 0) + (magnitude >= 2 ? 1 : 0) + (magnitude >= 4 ? 1 : 0) + (magnitude >= 8 ? 1 : 0);
    int age = now - eventK;
    int ageStep = 10 + (age >= 5 ? 1 : 0) + (age >= 9 ? 1 : 0) + (age >= 17 ? 1 : 0);
    segment[5] = (bothEvents ? Integer.signum(delta) * step + 5 : 0) + (!queryEvent && keyEvent ? ageStep : 0);
    int linkQ = q[columnLink];
    int linkK = k[columnLink];
    boolean forward = linkQ > 0 && linkQ == eventK;
    boolean backward = linkK > 0 && linkK == eventQ && !forward;
    int link = bothEvents ? (forward ? 1 : 0) + (backward ? 2 : 0) : 0;
    boolean seatToTile = typeQ == 1 && typeK == 2 && ownerQ > 1;
    boolean tileToSeat = typeQ == 2 && typeK == 1 && ownerK > 1;
    int safetyOfKey = 0;
    int safetyOfQuery = 0;
    for (int opponent = 0; opponent < safeties; opponent++) {
      safetyOfKey += ownerQ == 2 + opponent ? k[columnSafety + opponent] : 0;
      safetyOfQuery += ownerK == 2 + opponent ? q[columnSafety + opponent] : 0;
    }
    segment[6] = link + (seatToTile ? safetyOfKey + 3 : 0) + (tileToSeat ? safetyOfQuery + 7 : 0);
    return segment;
  }

  private static int phase(int number, int riichi) {
    return riichi > 0 ? (number >= riichi ? 1 : 0) + (number > riichi ? 1 : 0) + 1 : 0;
  }

  void mlp(Input in, int layer) {
    int tokens = fixed * in.rows + in.events;
    for (int t = 0; t < tokens; t++) {
      float[] x = mlpStage(slice(residual, t * hidden, hidden), slice(attentionOut, t * attention, attention), layer, false);
      System.arraycopy(x, 0, residual, t * hidden, hidden);
    }
  }

  void mlpKv(Input in, int layer) {
    int tokens = fixed * in.rows;
    for (int t = 0; t < tokens; t++) {
      float[] x = mlpStage(slice(residual, t * hidden, hidden), slice(attentionOut, t * attention, attention), layer, false);
      float[] y = normalize(x, 0, HipWeight.FINAL_NORM, 0);
      System.arraycopy(y, 0, state, t * hidden, hidden);
      int[] located = locate(in.rows, t);
      int base = (located[0] * fixed + located[1]) * 2 * attention;
      for (int c = 0; c < 2 * attention; c++) {
        float acc = 0f;
        for (int k = 0; k < hidden; k++) {
          acc += y[k] * weight(HipWeight.KEY_VALUE, 0, c * hidden + k);
        }
        memory[base + c] = bf(acc + weight(HipWeight.KEY_VALUE_BIAS, 0, c));
      }
    }
  }

  /** K3 の式（candidate なら候補の重み）。 */
  private float[] mlpStage(float[] x, float[] o, int layer, boolean candidate) {
    HipWeight outWeight = candidate ? HipWeight.CANDIDATE_OUTPUT : HipWeight.OUTPUT;
    HipWeight outBias = candidate ? HipWeight.CANDIDATE_OUTPUT_BIAS : HipWeight.OUTPUT_BIAS;
    HipWeight normGain = candidate ? HipWeight.CANDIDATE_FFN_NORM : HipWeight.FFN_NORM;
    HipWeight up = candidate ? HipWeight.CANDIDATE_UP : HipWeight.UP;
    HipWeight upBias = candidate ? HipWeight.CANDIDATE_UP_BIAS : HipWeight.UP_BIAS;
    HipWeight down = candidate ? HipWeight.CANDIDATE_DOWN : HipWeight.DOWN;
    HipWeight downBias = candidate ? HipWeight.CANDIDATE_DOWN_BIAS : HipWeight.DOWN_BIAS;
    float[] x1 = new float[hidden];
    for (int i = 0; i < hidden; i++) {
      float acc = 0f;
      for (int k = 0; k < attention; k++) {
        acc += o[k] * weight(outWeight, layer, i * attention + k);
      }
      x1[i] = bf(x[i] + bf(acc + weight(outBias, layer, i)));
    }
    float[] n = normalize(x1, 0, normGain, layer);
    float[] h = new float[ffn];
    for (int j = 0; j < ffn; j++) {
      float acc = 0f;
      for (int k = 0; k < hidden; k++) {
        acc += n[k] * weight(up, layer, j * hidden + k);
      }
      h[j] = bf(silu(bf(acc + weight(upBias, layer, j))));
    }
    float[] out = new float[hidden];
    for (int i = 0; i < hidden; i++) {
      float acc = 0f;
      for (int j = 0; j < ffn; j++) {
        acc += h[j] * weight(down, layer, i * ffn + j);
      }
      out[i] = bf(x1[i] + bf(acc + weight(downBias, layer, i)));
    }
    return out;
  }

  void candidate(Input in) {
    int keys = fixed;
    int categoryCallDiscard = pack.schema(HipSchema.TRANSITION_CALL_DISCARD);
    int categoryMainTile = pack.schema(HipSchema.TRANSITION_MAIN_TILE);
    int[] rowOf = transitionRows(in);
    for (int t = 0; t < in.transitions; t++) {
      int row = rowOf[t];
      int callDiscard = in.u8(Tensor.TRANSITION_CATEGORY, t, categoryCallDiscard);
      int mainTile = in.u8(Tensor.TRANSITION_CATEGORY, t, categoryMainTile);
      int tileId = callDiscard > 0 ? callDiscard : mainTile;
      float[] start = new float[hidden];
      for (int h = 0; h < hidden; h++) {
        float acc = 0f;
        if (tileId > 0) {
          int token = (1 + seats) * in.rows + tiles * row + tileId - 1;
          for (int k = 0; k < hidden; k++) {
            acc += state[token * hidden + k] * weight(HipWeight.TILE_ENTRY, 0, h * hidden + k);
          }
        }
        start[h] = bf(statics[t * hidden + h] + bf(acc));
      }
      float[] normalized = normalize(start, 0, HipWeight.QUERY_NORM, 0);
      float[] query = new float[attention];
      for (int c = 0; c < attention; c++) {
        float acc = 0f;
        for (int k = 0; k < hidden; k++) {
          acc += normalized[k] * weight(HipWeight.QUERY, 0, c * hidden + k);
        }
        query[c] = bf(acc + weight(HipWeight.QUERY_BIAS, 0, c));
      }
      float[] attended = new float[attention];
      for (int h = 0; h < heads; h++) {
        if (exactCrossAttention) {
          crossAttentionExact(in, t, row, h, query, attended);
          continue;
        }
        float[] logits = new float[keys];
        float maximum = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < keys; j++) {
          int code = in.u8(Tensor.TRANSITION_CODE, t, j);
          int base = (row * keys + j) * 2 * attention + 16 * h;
          float dot = 0f;
          for (int d = 0; d < 16; d++) {
            float key = memory[base + d] + weight(HipWeight.CODE_KEY, 0, code * attention + 16 * h + d);
            dot = Math.fma(query[16 * h + d], key, dot);
          }
          logits[j] = dot * scale;
          maximum = Math.max(maximum, logits[j]);
        }
        float total = 0f;
        for (int j = 0; j < keys; j++) {
          logits[j] = expf(logits[j] - maximum);
          total += logits[j];
        }
        float[] value = new float[16];
        for (int j = 0; j < keys; j++) {
          float p = logits[j] / total;
          int code = in.u8(Tensor.TRANSITION_CODE, t, j);
          int base = (row * keys + j) * 2 * attention + attention + 16 * h;
          for (int d = 0; d < 16; d++) {
            float v = memory[base + d] + weight(HipWeight.CODE_VALUE, 0, code * attention + 16 * h + d);
            value[d] = Math.fma(p, v, value[d]);
          }
        }
        for (int d = 0; d < 16; d++) {
          attended[16 * h + d] = bf(value[d]);
        }
      }
      float[] z = mlpStage(start, attended, 0, true);
      System.arraycopy(z, 0, transitions, t * hidden, hidden);
      float[] n = normalize(z, 0, HipWeight.DISCARD_NORM, 0);
      float sum = 0f;
      for (int g = 0; g < gate; g++) {
        float acc = 0f;
        for (int k = 0; k < hidden; k++) {
          acc += n[k] * weight(HipWeight.DISCARD_HIDDEN, 0, g * hidden + k);
        }
        float u = bf(acc + weight(HipWeight.DISCARD_HIDDEN_BIAS, 0, g));
        sum += silu(u) * weight(HipWeight.DISCARD_OUTPUT, 0, g);
      }
      float exact = 0f;
      for (int k = 0; k < Tensor.DISCARD_EXACT.width(); k++) {
        exact += in.f32(Tensor.DISCARD_EXACT, t, k) * weight(HipWeight.DISCARD_EXACT, 0, k);
      }
      values[t] = sum + exact;
    }
  }

  /** 遷移 t・頭 h の交差注意を FP64 で計算し、o を BF16 に 1 回だけ丸めて attended へ。 */
  private void crossAttentionExact(Input in, int t, int row, int h, float[] query, float[] attended) {
    int keys = fixed;
    double[] logits = new double[keys];
    double maximum = Double.NEGATIVE_INFINITY;
    for (int j = 0; j < keys; j++) {
      int code = in.u8(Tensor.TRANSITION_CODE, t, j);
      int base = (row * keys + j) * 2 * attention + 16 * h;
      double dot = 0;
      for (int d = 0; d < 16; d++) {
        double key = (double) memory[base + d] + weight(HipWeight.CODE_KEY, 0, code * attention + 16 * h + d);
        dot += query[16 * h + d] * key;
      }
      logits[j] = dot * scale;
      maximum = Math.max(maximum, logits[j]);
    }
    double total = 0;
    for (int j = 0; j < keys; j++) {
      logits[j] = Math.exp(logits[j] - maximum);
      total += logits[j];
    }
    double[] value = new double[16];
    for (int j = 0; j < keys; j++) {
      double p = logits[j] / total;
      int code = in.u8(Tensor.TRANSITION_CODE, t, j);
      int base = (row * keys + j) * 2 * attention + attention + 16 * h;
      for (int d = 0; d < 16; d++) {
        value[d] += p * ((double) memory[base + d] + weight(HipWeight.CODE_VALUE, 0, code * attention + 16 * h + d));
      }
    }
    for (int d = 0; d < 16; d++) {
      attended[16 * h + d] = bf(value[d]);
    }
  }

  void valueGraph(Input in) {
    int exactWidth = Tensor.ACTION_EXACT.width();
    float[] delta = new float[in.actions];
    for (int a = 0; a < in.actions; a++) {
      int role = in.action(a, ActionColumn.ROLE);
      int kind = pack.gate(role);
      if (kind < 0) {
        continue;
      }
      HipComposition composition = pack.composition(role);
      int representative = in.action(a, ActionColumn.REPRESENTATIVE);
      float[] chosen =
          aggregate(in.action(representative, ActionColumn.FIRST_TRANSITION),
              in.action(representative, ActionColumn.TRANSITIONS));
      int row = in.action(a, ActionColumn.ROW);
      float[] base;
      if (composition == HipComposition.TURN_GATE) {
        base = bar(in, row);
      } else {
        int baseAction = in.action(a, ActionColumn.BASE);
        int baseRepresentative = in.action(baseAction, ActionColumn.REPRESENTATIVE);
        base = slice(transitions, in.action(baseRepresentative, ActionColumn.FIRST_TRANSITION) * hidden, hidden);
      }
      float[] input = new float[3 * hidden];
      System.arraycopy(normalize(chosen, 0, HipWeight.GATE_NORM, 0), 0, input, 0, hidden);
      System.arraycopy(normalize(base, 0, HipWeight.GATE_NORM, 0), 0, input, hidden, hidden);
      System.arraycopy(state, row * hidden, input, 2 * hidden, hidden);
      float sum = 0f;
      for (int g = 0; g < gate; g++) {
        float acc = 0f;
        for (int k = 0; k < 3 * hidden; k++) {
          acc += input[k] * weight(HipWeight.GATE_HIDDEN, 0, g * 3 * hidden + k);
        }
        float projected = 0f;
        for (int x = 0; x < exactWidth; x++) {
          projected +=
              weight(HipWeight.GATE_EXACT_PROJECTION, 0, (kind * gate + g) * exactWidth + x)
                  * in.f32(Tensor.ACTION_EXACT, a, x);
        }
        float u =
            (bf(acc + weight(HipWeight.GATE_HIDDEN_BIAS, 0, g)) + weight(HipWeight.GATE_KIND_OFFSET, 0, kind * gate + g))
                + projected;
        sum += silu(u) * weight(HipWeight.GATE_OUTPUT, 0, kind * gate + g);
      }
      float direct = 0f;
      for (int x = 0; x < exactWidth; x++) {
        direct += in.f32(Tensor.ACTION_EXACT, a, x) * weight(HipWeight.GATE_EXACT_OUTPUT, 0, kind * exactWidth + x);
      }
      delta[a] = (sum + weight(HipWeight.GATE_OUTPUT_BIAS, 0, kind)) + direct;
    }
    for (int r = 0; r < in.rows; r++) {
      int first = in.row(r, RowColumn.FIRST_ACTION);
      int count = in.row(r, RowColumn.ACTIONS);
      float pivot = 0f;
      boolean havePivot = false;
      for (int a = first; a < first + count; a++) {
        HipComposition composition = pack.composition(in.action(a, ActionColumn.ROLE));
        float score;
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
          default -> score = 0f;
        }
        score = 0f + score;
        scores[a] = score;
        if (composition == HipComposition.DISCARD_VALUE || composition == HipComposition.RIICHI_GATE) {
          pivot = havePivot ? Math.max(pivot, score) : score;
          havePivot = true;
        }
      }
      for (int a = first; a < first + count; a++) {
        if (pack.composition(in.action(a, ActionColumn.ROLE)) == HipComposition.TURN_GATE) {
          scores[a] = 0f + ((havePivot ? pivot : 0f) + delta[a]);
        }
      }
    }
  }

  /** softmax(B/τ) で z を集める（遷移の順、積と和は分けて丸める）。 */
  private float[] aggregate(int first, int count) {
    if (count == 1) {
      return slice(transitions, first * hidden, hidden);
    }
    float maximum = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < count; i++) {
      maximum = Math.max(maximum, values[first + i] / tau);
    }
    float total = 0f;
    for (int i = 0; i < count; i++) {
      total += expf(values[first + i] / tau - maximum);
    }
    float[] e = new float[hidden];
    for (int i = 0; i < count; i++) {
      float weight = expf(values[first + i] / tau - maximum) / total;
      for (int h = 0; h < hidden; h++) {
        e[h] = e[h] + weight * transitions[(first + i) * hidden + h];
      }
    }
    return e;
  }

  /** ē: 行の代表の打牌の softmax(B/τ) で z を集める。 */
  private float[] bar(Input in, int row) {
    int first = in.row(row, RowColumn.FIRST_ACTION);
    int count = in.row(row, RowColumn.ACTIONS);
    float maximum = Float.NEGATIVE_INFINITY;
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        maximum = Math.max(maximum, values[in.action(a, ActionColumn.FIRST_TRANSITION)] / tau);
      }
    }
    float total = 0f;
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        total += expf(values[in.action(a, ActionColumn.FIRST_TRANSITION)] / tau - maximum);
      }
    }
    float[] e = new float[hidden];
    for (int a = first; a < first + count; a++) {
      if (representativeDiscard(in, a)) {
        int t = in.action(a, ActionColumn.FIRST_TRANSITION);
        float weight = expf(values[t] / tau - maximum) / total;
        for (int h = 0; h < hidden; h++) {
          e[h] = e[h] + weight * transitions[t * hidden + h];
        }
      }
    }
    return e;
  }

  private boolean representativeDiscard(Input in, int action) {
    return pack.composition(in.action(action, ActionColumn.ROLE)) == HipComposition.DISCARD_VALUE
        && in.action(action, ActionColumn.REPRESENTATIVE) == action;
  }

  /** ⌊(v·r)·g⌉（r = 1/√(mean(v²) + ε)、FP32）。 */
  private float[] normalize(float[] source, int at, HipWeight gain, int layer) {
    float sum = 0f;
    for (int k = 0; k < hidden; k++) {
      sum += source[at + k] * source[at + k];
    }
    float r = (float) (1.0 / Math.sqrt(sum / hidden + epsilon));
    float[] out = new float[hidden];
    for (int k = 0; k < hidden; k++) {
      out[k] = bf((source[at + k] * r) * weight(gain, layer, k));
    }
    return out;
  }

  private int tokenIndex(int rows, int row, int firstEvent, int position) {
    if (position == 0) {
      return row;
    }
    if (position <= seats) {
      return rows + seats * row + position - 1;
    }
    if (position < fixed) {
      return (1 + seats) * rows + tiles * row + position - 1 - seats;
    }
    return fixed * rows + firstEvent + position - fixed;
  }

  private int type(int position) {
    return position == 0 ? 0 : position <= seats ? 1 : position < fixed ? 2 : 3;
  }

  /** 局・席・牌のトークンの番号の行と位置。 */
  private int[] locate(int rows, int token) {
    if (token < rows) {
      return new int[] {token, 0};
    }
    if (token < (1 + seats) * rows) {
      int u = token - rows;
      return new int[] {u / seats, 1 + u % seats};
    }
    int u = token - (1 + seats) * rows;
    return new int[] {u / tiles, 1 + seats + u % tiles};
  }

  private static int[] transitionRows(Input in) {
    int[] rows = new int[in.transitions];
    for (int r = 0; r < in.rows; r++) {
      int first = in.row(r, RowColumn.FIRST_TRANSITION);
      int count = in.row(r, RowColumn.TRANSITIONS);
      for (int t = first; t < first + count; t++) {
        rows[t] = r;
      }
    }
    return rows;
  }

  private static Tensor tensor(int section) {
    return Tensor.valueOf(HipSection.values()[section].name());
  }

  private static float[] slice(float[] values, int at, int length) {
    float[] out = new float[length];
    System.arraycopy(values, at, out, 0, length);
    return out;
  }

  /** float を BF16 に最近接偶数で丸めた値（NaN は NaN）。 */
  static float bf(float value) {
    int bits = Float.floatToRawIntBits(value);
    if ((bits & 0x7fffffff) > 0x7f800000) {
      return Float.intBitsToFloat(0x7fc00000);
    }
    return Float.intBitsToFloat(((bits + 0x7fff + ((bits >>> 16) & 1)) >>> 16) << 16);
  }

  /** double を BF16 に最近接偶数で丸めた値。float を経て BF16 のちょうど中間に寄ったときは元の値の側で決める（2 回の丸めの誤りを避ける）。 */
  static float bf(double value) {
    float single = (float) value;
    int bits = Float.floatToRawIntBits(single);
    if ((bits & 0xffff) != 0x8000 || value == single || Float.isNaN(single) || Float.isInfinite(single)) {
      return bf(single);
    }
    int upper = bits >>> 16;  // 符号と BF16 の上位。大きさの向きに 1 足すと遠い側
    return Float.intBitsToFloat((Math.abs(value) > Math.abs(single) ? upper + 1 : upper) << 16);
  }

  static float silu(float value) {
    return value / (1f + expf(-value));
  }

  static float expf(float value) {
    return (float) Math.exp(value);
  }
}
