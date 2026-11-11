package com.epsilon.prism.backend;

import com.epsilon.prism.hip.HipArchitecture;
import com.epsilon.prism.hip.HipBatch;
import com.epsilon.prism.hip.HipComposition;
import com.epsilon.prism.hip.HipCounts;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipFfn;
import com.epsilon.prism.hip.HipPack;
import com.epsilon.prism.hip.HipSchema;
import com.epsilon.prism.hip.HipSection;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.hip.LayerRole;
import com.epsilon.prism.hip.TokenGroup;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputBlob.ActionColumn;
import com.epsilon.prism.input.InputBlob.RowColumn;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.model.EagerStages;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.ToIntFunction;

/**
 * HIP の K0〜K9 を CPU で計算する参照の実装。WeightPack（ヘッダーを native の並びで読み直す）と入力の blob だけを読む。式と丸める所は
 * カーネルと同じにし、積和は素朴な FP32 で計算する。形はヘッダーにある形の記述から読む（層の種類、型ごとの FFN の 2 区間、候補層の 2 段目、
 * 数）。四人打ち・三人打ちごとの入力の形式は読まない（領域の幅もヘッダーかバッチから取る）。
 *
 * <p>和の順がカーネルと違うのは、行列積（WMMA）と 1/rms だけ。ほかの所（K1、Attention の softmax と p·v、候補の Cross-Attention、集約、
 * 合成）はカーネルと同じ順で足す。層 0 の q・k・v は K1 と同じ L0-FOLD の式と同じ FP32 の順で作るので、K1 の x と qkv0 はカーネルと
 * ビット単位で一致する。候補層の 2 段目（K9）の softmax は素朴な順で計算する（カーネルはオンラインの softmax）。
 *
 * <p>作業領域の値は、HIP の作業領域と同じ並びの float で持つ（BF16 の領域には BF16 で表せる値が入る）。計算は行ごとに閉じているので、同じ行の
 * s はバッチの組み方によらずビット単位で一致する。方策のテストでは GPU の計画の代わりに使う（公開するのは {@link #forward} だけ）。
 */
public final class HipReference {

  final PackView pack;
  final HipArchitecture architecture;
  final int hidden;
  final int attention;
  final int heads;
  final int gate;
  final int depth;
  final float epsilon;
  final int seats;
  final int tiles;
  final int fixed;
  final float scale;
  final float tau;

  /** 重みごと・項目ごとの値（BF16 と UINT8 は float に直した値。形の記述にない項目は null）。 */
  private final float[][][] cache = new float[HipWeight.values().length][][];

  float[] residual;
  float[] attentionOut;
  float[] state;
  float[] memory;
  float[] statics;
  float[] transitions;
  float[] values;
  float[] scores;

  /** 局・席・牌の層ごとの、イベントのキーとバリュー {@code [イベント, 2A]}。2 つ目からの層の分は、EVENT_KEY_VALUE の領域を層ごとに分けたもの。 */
  float[][] eventKeyValues;

  /** 候補層の 2 段目のキーとバリュー {@code [F·R + イベント, 2A]}（REREAD_MEMORY と同じ並び。2 段目のない形では null）。 */
  float[] rereadMemory;

  /**
   * K6 の Cross-Attention を FP64 で計算し、o を BF16 に 1 回だけ丸める。カーネルの FP32 の順の精度を測る物差しに使う（既定はカーネルと
   * 同じ順の FP32）。
   */
  boolean exactCrossAttention;

  /**
   * K1 の L0-FOLD で作る qkv0（[トークン, 3A]。列はヘッドの対の並びで、QKV0 の領域と同じ値の並び）。重みの束の折り込んだ表から
   * ⌊r·(ΣT'[idx] + ΣW'_d·f + b') + b_qkv⌉ で作る。r は丸める前の x の 1/rms。
   */
  private float[] layer0Qkv;

  /**
   * @param weights 重みの束（ヘッダーを native の並びで読み直す）
   */
  public HipReference(WeightPack weights) {
    pack = new PackView(weights.buffer());
    architecture = weights.layout().architecture();
    hidden = architecture.hidden();
    attention = architecture.attention();
    heads = architecture.heads();
    gate = architecture.gate();
    depth = architecture.depth();
    epsilon = architecture.epsilon();
    seats = architecture.counts().seats();
    tiles = architecture.counts().tiles();
    fixed = architecture.counts().fixed();
    scale = architecture.scale();
    tau = architecture.tau();
    HipPack.Header header = weights.layout().header();
    for (HipWeight weight : HipWeight.values()) {
      cache[weight.ordinal()] = new float[weight.index().entries()][];
      for (int entry = 0; entry < weight.index().entries(); entry++) {
        if (!architecture.holds(weight, entry) || pack.weightOrAbsent(weight, entry) < 0) {
          continue;
        }
        int elements = Math.toIntExact(header.elements(weight, entry));
        float[] values = new float[elements];
        for (int i = 0; i < elements; i++) {
          values[i] =
              switch (weight.element()) {
                case U8 -> pack.u8(weight, entry, i);
                case BF16 -> pack.bf16(weight, entry, i);
                case FP32 -> pack.f32(weight, entry, i);
                case I32 -> throw new IllegalStateException("no int32 weights");
              };
        }
        cache[weight.ordinal()][entry] = values;
      }
    }
  }

  /** 重みの項目の値。項目は層の番号か、FFN の項目か組の番号（項目を持たない重みは 0）。 */
  float weight(HipWeight weight, int entry, int index) {
    return cache[weight.ordinal()][entry][index];
  }

  /** バッチの入力のリーダー。blob の領域を読む（領域の幅は、重みの束のヘッダーかバッチの入力の形式から取る）。 */
  static final class Input {
    final ByteBuffer blob;
    final int rows;
    final int events;
    final int transitions;
    final int actions;
    private final long[] offsets;
    private final int[] widths = new int[Tensor.values().length];

    /** 確定したバッチから読む（領域の幅はバッチの入力の形式から取る）。 */
    Input(HostBatch batch) {
      this(batch.blob(), HipBackend.batchOf(batch), batch.plan().format()::width);
    }

    /**
     * blob と領域の位置から読む。
     *
     * @param blob 入力の blob（位置 0 から）
     * @param batch バッチの各数（行・イベント・遷移・手）と領域の位置
     * @param widths 領域ごとの、1 個あたりの要素の数
     */
    Input(ByteBuffer blob, HipBatch batch, ToIntFunction<Tensor> widths) {
      this.blob = blob.duplicate().order(ByteOrder.LITTLE_ENDIAN);
      rows = batch.rows();
      events = batch.events();
      transitions = batch.transitions();
      actions = batch.actions();
      offsets = batch.sections();
      for (Tensor tensor : Tensor.values()) {
        this.widths[tensor.ordinal()] = widths.applyAsInt(tensor);
      }
    }

    /** 領域の 1 個あたりの要素の数。 */
    int width(Tensor tensor) {
      return widths[tensor.ordinal()];
    }

    private int at(Tensor tensor, int row, int column, int bytes) {
      return Math.toIntExact(offsets[HipSection.valueOf(tensor.name()).ordinal()]
          + ((long) row * widths[tensor.ordinal()] + column) * bytes);
    }

    int u8(Tensor tensor, int row, int column) {
      return blob.get(at(tensor, row, column, 1)) & 0xFF;
    }

    float bf16(Tensor tensor, int row, int column) {
      return Float.intBitsToFloat((blob.getShort(at(tensor, row, column, Short.BYTES)) & 0xFFFF) << 16);
    }

    float f32(Tensor tensor, int row, int column) {
      return blob.getFloat(at(tensor, row, column, Float.BYTES));
    }

    int i32(Tensor tensor, int row, int column) {
      return blob.getInt(at(tensor, row, column, Integer.BYTES));
    }

    int row(int row, RowColumn column) {
      return i32(Tensor.ROW_COLUMN, row, column.ordinal());
    }

    int action(int action, ActionColumn column) {
      return i32(Tensor.ACTION_COLUMN, action, column.ordinal());
    }
  }

  /**
   * 入力のリーダーを作る（領域の幅は重みの束のヘッダーから取る）。
   *
   * @param blob 入力の blob（位置 0 から）
   * @param batch バッチの各数と領域の位置
   * @return リーダー
   */
  Input input(ByteBuffer blob, HipBatch batch) {
    return new Input(blob, batch, tensor -> pack.width(HipSection.valueOf(tensor.name())));
  }

  /**
   * 1 バッチの順伝播（領域の幅はバッチの入力の形式から読む）。
   *
   * @param batch 確定したバッチ
   * @return s（バッチの手の順）
   */
  public float[] forward(HostBatch batch) {
    return forward(new Input(batch));
  }

  /**
   * 1 バッチの順伝播（領域の幅は重みの束のヘッダーから読む）。
   *
   * @param blob 入力の blob（位置 0 から）
   * @param batch バッチの各数と領域の位置
   * @return s（バッチの手の順）
   */
  public float[] forward(ByteBuffer blob, HipBatch batch) {
    return forward(input(blob, batch));
  }

  /** 1 バッチの順伝播（native と同じ起動の順で計算する）。 */
  float[] forward(Input in) {
    run(in, null);
    return scores.clone();
  }

  /**
   * 1 バッチの順伝播を、段ごとの値として返す（eager の段と同じ並び。最後の層の x は、最後の正規化の前の値）。
   *
   * @param in バッチの入力
   * @return 段ごとの値
   */
  EagerStages.Stages stages(Input in) {
    Trace trace = new Trace();
    run(in, trace);
    return new EagerStages.Stages(
        trace.embedded,
        statics.clone(),
        trace.attention,
        trace.layers,
        Arrays.stream(eventKeyValues).map(float[]::clone).toList(),
        state.clone(),
        memory.clone(),
        rereadMemory == null ? null : rereadMemory.clone(),
        trace.candidates,
        transitions.clone(),
        values.clone(),
        scores.clone());
  }

  /** 段ごとの値のコピー（{@link #stages} が集める）。 */
  private static final class Trace {
    float[] embedded;
    final List<float[]> attention = new ArrayList<>();
    final List<float[]> layers = new ArrayList<>();
    float[] candidates;
  }

  private void run(Input in, Trace trace) {
    allocate(in);
    embed(in);
    if (trace != null) {
      trace.embedded = residual.clone();
    }
    int allTokens = architecture.allTokenLayers();
    int fixedTokens = fixed * in.rows;
    int everyToken = fixedTokens + in.events;
    for (int layer = 0; layer < allTokens; layer++) {
      attention(in, layer);
      mlp(in, layer);
      if (trace != null) {
        trace.attention.add(Arrays.copyOf(attentionOut, everyToken * attention));
        trace.layers.add(residual.clone());
      }
    }
    eventKeyValue(in);
    for (int layer = allTokens; layer < depth; layer++) {
      attention(in, layer);
      if (trace != null) {
        trace.attention.add(Arrays.copyOf(attentionOut, fixedTokens * attention));
      }
      if (layer + 1 < depth) {
        mlp(in, layer);
      } else {
        mlpLast(in, layer);
      }
      if (trace != null) {
        trace.layers.add(Arrays.copyOf(residual, fixedTokens * hidden));
      }
    }
    if (architecture.reread()) {
      rereadKeyValue(in);
    }
    candidate(in);
    if (trace != null) {
      trace.candidates = transitions.clone();
    }
    if (architecture.reread()) {
      reread(in);
    }
    valueGraph(in);
  }

  /** 作業領域をバッチの大きさで確保する。 */
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
    eventKeyValues = new float[architecture.fixedLayers()][in.events * 2 * attention];
    rereadMemory = architecture.reread() ? new float[tokens * 2 * attention] : null;
    layer0Qkv = new float[tokens * 3 * attention];
  }

  /** 指定した型のトークンの数。 */
  int slotCount(HipEmbedding.Slot slot, Input in) {
    return switch (slot) {
      case ROUND -> in.rows;
      case SEAT -> seats * in.rows;
      case TILE -> tiles * in.rows;
      case EVENT -> in.events;
      case TRANSITION -> in.transitions;
    };
  }

  /** 指定した型の最初のトークンの番号（遷移は遷移の領域の中の番号）。 */
  int slotStart(HipEmbedding.Slot slot, Input in) {
    return switch (slot) {
      case ROUND, TRANSITION -> 0;
      case SEAT -> in.rows;
      case TILE -> (1 + seats) * in.rows;
      case EVENT -> fixed * in.rows;
    };
  }

  /** K1 が作った、層 0 の qkv0（QKV0 の領域と同じ並び）。 */
  float[] layer0Qkv() {
    return layer0Qkv;
  }

  void embed(Input in) {
    int width = 3 * attention;
    for (HipEmbedding.Slot slot : HipEmbedding.Slot.values()) {
      PackView.Spec spec = pack.embedding(slot);
      int count = slotCount(slot, in);
      int start = slotStart(slot, in);
      float[] out = slot == HipEmbedding.Slot.TRANSITION ? statics : residual;
      Tensor category = tensor(spec.category());
      Tensor scalar = spec.scalar() < 0 ? null : tensor(spec.scalar());
      Tensor exact = spec.exact() < 0 ? null : tensor(spec.exact());
      boolean folding = slot != HipEmbedding.Slot.TRANSITION;
      int[] entries = new int[spec.tableOffset().length];
      float[] features = new float[spec.source().length];
      for (int t = 0; t < count; t++) {
        float[] acc = new float[hidden];
        for (int table = 0; table < spec.tableOffset().length; table++) {
          int entry = 0;
          for (int m = 0; m < spec.columns()[table].length; m++) {
            entry += in.u8(category, t, spec.columns()[table][m]) * spec.strides()[table][m];
          }
          entries[table] = entry;
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
          features[k] = feature;
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
        if (folding) {
          foldToken(spec, acc, entries, features, (start + t) * width);
        }
      }
    }
  }

  /**
   * K1 の L0-FOLD を 1 トークン分計算する（k1_embed.hip と同じ順）。r = 1/√(Σ/H + ε) で、Σ は lane ごとに H/32 列の acc² を fma で順に足し、
   * 32 lane の分を xor の木で足したもの。列ごとに s = 0 から、表の順に T'、特徴の順に fma で W'_d·f、最後に b' を足し、⌊r·s + b_qkv⌉ を書く
   * （積と和は別々に丸める）。
   */
  private void foldToken(PackView.Spec spec, float[] acc, int[] entries, float[] features, int at) {
    int width = 3 * attention;
    int lanes = 32;
    int perLane = hidden / lanes;
    float[] partial = new float[lanes];
    for (int lane = 0; lane < lanes; lane++) {
      float square = 0f;
      for (int c = 0; c < perLane; c++) {
        square = Math.fma(acc[lane * perLane + c], acc[lane * perLane + c], square);
      }
      partial[lane] = square;
    }
    for (int stride = lanes / 2; stride > 0; stride >>= 1) {
      float[] next = new float[lanes];
      for (int lane = 0; lane < lanes; lane++) {
        next[lane] = partial[lane] + partial[lane ^ stride];
      }
      partial = next;
    }
    float r = 1f / (float) Math.sqrt(partial[0] / hidden + epsilon);
    int dense = features.length;
    for (int o = 0; o < width; o++) {
      float s = 0f;
      for (int table = 0; table < entries.length; table++) {
        s += pack.bytes.getFloat(spec.tableFold()[table] + (entries[table] * width + o) * Float.BYTES);
      }
      for (int k = 0; k < dense; k++) {
        s = Math.fma(pack.bytes.getFloat(spec.fold() + (k * width + o) * Float.BYTES), features[k], s);
      }
      s += pack.bytes.getFloat(spec.fold() + (dense * width + o) * Float.BYTES);
      layer0Qkv[at + o] = bf(r * s + pack.bytes.getFloat(spec.fold() + ((dense + 1) * width + o) * Float.BYTES));
    }
  }

  /** バックボーンの 1 層について、Attention の入力 n = ⌊(x·r)·g⌉ を x の全てのトークンで求める（NORMALIZED の領域と同じ並び）。 */
  float[] attentionInput(float[] x, int layer) {
    float[] out = new float[x.length];
    for (int t = 0; t < x.length / hidden; t++) {
      System.arraycopy(normalize(x, t * hidden, HipWeight.ATTENTION_NORM, layer), 0, out, t * hidden, hidden);
    }
    return out;
  }

  /**
   * バックボーンの 1 層の Attention。全トークンの層（K2）では、全てのトークンがクエリになる。局・席・牌の層（K4）では局・席・牌だけが
   * クエリになり、局・席・牌のキーとバリューは残差の流れから射影し、イベントのキーとバリューは {@link #eventKeyValues} から読む。
   */
  void attention(Input in, int layer) {
    boolean all = architecture.layer(layer).role() == LayerRole.ALL_TOKENS;
    float[] eventKeyValue = all ? null : eventKeyValues[architecture.fixedIndex(layer)];
    int relationWidth = in.width(Tensor.RELATION);
    int columnTile = pack.schema(HipSchema.RELATION_TILE);
    int columnOwner = pack.schema(HipSchema.RELATION_OWNER);
    int columnEvent = pack.schema(HipSchema.RELATION_EVENT);
    int columnRiichi = pack.schema(HipSchema.RELATION_RIICHI);
    int columnDiscard = pack.schema(HipSchema.RELATION_LAST_DISCARD);
    int columnLink = pack.schema(HipSchema.RELATION_LINK);
    int columnSafety = pack.schema(HipSchema.RELATION_SAFETY);
    int safeties = architecture.counts().opponents();
    for (int r = 0; r < in.rows; r++) {
      int firstEvent = in.row(r, RowColumn.FIRST_EVENT);
      int tokens = fixed + in.row(r, RowColumn.EVENTS);
      int queries = all ? tokens : fixed;
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
        if (layer == 0) {
          // L0-FOLD: K1 が作った qkv0（ヘッドの対の並び）を、q・k・v の並びに直して読む。
          for (int col = 0; col < 3 * attention; col++) {
            if (col < attention && p >= queries) {
              continue;
            }
            qkv[p][col] = layer0Qkv[global[p] * 3 * attention + HipEmbedding.foldColumn(col, attention)];
          }
          continue;
        }
        if (!all && p >= fixed) {
          // イベントの k・v（最後の全トークンの層で、イベントの x から作ったもの）。
          int event = firstEvent + p - fixed;
          System.arraycopy(eventKeyValue, event * 2 * attention, qkv[p], attention, 2 * attention);
          continue;
        }
        float[] normalized = normalize(residual, global[p] * hidden, HipWeight.ATTENTION_NORM, layer);
        for (int col = 0; col < 3 * attention; col++) {
          if (col < attention && p >= queries) {
            continue;
          }
          qkv[p][col] = bf(dot(normalized, HipWeight.QKV, layer, col) + weight(HipWeight.QKV_BIAS, layer, col));
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

  /** eager の packed Attention の GPU カーネルと同じ順の和。64 個の部分和（v 個目はキー v, v+64, … を順に足す）を 2 分木で足す。 */
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

  /**
   * RelationClass と同じ、7 区分の関係のコード。席の区分は席の数 S から決める（クエリとキーの両方に持ち主の席があれば
   * (oj − oi + S) mod S + 1、キーだけなら S + oj、クエリだけなら 2S + oi）。
   */
  int[] segments(
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
      segment[3] = Math.floorMod(ownerK - ownerQ + seats, seats) + 1;
    } else if (ownerK > 0) {
      segment[3] = ownerK + seats;
    } else if (ownerQ > 0) {
      segment[3] = ownerQ + 2 * seats;
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

  /** 行のトークンより後ろのキーに K0 が書くワード。セグメント 6 に 15 を置き、Attention から外す印にする。 */
  static final int PADDED_KEY_CODE = 15 << 27;

  /**
   * K0 が 1 つの行に書く関係のコードを求める。行のトークンを {@link HipCounts#CODE_BLOCK} 個ずつのタイルに分け、クエリとキーのタイルの
   * 組ごとに 256 ワードを書く。ワードはウェーブのレーンごとに 8 つずつ並び、クエリはレーンの下 4 bit、キーの偶奇はレーンの 5 bit 目で
   * 決まる。行のトークンより後ろのキーは {@link #PADDED_KEY_CODE}、後ろのクエリは 0 になる。
   *
   * @param in バッチの入力
   * @param row 行
   * @return クエリのタイルごとの、キーのタイル × 256 ワード
   */
  int[][] relationCodes(Input in, int row) {
    int relationWidth = in.width(Tensor.RELATION);
    int columnTile = pack.schema(HipSchema.RELATION_TILE);
    int columnOwner = pack.schema(HipSchema.RELATION_OWNER);
    int columnEvent = pack.schema(HipSchema.RELATION_EVENT);
    int columnRiichi = pack.schema(HipSchema.RELATION_RIICHI);
    int columnDiscard = pack.schema(HipSchema.RELATION_LAST_DISCARD);
    int columnLink = pack.schema(HipSchema.RELATION_LINK);
    int columnSafety = pack.schema(HipSchema.RELATION_SAFETY);
    int safeties = architecture.counts().opponents();
    int firstEvent = in.row(row, RowColumn.FIRST_EVENT);
    int tokens = fixed + in.row(row, RowColumn.EVENTS);
    int[][] attributes = new int[tokens][relationWidth];
    for (int p = 0; p < tokens; p++) {
      int token = tokenIndex(in.rows, row, firstEvent, p);
      for (int c = 0; c < relationWidth; c++) {
        attributes[p][c] = in.u8(Tensor.RELATION, token, c);
      }
    }
    int now = attributes[0][columnEvent];
    int blocks = Math.ceilDiv(tokens, HipCounts.CODE_BLOCK);
    int[][] words = new int[blocks][blocks * 256];
    for (int queryBlock = 0; queryBlock < blocks; queryBlock++) {
      for (int keyBlock = 0; keyBlock < blocks; keyBlock++) {
        for (int t = 0; t < 256; t++) {
          int lane = t >> 3;
          int i = HipCounts.CODE_BLOCK * queryBlock + (lane & 15);
          int j = HipCounts.CODE_BLOCK * keyBlock + 2 * (t & 7) + (lane >> 4);
          int word = PADDED_KEY_CODE;
          if (j < tokens) {
            word =
                i < tokens
                    ? code(
                        segments(type(i), type(j), attributes[i], attributes[j], now, columnTile, columnOwner,
                            columnEvent, columnRiichi, columnDiscard, columnLink, columnSafety, safeties))
                    : 0;
          }
          words[queryBlock][keyBlock * 256 + t] = word;
        }
      }
    }
    return words;
  }

  /** 7 つのセグメントを 1 ワードに詰める。セグメント 0〜5 は 2 つずつ 3 つの表の 8 bit に、セグメント 6 は上の 5 bit に入れる。 */
  private static int code(int[] segment) {
    int code = segment[6] << 27;
    for (int table = 0; table < 3; table++) {
      int a = segment[2 * table];
      int b = segment[2 * table + 1];
      code |= ((a ^ b) | b << 4) << (8 * table + 3);
    }
    return code;
  }

  /**
   * バックボーンの 1 層の MLP（最後の層を除く）。全トークンの層（K3 の MLP_ALL）は全てのトークンを、局・席・牌の層（K3 の MLP_FIXED）は
   * 局・席・牌だけを書き換える。型ごとの FFN なら、局・席と牌をそれぞれの組の FFN に通す。
   */
  void mlp(Input in, int layer) {
    boolean all = architecture.layer(layer).role() == LayerRole.ALL_TOKENS;
    int tokens = all ? fixed * in.rows + in.events : fixed * in.rows;
    for (int t = 0; t < tokens; t++) {
      float[] x = layerStage(in, layer, t);
      System.arraycopy(x, 0, residual, t * hidden, hidden);
    }
  }

  /**
   * 最後の層の MLP（K5）。局・席・牌の x から最後の正規化 y（組ごとのゲイン）を求め、候補層の Cross-Attention のキーとバリューを作る。
   * x は残差の流れの局・席・牌にも書く（K5 は書かないが、後の段はここを読まない）。
   */
  void mlpLast(Input in, int layer) {
    int tokens = fixed * in.rows;
    for (int t = 0; t < tokens; t++) {
      float[] x = layerStage(in, layer, t);
      System.arraycopy(x, 0, residual, t * hidden, hidden);
      TokenGroup group = architecture.finalGroups().contains(TokenGroup.ALL) ? TokenGroup.ALL : contextOrTile(in, t);
      float[] y = normalize(x, 0, HipWeight.FINAL_NORM, group.ordinal());
      System.arraycopy(y, 0, state, t * hidden, hidden);
      int[] located = locate(in.rows, t);
      project(y, HipWeight.KEY_VALUE, HipWeight.KEY_VALUE_BIAS, 0, 2 * attention, memory,
          (located[0] * fixed + located[1]) * 2 * attention);
    }
  }

  /** 1 層の 1 トークンについて、O 射影と残差を足してから組の FFN に通す（型ごとの FFN なら、そのトークンの組の重みを使う）。 */
  private float[] layerStage(Input in, int layer, int t) {
    HipFfn ffn = architecture.layer(layer).ffn();
    TokenGroup group = ffn instanceof HipFfn.Typed ? contextOrTile(in, t) : TokenGroup.ALL;
    return mlpStage(slice(residual, t * hidden, hidden), slice(attentionOut, t * attention, attention),
        HipWeight.OUTPUT, HipWeight.OUTPUT_BIAS, layer, HipWeight.ffnEntry(layer, group));
  }

  /** 局・席・牌のトークン（トークンの軸の番号）が属する組。局・席は先頭の (1 + S)·R 個で、牌はその後。 */
  private TokenGroup contextOrTile(Input in, int t) {
    return t < (1 + seats) * in.rows ? TokenGroup.CONTEXT : TokenGroup.TILE;
  }

  /**
   * 最後の全トークンの層のイベントの x から、局・席・牌の層ごとのイベントの [k | v] を作る。使うのは、その層の Attention の前のゲインと、
   * W_qkv の k・v の行。カーネルでは、最初の層の分は K4 が n から作り、後の層の分は最後の全トークンの層の K3 が作る。2 段目のある形では、
   * 2 段目のイベントのキーとバリューも作る（2 段目のイベントのゲインと、キー・バリューの射影を使う。カーネルでは最後の全トークンの層の K3 が作る）。
   */
  void eventKeyValue(Input in) {
    int first = fixed * in.rows;
    for (int layer = architecture.allTokenLayers(); layer < depth; layer++) {
      float[] out = eventKeyValues[architecture.fixedIndex(layer)];
      for (int e = 0; e < in.events; e++) {
        float[] n = normalize(residual, (first + e) * hidden, HipWeight.ATTENTION_NORM, layer);
        for (int c = 0; c < 2 * attention; c++) {
          out[e * 2 * attention + c] =
              bf(dot(n, HipWeight.QKV, layer, attention + c) + weight(HipWeight.QKV_BIAS, layer, attention + c));
        }
      }
    }
    if (architecture.reread()) {
      for (int e = 0; e < in.events; e++) {
        float[] n = normalize(residual, (first + e) * hidden, HipWeight.REREAD_EVENT_NORM, 0);
        project(n, HipWeight.REREAD_KEY_VALUE, HipWeight.REREAD_KEY_VALUE_BIAS, 0, 2 * attention, rereadMemory,
            (first + e) * 2 * attention);
      }
    }
  }

  /** K8 の REREAD_KEY_VALUE。最後の正規化の後の y から、2 段目の局・席・牌のキーとバリューを [行, 位置, 2A] に書く（ここでは正規化しない）。 */
  void rereadKeyValue(Input in) {
    for (int t = 0; t < fixed * in.rows; t++) {
      int[] located = locate(in.rows, t);
      project(slice(state, t * hidden, hidden), HipWeight.REREAD_KEY_VALUE, HipWeight.REREAD_KEY_VALUE_BIAS, 0,
          2 * attention, rereadMemory, (located[0] * fixed + located[1]) * 2 * attention);
    }
  }

  /** out[at + c] = ⌊v·W[c]ᵀ + b[c]⌉（c < width、W の行は H 列）。 */
  private void project(float[] v, HipWeight weight, HipWeight bias, int entry, int width, float[] out, int at) {
    for (int c = 0; c < width; c++) {
      out[at + c] = bf(dot(v, weight, entry, c) + weight(bias, entry, c));
    }
  }

  /** Σ_k v[k]·W[row, k]（W は [?, H]。FP32 で k の順に足す）。 */
  private float dot(float[] v, HipWeight weight, int entry, int row) {
    float acc = 0f;
    for (int k = 0; k < hidden; k++) {
      acc += v[k] * weight(weight, entry, row * hidden + k);
    }
    return acc;
  }

  /** K3 の式: x1 = ⌊x + ⌊o·W_Oᵀ + b_O⌉⌉、x1 + ⌊W_d·⌊silu(⌊W_u·N(x1) + b_u⌉)⌉ + b_d⌉（積の和はそれぞれ 1 回だけ丸める）。 */
  private float[] mlpStage(float[] x, float[] o, HipWeight outWeight, HipWeight outBias, int outEntry, int entry) {
    int ffn = architecture.ffnWidth(entry);
    float[] x1 = new float[hidden];
    for (int i = 0; i < hidden; i++) {
      float acc = 0f;
      for (int k = 0; k < attention; k++) {
        acc += o[k] * weight(outWeight, outEntry, i * attention + k);
      }
      x1[i] = bf(x[i] + bf(acc + weight(outBias, outEntry, i)));
    }
    float[] n = normalize(x1, 0, HipWeight.FFN_NORM, entry);
    float[] h = new float[ffn];
    for (int j = 0; j < ffn; j++) {
      h[j] = bf(silu(bf(dot(n, HipWeight.UP, entry, j) + weight(HipWeight.UP_BIAS, entry, j))));
    }
    float[] out = new float[hidden];
    for (int i = 0; i < hidden; i++) {
      float acc = 0f;
      for (int j = 0; j < ffn; j++) {
        acc += h[j] * weight(HipWeight.DOWN, entry, i * ffn + j);
      }
      out[i] = bf(x1[i] + bf(acc + weight(HipWeight.DOWN_BIAS, entry, i)));
    }
    return out;
  }

  /** K6。候補層の 1 段目の z（2 段目のある形では z1）を作る。1 段だけの形では B も書く。 */
  void candidate(Input in) {
    int keys = fixed;
    int[] rowOf = transitionRows(in);
    for (int t = 0; t < in.transitions; t++) {
      int row = rowOf[t];
      int tileId = discardTile(in, t);
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
        query[c] = bf(dot(normalized, HipWeight.QUERY, 0, c) + weight(HipWeight.QUERY_BIAS, 0, c));
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
      float[] z =
          mlpStage(start, attended, HipWeight.CANDIDATE_OUTPUT, HipWeight.CANDIDATE_OUTPUT_BIAS, 0,
              HipWeight.CANDIDATE_FFN);
      System.arraycopy(z, 0, transitions, t * hidden, hidden);
      if (!architecture.reread()) {
        values[t] = discardValue(in, t, z);
      }
    }
  }

  /** 1 つの遷移・1 つのヘッドの Cross-Attention を FP64 で計算し、o を BF16 に 1 回だけ丸めて結果の配列に書く。 */
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

  /**
   * K9。候補層の 2 段目。遷移ごとに z1 からクエリを作り、行の局・席・牌（遷移のコードを使う）とイベント（τ とイベントの牌の関係のコードを
   * 使う）のキーとバリューを読み直して、z = ⌊z1 + ⌊o·W_Oᵀ + b_O⌉⌉ と B を書く。
   *
   * <pre>
   * q = ⌊⌊N_q(z1)⌉·W_qᵀ + b_q⌉
   * o = ⌊Σ_j softmax_j(q·(K_j + E_k[c_j])·scale)·(V_j + E_v[c_j])⌉
   * c_j = 遷移のコード（局・席・牌のキー）、keyCodes + 1 + 牌の関係の種類（τ とイベントの牌が両方あるとき）、keyCodes（ほか）
   * </pre>
   */
  void reread(Input in) {
    int width = 2 * attention;
    int keyCodes = architecture.counts().keyCodes();
    int tileColumn = pack.schema(HipSchema.RELATION_TILE);
    int eventBase = fixed * in.rows;
    int[] rowOf = transitionRows(in);
    for (int t = 0; t < in.transitions; t++) {
      int row = rowOf[t];
      int firstEvent = in.row(row, RowColumn.FIRST_EVENT);
      int events = in.row(row, RowColumn.EVENTS);
      int keys = fixed + events;
      int discard = discardTile(in, t);
      float[] z1 = slice(transitions, t * hidden, hidden);
      float[] normalized = normalize(z1, 0, HipWeight.REREAD_QUERY_NORM, 0);
      float[] query = new float[attention];
      for (int c = 0; c < attention; c++) {
        query[c] = bf(dot(normalized, HipWeight.REREAD_QUERY, 0, c) + weight(HipWeight.REREAD_QUERY_BIAS, 0, c));
      }
      int[] codes = new int[keys];
      int[] bases = new int[keys];
      for (int j = 0; j < keys; j++) {
        if (j < fixed) {
          codes[j] = in.u8(Tensor.TRANSITION_CODE, t, j);
          bases[j] = (row * fixed + j) * width;
        } else {
          int event = firstEvent + j - fixed;
          int eventTile = in.u8(Tensor.RELATION, eventBase + event, tileColumn);
          codes[j] =
              discard > 0 && eventTile > 0
                  ? keyCodes + 1 + (int) weight(HipWeight.TILE_RELATION, 0, (discard - 1) * tiles + eventTile - 1)
                  : keyCodes;
          bases[j] = (eventBase + event) * width;
        }
      }
      float[] attended = new float[attention];
      for (int h = 0; h < heads; h++) {
        float[] logits = new float[keys];
        float maximum = Float.NEGATIVE_INFINITY;
        for (int j = 0; j < keys; j++) {
          float dot = 0f;
          for (int d = 0; d < 16; d++) {
            int column = 16 * h + d;
            float key = rereadMemory[bases[j] + column] + weight(HipWeight.REREAD_CODE_KEY, 0, codes[j] * attention + column);
            dot = Math.fma(query[column], key, dot);
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
          for (int d = 0; d < 16; d++) {
            int column = 16 * h + d;
            float v =
                rereadMemory[bases[j] + attention + column]
                    + weight(HipWeight.REREAD_CODE_VALUE, 0, codes[j] * attention + column);
            value[d] = Math.fma(p, v, value[d]);
          }
        }
        for (int d = 0; d < 16; d++) {
          attended[16 * h + d] = bf(value[d]);
        }
      }
      for (int i = 0; i < hidden; i++) {
        float acc = 0f;
        for (int k = 0; k < attention; k++) {
          acc += attended[k] * weight(HipWeight.REREAD_OUTPUT, 0, i * attention + k);
        }
        transitions[t * hidden + i] = bf(z1[i] + bf(acc + weight(HipWeight.REREAD_OUTPUT_BIAS, 0, i)));
      }
      values[t] = discardValue(in, t, slice(transitions, t * hidden, hidden));
    }
  }

  /** 遷移の τ。鳴いた後に捨てる牌で、無ければ主牌（牌の番号 + 1。どちらも無ければ 0）。 */
  private int discardTile(Input in, int t) {
    int callDiscard = in.u8(Tensor.TRANSITION_CATEGORY, t, pack.schema(HipSchema.TRANSITION_CALL_DISCARD));
    return callDiscard > 0 ? callDiscard : in.u8(Tensor.TRANSITION_CATEGORY, t, pack.schema(HipSchema.TRANSITION_MAIN_TILE));
  }

  /** B = Σ_g silu(⌊W_B·N_B(z) + c_B⌉)·wB + Σ uBx·B（後ろの和は厳密な特徴の経路。K6 と K9 に共通の式）。 */
  private float discardValue(Input in, int t, float[] z) {
    float[] n = normalize(z, 0, HipWeight.DISCARD_NORM, 0);
    float sum = 0f;
    for (int g = 0; g < gate; g++) {
      float u = bf(dot(n, HipWeight.DISCARD_HIDDEN, 0, g) + weight(HipWeight.DISCARD_HIDDEN_BIAS, 0, g));
      sum += silu(u) * weight(HipWeight.DISCARD_OUTPUT, 0, g);
    }
    float exact = 0f;
    for (int k = 0; k < in.width(Tensor.DISCARD_EXACT); k++) {
      exact += in.f32(Tensor.DISCARD_EXACT, t, k) * weight(HipWeight.DISCARD_EXACT, 0, k);
    }
    return sum + exact;
  }

  void valueGraph(Input in) {
    int exactWidth = in.width(Tensor.ACTION_EXACT);
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

  /** softmax(B/τ) の重みで z を集める（遷移の順に足し、積と和は別々に丸める）。 */
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

  /** ē。行の代表の打牌について、softmax(B/τ) の重みで z を集める。 */
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
  private float[] normalize(float[] source, int at, HipWeight gain, int entry) {
    float sum = 0f;
    for (int k = 0; k < hidden; k++) {
      sum += source[at + k] * source[at + k];
    }
    float r = (float) (1.0 / Math.sqrt(sum / hidden + epsilon));
    float[] out = new float[hidden];
    for (int k = 0; k < hidden; k++) {
      out[k] = bf((source[at + k] * r) * weight(gain, entry, k));
    }
    return out;
  }

  int tokenIndex(int rows, int row, int firstEvent, int position) {
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

  int type(int position) {
    return position == 0 ? 0 : position <= seats ? 1 : position < fixed ? 2 : 3;
  }

  /** 局・席・牌のトークンの番号から、行と行の中の位置を求める。 */
  int[] locate(int rows, int token) {
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

  static int[] transitionRows(Input in) {
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

  static Tensor tensor(int section) {
    return Tensor.valueOf(HipSection.values()[section].name());
  }

  static float[] slice(float[] values, int at, int length) {
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

  /**
   * double を BF16 に最近接偶数で丸めた値。float に直した時点で BF16 のちょうど中間に来たときは、元の値がどちら側にあるかで決める
   * （2 回丸める誤りを避ける）。
   */
  static float bf(double value) {
    float single = (float) value;
    int bits = Float.floatToRawIntBits(single);
    if ((bits & 0xffff) != 0x8000 || value == single || Float.isNaN(single) || Float.isInfinite(single)) {
      return bf(single);
    }
    int upper = bits >>> 16;  // 符号と BF16 の上位ビット。1 足すと絶対値の大きい側になる
    return Float.intBitsToFloat((Math.abs(value) > Math.abs(single) ? upper + 1 : upper) << 16);
  }

  static float silu(float value) {
    return value / (1f + expf(-value));
  }

  static float expf(float value) {
    return (float) Math.exp(value);
  }
}
