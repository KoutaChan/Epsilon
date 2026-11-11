package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.agent.facts.MahjongTileRelation;
import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.prism.hip.HipEmbedding;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.model.PrismNetwork;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * HIP のカーネルが読む推論の重みの束。ネットワークを公開するたびに FP32 のマスターの重みから作り、ネットワークのバージョン番号を持つ。
 * 推論のたびにこの番号を照合するので、古い重みが残ったまま推論することはない。中身は {@link WeightLayout} の並びに置いた direct の
 * ByteBuffer（little-endian、ヘッダーつき）。
 *
 * <p>埋め込みの組の表は、BF16 に丸めた W と E の積を FP32 で足したもの。列ごとに P_m[c] = W_m·E[off_m + c] を作り、組の各行には
 * Σ_m P_m[c_m] を列の順に足して置く。eager の Linear との違いは足す順だけ。
 *
 * <p>局・席・牌・イベントの型では、層 0 の Attention の QKV の射影もこの表へ折り込む。書いた束の T・W_d・b（BF16 の値）と層 0 の
 * g・W_qkv（BF16 の値）から、T' = T·diag(g)·W_qkvᵀ、W'_d = W_d·diag(g)·W_qkvᵀ、b' = b·diag(g)·W_qkvᵀ を FP64 で求め、FP32 に
 * 丸めて置く。b' の後ろには b_qkv（BF16 に丸めた値）を写す。列はヘッドの対の並び（{@link HipEmbedding#foldColumn}）。
 */
public final class WeightPack {

  private final WeightLayout layout;
  private final ByteBuffer buffer;
  private final long version;

  private WeightPack(WeightLayout layout, ByteBuffer buffer, long version) {
    this.layout = layout;
    this.buffer = buffer;
    this.version = version;
  }

  /**
   * 保存したネットワークから作る。重みを CPU に FP32 で読み、表と束を作ってからネットワークを捨てる。束と表はネットワークの配列を
   * 参照しない。
   *
   * @param model ネットワークのディレクトリ（weights.bin と model.json）
   * @param version ネットワークのバージョン番号（0 以上）
   * @return 重みの束（表は {@link #layout()}）
   * @throws IOException 読めないとき。形式・フィンガープリント・重みが合わないとき
   */
  public static WeightPack load(Path model, long version) throws IOException {
    NetworkRuntime.initialize();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      PrismNetwork network = PrismNetwork.load(model, manager);
      return build(WeightLayout.of(network), network, version);
    }
  }

  /**
   * ネットワークの今の重みから作る。
   *
   * @param layout 表（形とパラメーターの名前が同じネットワークから作ったもの）
   * @param network ネットワーク（推論の重みの束を読む。どのデバイスにあってもよい）
   * @param version ネットワークのバージョン番号（0 以上）
   * @return 重みの束
   */
  public static WeightPack build(WeightLayout layout, PrismNetwork network, long version) {
    return build(layout, network.inferenceParameters(), version);
  }

  /**
   * 推論の重みの束から作る。牌の関係の種類は、表の入力の形式から作る。中身は牌のトークンの対ごとの {@link MahjongTileRelation} の
   * ordinal で、形は [牌, 牌]（{@link TileTokens#relations()}）。
   *
   * @param layout 表
   * @param parameters 推論の重みの束（{@link PrismNetwork#inferenceParameters()}）
   * @param version ネットワークのバージョン番号（0 以上）
   * @return 重みの束
   * @throws IllegalArgumentException 表にあるパラメーターが束に無いとき。値が有限でないとき。牌の関係の表の大きさが、形の記述の牌の数の
   *     2 乗でないとき
   */
  public static WeightPack build(WeightLayout layout, PairList<String, Parameter> parameters, long version) {
    Objects.requireNonNull(layout, "layout");
    if (version < 0) {
      throw new IllegalArgumentException("weights version must not be negative: " + version);
    }
    Map<String, float[]> values = new HashMap<>();
    for (Pair<String, Parameter> pair : parameters) {
      values.put(pair.getKey(), pair.getValue().getArray().toFloatArray());
    }
    ByteBuffer buffer =
        ByteBuffer.allocateDirect(Math.toIntExact(layout.bytes())).order(ByteOrder.LITTLE_ENDIAN);
    layout.header().write(buffer);
    for (WeightLayout.Placement placement : layout.placements()) {
      write(buffer, placement, require(values, placement.parameter()));
    }
    for (WeightLayout.Embedding embedding : layout.embeddings().values()) {
      fold(buffer, layout, embedding, values);
    }
    double[] projection = layer0Projection(buffer, layout);
    for (WeightLayout.Embedding embedding : layout.embeddings().values()) {
      if (embedding.foldOffset() >= 0) {
        foldLayer0(buffer, layout, embedding, projection);
      }
    }
    writeTileRelations(
        buffer,
        layout.header().weight(HipWeight.TILE_RELATION, 0),
        layout.header().elements(HipWeight.TILE_RELATION, 0),
        layout.format().tileTokens().relations());
    return new WeightPack(layout, buffer, version);
  }

  public WeightLayout layout() {
    return layout;
  }

  /** ネットワークのバージョン番号。 */
  public long version() {
    return version;
  }

  /** 中身を返す。位置 0 から {@link WeightLayout#bytes()} バイトを見る、新しく作った little-endian のビュー。 */
  public ByteBuffer buffer() {
    return buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).limit(Math.toIntExact(layout.bytes()));
  }

  /** float を BF16 に丸めた値を返す。有限でない値には例外を投げる。 */
  static float round(float value) {
    return Bfloat16.toFloat(Bfloat16.fromFloat(value));
  }

  private static float[] require(Map<String, float[]> values, String name) {
    float[] array = values.get(name);
    if (array == null) {
      throw new IllegalArgumentException("the network has no parameter " + name);
    }
    return array;
  }

  private static void write(ByteBuffer buffer, WeightLayout.Placement placement, float[] values) {
    int at = Math.toIntExact(placement.offset());
    switch (placement.conversion()) {
      case BF16 -> {
        for (int i = 0; i < values.length; i++) {
          buffer.putShort(at + i * Short.BYTES, Bfloat16.fromFloat(values[i]));
        }
      }
      case ROUNDED -> {
        for (int i = 0; i < values.length; i++) {
          buffer.putFloat(at + i * Float.BYTES, round(values[i]));
        }
      }
      case FP32 -> {
        for (int i = 0; i < values.length; i++) {
          if (!Float.isFinite(values[i])) {
            throw new IllegalArgumentException(placement.parameter() + " has a non-finite value");
          }
          buffer.putFloat(at + i * Float.BYTES, values[i]);
        }
      }
    }
  }

  /** 1 つの型の埋め込みを書く。組の表、表に折り込まない特徴の W（BF16、転置）、バイアス（BF16 に丸めた値）を書く。 */
  private static void fold(
      ByteBuffer buffer, WeightLayout layout, WeightLayout.Embedding embedding, Map<String, float[]> values) {
    EmbeddingSource source = embedding.source();
    int hidden = layout.architecture().hidden();
    int dimension = embedding.dimension();
    int inputs = source.embeddedCount() * dimension + source.denseCount();
    float[] table = require(values, embedding.table());
    float[] projection = require(values, embedding.projection());
    float[] bias = require(values, embedding.bias());
    float[] weights = new float[projection.length];
    for (int i = 0; i < weights.length; i++) {
      weights[i] = round(projection[i]);
    }
    for (WeightLayout.Group group : embedding.groups()) {
      int[] members = group.members();
      // P_m[c][h] = Σ_k W[h][e_m·d + k]·E[off_m + c][k]（FP32、k の順）
      float[][][] partial = new float[members.length][][];
      for (int m = 0; m < members.length; m++) {
        int column = members[m];
        int vocabulary = source.vocabulary(column);
        int namespace = source.namespace(column);
        partial[m] = new float[vocabulary][hidden];
        for (int c = 0; c < vocabulary; c++) {
          int entry = (namespace + c) * dimension;
          for (int h = 0; h < hidden; h++) {
            int weightRow = h * inputs + column * dimension;
            float sum = 0f;
            for (int k = 0; k < dimension; k++) {
              sum += weights[weightRow + k] * round(table[entry + k]);
            }
            partial[m][c][h] = sum;
          }
        }
      }
      int at = Math.toIntExact(group.offset());
      int[] digits = new int[members.length];
      for (int row = 0; row < group.rows(); row++) {
        int rest = row;
        for (int m = members.length - 1; m >= 0; m--) {
          digits[m] = rest / group.strides()[m];
          rest -= digits[m] * group.strides()[m];
        }
        for (int h = 0; h < hidden; h++) {
          float sum = partial[0][digits[0]][h];
          for (int m = 1; m < members.length; m++) {
            sum += partial[m][digits[m]][h];
          }
          buffer.putFloat(at + (row * hidden + h) * Float.BYTES, sum);
        }
      }
    }
    int denseAt = Math.toIntExact(embedding.denseOffset());
    int embeddedWidth = source.embeddedCount() * dimension;
    for (int k = 0; k < source.denseCount(); k++) {
      for (int h = 0; h < hidden; h++) {
        buffer.putShort(
            denseAt + (k * hidden + h) * Short.BYTES, Bfloat16.fromFloat(projection[h * inputs + embeddedWidth + k]));
      }
    }
    int biasAt = Math.toIntExact(embedding.biasOffset());
    for (int h = 0; h < hidden; h++) {
      buffer.putFloat(biasAt + h * Float.BYTES, round(bias[h]));
    }
  }

  /**
   * 層 0 の M = diag(g)·W_qkvᵀ を FP64 の {@code [H, 3A]} で返す。列はヘッドの対の並び。g は FP32、W_qkv は書いた束の BF16 の値を使う。
   */
  private static double[] layer0Projection(ByteBuffer buffer, WeightLayout layout) {
    int hidden = layout.architecture().hidden();
    int attention = layout.architecture().attention();
    int width = 3 * attention;
    int gain = Math.toIntExact(layout.header().weight(HipWeight.ATTENTION_NORM, 0));
    int qkv = Math.toIntExact(layout.header().weight(HipWeight.QKV, 0));
    double[] projection = new double[hidden * width];
    for (int c = 0; c < width; c++) {
      int column = HipEmbedding.foldColumn(c, attention);
      for (int h = 0; h < hidden; h++) {
        float weight = Float.intBitsToFloat((buffer.getShort(qkv + (c * hidden + h) * Short.BYTES) & 0xFFFF) << 16);
        projection[h * width + column] = (double) buffer.getFloat(gain + h * Float.BYTES) * weight;
      }
    }
    return projection;
  }

  /**
   * 1 つの型について、層 0 の QKV まで折り込んだ表を書く。書いた束の T・W_d・b に FP64 で M を掛け、FP32 に丸めた T'・W'_d・b' を置く。
   * その後ろに b_qkv を写す。
   */
  private static void foldLayer0(
      ByteBuffer buffer, WeightLayout layout, WeightLayout.Embedding embedding, double[] projection) {
    int hidden = layout.architecture().hidden();
    int attention = layout.architecture().attention();
    int width = 3 * attention;
    double[] source = new double[hidden];
    double[] projected = new double[width];
    for (WeightLayout.Group group : embedding.groups()) {
      int from = Math.toIntExact(group.offset());
      int to = Math.toIntExact(group.foldOffset());
      for (int row = 0; row < group.rows(); row++) {
        for (int h = 0; h < hidden; h++) {
          source[h] = buffer.getFloat(from + (row * hidden + h) * Float.BYTES);
        }
        project(source, projection, projected);
        for (int c = 0; c < width; c++) {
          buffer.putFloat(to + (row * width + c) * Float.BYTES, (float) projected[c]);
        }
      }
    }
    int dense = embedding.source().denseCount();
    int denseAt = Math.toIntExact(embedding.denseOffset());
    int to = Math.toIntExact(embedding.foldOffset());
    for (int k = 0; k <= dense; k++) {
      for (int h = 0; h < hidden; h++) {
        source[h] =
            k < dense
                ? Float.intBitsToFloat((buffer.getShort(denseAt + (k * hidden + h) * Short.BYTES) & 0xFFFF) << 16)
                : buffer.getFloat(Math.toIntExact(embedding.biasOffset()) + h * Float.BYTES);
      }
      project(source, projection, projected);
      for (int c = 0; c < width; c++) {
        buffer.putFloat(to + (k * width + c) * Float.BYTES, (float) projected[c]);
      }
    }
    int qkvBias = Math.toIntExact(layout.header().weight(HipWeight.QKV_BIAS, 0));
    for (int c = 0; c < width; c++) {
      buffer.putFloat(
          to + ((dense + 1) * width + HipEmbedding.foldColumn(c, attention)) * Float.BYTES,
          buffer.getFloat(qkvBias + c * Float.BYTES));
    }
  }

  /** out = v·M（M は {@code [v の長さ, out の長さ]}、FP64）。 */
  private static void project(double[] v, double[] m, double[] out) {
    int width = out.length;
    Arrays.fill(out, 0);
    for (int h = 0; h < v.length; h++) {
      double x = v[h];
      if (x == 0) {
        continue;
      }
      int at = h * width;
      for (int c = 0; c < width; c++) {
        out[c] += x * m[at + c];
      }
    }
  }

  /** 牌の関係の種類を書く。値は MahjongTileRelation の ordinal で、[牌, 牌] の UINT8 で置く。 */
  private static void writeTileRelations(ByteBuffer buffer, long offset, long elements, int[] relations) {
    if (relations.length != elements) {
      throw new IllegalArgumentException(
          "tile relations hold " + relations.length + " classes but the architecture's tiles need " + elements);
    }
    int at = Math.toIntExact(offset);
    for (int i = 0; i < relations.length; i++) {
      buffer.put(at + i, (byte) relations[i]);
    }
  }
}
