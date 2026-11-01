package com.epsilon.prism.backend;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.pytorch.engine.PtEngine;
import ai.djl.util.Pair;
import ai.djl.util.PairList;
import com.epsilon.ai.model.MahjongTileRelation;
import com.epsilon.network.NetworkRuntime;
import com.epsilon.prism.codec.Bfloat16;
import com.epsilon.prism.hip.HipWeight;
import com.epsilon.prism.model.PrismNetwork;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;

/**
 * HIP のカーネルが読む推論の重みの束。公開のたびに網の FP32 の正本から作り、網の版の番号を持つ（forward で照らし、古い写しが
 * 残る穴を作りから塞ぐ）。中身は {@link WeightLayout} の並びの直の ByteBuffer（little-endian、頭つき）。
 *
 * <p>埋め込みの畳んだ組の表は、BF16 に丸めた W と E の積を FP32 で足したもの（欄ごとに P_m[c] = W_m·E[off_m + c] を作り、組の行は
 * Σ_m P_m[c_m] を欄の順に）。eager の Linear との差は和の順だけ。
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
   * 保存した網から作る。重みを CPU に FP32 で読み、表と束を作ってから網を捨てる（束と表は網の配列を参照しない）。
   *
   * @param model 網のディレクトリ（weights.bin と model.json）
   * @param version 網の版の番号（0 以上）
   * @return 重みの束（表は {@link #layout()}）
   * @throws IOException 読めないか、形式・指紋・重みが合わないとき
   */
  public static WeightPack load(Path model, long version) throws IOException {
    NetworkRuntime.initialize();
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      PrismNetwork network = PrismNetwork.load(model, manager);
      return build(WeightLayout.of(network), network, version);
    }
  }

  /**
   * 網の今の重みから作る。
   *
   * @param layout 網の表（同じ形・同じ名前の網から作ったもの）
   * @param network 網（推論の重みの束を読む。どのデバイスでもよい）
   * @param version 網の版の番号（0 以上）
   * @return 重みの束
   */
  public static WeightPack build(WeightLayout layout, PrismNetwork network, long version) {
    return build(layout, network.inferenceParameters(), version);
  }

  /**
   * 推論の重みの束から作る。
   *
   * @param layout 表
   * @param parameters 推論の重みの束（{@link PrismNetwork#inferenceParameters()}）
   * @param version 網の版の番号（0 以上）
   * @return 重みの束
   * @throws IllegalArgumentException 表にないパラメーターがあるか、値が有限でないとき
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
    writeTileRelations(
        buffer,
        layout.header().weight(HipWeight.TILE_RELATION, 0),
        HipWeight.TILE_RELATION.elements(layout.dimensions()));
    return new WeightPack(layout, buffer, version);
  }

  public WeightLayout layout() {
    return layout;
  }

  /** 網の版の番号。 */
  public long version() {
    return version;
  }

  /** 中身（位置 0 から {@link WeightLayout#bytes()} バイト。写しの view、little-endian）。 */
  public ByteBuffer buffer() {
    return buffer.duplicate().order(ByteOrder.LITTLE_ENDIAN).position(0).limit(Math.toIntExact(layout.bytes()));
  }

  /** float を BF16 に丸めた値（有限でなければ例外）。 */
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

  /** 1 つの型の埋め込み: 畳んだ組の表・表に畳まない特徴の W（BF16、転置）・偏り（BF16 に丸めた値）を書く。 */
  private static void fold(
      ByteBuffer buffer, WeightLayout layout, WeightLayout.Embedding embedding, Map<String, float[]> values) {
    EmbeddingSource source = embedding.source();
    int hidden = layout.shape().hidden();
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

  /** 牌の関係の類（MahjongTileRelation の ordinal、[34·34] の UINT8）。 */
  private static void writeTileRelations(ByteBuffer buffer, long offset, long elements) {
    long[] relations = MahjongTileRelation.relationIds();
    if (relations.length != elements) {
      throw new IllegalStateException(
          "tile relations hold " + relations.length + " classes but the HIP slot holds " + elements);
    }
    int at = Math.toIntExact(offset);
    for (int i = 0; i < relations.length; i++) {
      buffer.put(at + i, (byte) Math.toIntExact(relations[i]));
    }
  }
}
