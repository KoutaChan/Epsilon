package com.epsilon.io;

import java.io.IOException;
import java.io.RandomAccessFile;
import java.nio.file.Path;

/** 隠れ層の幅を保存していなかったBelief v4のパラメーターから、先頭2つの埋め込み層を読んで幅を復元する。 */
public final class LegacyBeliefParameters {
  private LegacyBeliefParameters() {}

  /** DJLのテンソルを生成せず、牌種の埋め込み行列の形状から隠れ層の幅を復元する。 */
  public static int readHidden(Path parameters) throws IOException {
    try (var input = new RandomAccessFile(parameters.toFile(), "r")) {
      require(input.readInt() == 0x444a4c40 && input.readInt() == 1, "DJL model header");
      require(input.readUTF().equals("belief"), "Belief model name");
      floatingBytes(input.readUTF());
      int inputs = count(input.readInt(), 64, "model input count");
      for (int i = 0; i < inputs; i++) {
        input.readUTF();
        shape(input);
      }
      int properties = count(input.readInt(), 256, "model property count");
      for (int i = 0; i < properties; i++) {
        input.readUTF();
        input.readUTF();
      }
      block(input); // EpsilonBeliefNetwork
      block(input); // EpsilonMahjongStateEncoder
      block(input); // stateFeatureEmbedding
      long[] features = embedding(input);
      require(
          features.length == 2 && features[0] > 35 && features[1] == 8,
          "stateFeatureEmbedding shape");
      block(input); // 牌種の埋め込み行列
      long[] tiles = embedding(input);
      require(
          tiles.length == 2
              && tiles[0] == 35
              && tiles[1] > 0
              && tiles[1] <= Integer.MAX_VALUE
              && tiles[1] % 4 == 0,
          "canonicalTileEmbedding shape");
      return (int) tiles[1];
    } catch (ArithmeticException failure) {
      throw new IOException("Invalid legacy Belief tensor length", failure);
    }
  }

  private static void block(RandomAccessFile input) throws IOException {
    require(input.readUnsignedByte() == 1, "block version");
    int inputs = count(input.readInt(), 64, "block input count");
    for (int i = 0; i < inputs; i++) shape(input);
  }

  private static long[] embedding(RandomAccessFile input) throws IOException {
    require(
        input.readChar() == 'P' && input.readUnsignedByte() == 1, "initialized parameter header");
    require(input.readUTF().equals("embedding"), "embedding parameter name");
    require(input.readUTF().equals("NDAR"), "NDArray header");
    int version = input.readInt();
    require(version >= 1 && version <= 3, "NDArray version");
    if (version > 1) {
      int named = input.readUnsignedByte();
      require(named == 0 || named == 1, "NDArray name flag");
      if (named == 1) input.readUTF();
    }
    require(input.readUTF().equals("DENSE"), "embedding sparse format");
    int bytes = floatingBytes(input.readUTF());
    long[] dimensions = shape(input);
    long elements = 1;
    for (long dimension : dimensions) {
      require(dimension > 0, "embedding dimension");
      elements = Math.multiplyExact(elements, dimension);
    }
    if (version > 2) {
      int order = input.readUnsignedByte();
      require(order == '<' || order == '>', "NDArray byte order");
    }
    int length = input.readInt();
    require(length >= 0 && length == Math.multiplyExact(elements, bytes), "tensor byte count");
    long end = Math.addExact(input.getFilePointer(), length);
    require(end <= input.length(), "complete embedding payload");
    input.seek(end);
    return dimensions;
  }

  private static long[] shape(RandomAccessFile input) throws IOException {
    int rank = count(input.readInt(), 8, "tensor rank");
    long[] dimensions = new long[rank];
    for (int i = 0; i < rank; i++) {
      dimensions[i] = input.readLong();
      require(dimensions[i] >= -1, "shape dimension");
    }
    int layout = count(input.readInt(), rank, "shape layout count");
    for (int i = 0; i < layout; i++) input.readChar();
    return dimensions;
  }

  private static int floatingBytes(String type) throws IOException {
    return switch (type) {
      case "FLOAT16", "BFLOAT16" -> 2;
      case "FLOAT32" -> 4;
      case "FLOAT64" -> 8;
      default -> throw new IOException("Invalid legacy Belief floating data type: " + type);
    };
  }

  private static int count(int count, int maximum, String label) throws IOException {
    require(count >= 0 && count <= maximum, label);
    return count;
  }

  private static void require(boolean valid, String label) throws IOException {
    if (!valid) throw new IOException("Invalid legacy Belief " + label);
  }
}
