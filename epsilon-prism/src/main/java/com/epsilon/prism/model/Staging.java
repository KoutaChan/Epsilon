package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * ホストで並べた区画を、型ごとに 1 本の配列でデバイスへ送る（各区画はその切り出し）。1 回の順伝播の索引（{@link BatchIndex}）・損失の入力
 * （{@link LossInputs}）・推論の手の枠の位置が使う。区画は {@link Uploads} のページ固定のバッファに直に並べ、型ごとに 1 回写す。送る値は
 * DJL がホストの配列から作る配列と同じバイト（真偽は 1 と 0）。
 */
public final class Staging {

  /** 区画の型。 */
  enum Kind {
    LONG(DataType.INT64) {
      @Override
      void write(ByteBuffer buffer, int offset, Object values) {
        buffer.asLongBuffer().put(offset, (long[]) values);
      }
    },
    INT(DataType.INT32) {
      @Override
      void write(ByteBuffer buffer, int offset, Object values) {
        buffer.asIntBuffer().put(offset, (int[]) values);
      }
    },
    FLOAT(DataType.FLOAT32) {
      @Override
      void write(ByteBuffer buffer, int offset, Object values) {
        buffer.asFloatBuffer().put(offset, (float[]) values);
      }
    },
    BOOLEAN(DataType.BOOLEAN) {
      @Override
      void write(ByteBuffer buffer, int offset, Object values) {
        boolean[] flags = (boolean[]) values;
        for (int i = 0; i < flags.length; i++) {
          buffer.put(offset + i, (byte) (flags[i] ? 1 : 0));
        }
      }
    };

    private final DataType type;

    Kind(DataType type) {
      this.type = type;
    }

    DataType type() {
      return type;
    }

    /** この型の Java の配列 values を、バッファの要素の位置 offset から書く。 */
    abstract void write(ByteBuffer buffer, int offset, Object values);
  }

  private static final Kind[] KINDS = Kind.values();

  /** 送る区画。{@link #send} が型ごとの 1 本の配列の切り出しを付け、それから {@link #array} で読む。 */
  public static final class Section {

    private final Kind kind;
    private final Object values;
    private final int offset;
    private final int length;
    private final Shape shape;
    private NDArray array;

    private Section(Kind kind, Object values, int offset, int length, Shape shape) {
      this.kind = kind;
      this.values = values;
      this.offset = offset;
      this.length = length;
      this.shape = shape;
    }

    /**
     * 送った配列の切り出しを返す。
     *
     * @return 配列（形は載せたときの形）
     * @throws IllegalStateException 送る前のとき
     */
    public NDArray array() {
      if (array == null) {
        throw new IllegalStateException("the section has not been sent");
      }
      return array;
    }
  }

  private final List<Section> sections = new ArrayList<>();
  private final int[] sizes = new int[KINDS.length];

  /**
   * INT64 の区画を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 区画
   * @throws IllegalStateException 空か、形が値の数と合わないとき
   */
  public Section add(long[] values, Shape shape) {
    return add(Kind.LONG, values, values.length, shape);
  }

  /**
   * INT32 の区画を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 区画
   * @throws IllegalStateException 空か、形が値の数と合わないとき
   */
  public Section add(int[] values, Shape shape) {
    return add(Kind.INT, values, values.length, shape);
  }

  /**
   * FP32 の区画を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 区画
   * @throws IllegalStateException 空か、形が値の数と合わないとき
   */
  public Section add(float[] values, Shape shape) {
    return add(Kind.FLOAT, values, values.length, shape);
  }

  /**
   * BOOLEAN の区画を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 区画
   * @throws IllegalStateException 空か、形が値の数と合わないとき
   */
  public Section add(boolean[] values, Shape shape) {
    return add(Kind.BOOLEAN, values, values.length, shape);
  }

  /** 整数の列を INT64 の区画にする（空なら null）。 */
  Section addLongs(List<? extends Number> values) {
    if (values.isEmpty()) {
      return null;
    }
    long[] array = new long[values.size()];
    for (int i = 0; i < array.length; i++) {
      array[i] = values.get(i).longValue();
    }
    return add(array, new Shape(array.length));
  }

  private Section add(Kind kind, Object values, int length, Shape shape) {
    if (length == 0 || shape.size() != length) {
      throw new IllegalStateException("invalid staged section " + shape + " of " + length);
    }
    Section section = new Section(kind, values, sizes[kind.ordinal()], length, shape);
    sections.add(section);
    sizes[kind.ordinal()] += length;
    return section;
  }

  /**
   * 型ごとに区画を置き場のバッファへ並べて 1 本の配列で送り、区画に切り出しを付ける。
   *
   * @param uploads 置き場（呼び手のスレッドの物）
   * @param manager 配列を作る管理元
   */
  public void send(Uploads uploads, NDManager manager) {
    NDArray[] arrays = new NDArray[KINDS.length];
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size == 0) {
        continue;
      }
      arrays[kind.ordinal()] =
          uploads.send(
              manager,
              kind,
              size,
              buffer -> {
                for (Section section : sections) {
                  if (section.kind == kind) {
                    kind.write(buffer, section.offset, section.values);
                  }
                }
              });
    }
    for (Section section : sections) {
      NDArray all = arrays[section.kind.ordinal()];
      NDArray slice =
          section.length == all.size()
              ? all
              : all.get(new NDIndex("{}:{}", section.offset, section.offset + section.length));
      section.array = slice.reshape(section.shape);
    }
  }
}
