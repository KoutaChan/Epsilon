package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.prism.input.InputSchema.Element;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * ホストで並べた区画を、型ごとに 1 本の配列でデバイスへ送る（各区画はその切り出し）。1 回の順伝播の索引（{@link BatchIndex}）・損失の入力
 * （{@link LossInputs}）・推論の手の枠の位置・前もって並べる学習の微小バッチ（{@link StagedBatch}）が使う。送る値は DJL がホストの配列から
 * 作る配列と同じバイト（真偽は 1 と 0）。
 *
 * <p>送り方は 2 つ。
 *
 * <ul>
 *   <li>{@link #send}: {@link Uploads} のページ固定のバッファに並べ、型ごとに同期で写す（写し終えてから返る）。
 *   <li>{@link #write} と {@link #enqueue}: 前もって（別のスレッドで）{@link UploadRegion} の型ごとのページ固定のバッファに並べておき、
 *       後でレーンのストリームに型ごとに 1 回の非同期の写しを積む。写しの後に記録した event が終わるまで、領域を書き換えも解放もしない
 *       （{@link UploadRegion#await}）。
 * </ul>
 */
public final class Staging {

  /**
   * 区画の先頭をそろえる境目（バイト）。PyTorch の装置の caching allocator は塊を 512 バイトの倍数で切るので、区画の組ごとに別の配列で
   * 送ると組の先頭は 512 バイトにそろう。{@link #align} の後の組は 1 本の配列の中でも各区画の先頭の 512 の剰余が別の配列のときと同じ
   * になり、先頭の揃いで決まる核のベクトル化の幅と和の順が変わらない。
   */
  static final int ALIGNMENT = 512;

  /** 区画の型。 */
  enum Kind {
    LONG(DataType.INT64) {
      @Override
      void writeArray(ByteBuffer buffer, int offset, Object values) {
        buffer.asLongBuffer().put(offset, (long[]) values);
      }
    },
    INT(DataType.INT32) {
      @Override
      void writeArray(ByteBuffer buffer, int offset, Object values) {
        buffer.asIntBuffer().put(offset, (int[]) values);
      }
    },
    FLOAT(DataType.FLOAT32) {
      @Override
      void writeArray(ByteBuffer buffer, int offset, Object values) {
        buffer.asFloatBuffer().put(offset, (float[]) values);
      }
    },
    BOOLEAN(DataType.BOOLEAN) {
      @Override
      void writeArray(ByteBuffer buffer, int offset, Object values) {
        boolean[] flags = (boolean[]) values;
        for (int i = 0; i < flags.length; i++) {
          buffer.put(offset + i, (byte) (flags[i] ? 1 : 0));
        }
      }
    },
    /** 符号なし 8 ビット（区画はバイト列のまま載せる）。 */
    BYTE(DataType.UINT8),
    /** bfloat16（区画はバイト列のまま載せる）。 */
    BFLOAT16(DataType.BFLOAT16);

    private final DataType type;

    Kind(DataType type) {
      this.type = type;
    }

    DataType type() {
      return type;
    }

    /** 網の入力の区画の要素の型。 */
    static Kind of(Element element) {
      return switch (element) {
        case U8 -> BYTE;
        case BF16 -> BFLOAT16;
        case FP32 -> FLOAT;
        case I32 -> INT;
      };
    }

    /** 区画の値（Java の配列か、ネイティブのバイト順のバイト列）を、バッファの要素の位置 offset から書く。 */
    final void write(ByteBuffer buffer, int offset, Object values) {
      if (values instanceof ByteBuffer bytes) {
        buffer.put(offset * type.getNumOfBytes(), bytes, bytes.position(), bytes.remaining());
      } else {
        writeArray(buffer, offset, values);
      }
    }

    /** この型の Java の配列 values を、バッファの要素の位置 offset から書く。 */
    void writeArray(ByteBuffer buffer, int offset, Object values) {
      throw new IllegalArgumentException(this + " sections are staged as bytes");
    }

    /** 要素の数を {@link #ALIGNMENT} バイトの倍数に切り上げる。 */
    int aligned(int elements) {
      int step = ALIGNMENT / type.getNumOfBytes();
      return Math.ceilDiv(elements, step) * step;
    }
  }

  private static final Kind[] KINDS = Kind.values();

  /** 送る区画。{@link #send} か {@link #enqueue} が型ごとの 1 本の配列の切り出しを付け、それから {@link #array} で読む。 */
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

  /** 区画を書いた領域（{@link #write} から {@link #enqueue} で写しを積むまで。それ以外は null）。 */
  private UploadRegion written;

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

  /**
   * ホストのバイト列の区画を載せる（転送の符号のまま、ネイティブのバイト順）。
   *
   * @param bytes 位置から残りの全てを読む（送るまで変えない）
   * @param kind 型
   * @param shape 送った配列の形
   * @return 区画
   * @throws IllegalStateException 空か、形がバイト数と合わないとき
   */
  Section add(ByteBuffer bytes, Kind kind, Shape shape) {
    int element = kind.type().getNumOfBytes();
    if (bytes.remaining() % element != 0) {
      throw new IllegalStateException(bytes.remaining() + " bytes are not " + kind + " elements");
    }
    return add(kind, bytes.slice(), bytes.remaining() / element, shape);
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

  /** 次に載せる区画の先頭を、型ごとに {@link #ALIGNMENT} バイトの境目にする（区画の組の区切り）。 */
  void align() {
    for (Kind kind : KINDS) {
      sizes[kind.ordinal()] = kind.aligned(sizes[kind.ordinal()]);
    }
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
   * 型ごとに区画を置き場のバッファへ並べて 1 本の配列で送り（同期）、区画に切り出しを付ける。
   *
   * @param uploads 置き場（呼び手のスレッドの物）
   * @param manager 配列を作る管理元
   */
  public void send(Uploads uploads, NDManager manager) {
    NDArray[] arrays = new NDArray[KINDS.length];
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size > 0) {
        arrays[kind.ordinal()] =
            uploads.send(manager, kind, size, buffer -> writeAll(kind, buffer));
      }
    }
    slice(arrays);
  }

  /**
   * 領域の前の写しの終わりを待ってから、型ごとに区画を領域のバッファへ並べる（ホストのスレッド。写しは {@link #enqueue} が積む）。
   *
   * @param region 領域（写しを積むまでこの組だけが使う）
   */
  void write(UploadRegion region) {
    region.await();
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size > 0) {
        writeAll(kind, region.buffer(kind, size));
      }
    }
    written = region;
  }

  /**
   * 領域に書いてあって写しをまだ積んでいないかを返す。
   *
   * @return {@link #write} の後、{@link #enqueue} の前なら true
   */
  boolean written() {
    return written != null;
  }

  /**
   * 書いた領域から、型ごとに 1 回の非同期の写しを呼び手のスレッドの今のストリームに積み、写しの後に領域の event を記録して、区画に
   * 切り出しを付ける（CPU の配列は同期で写す）。積んだ後に送り直すときは {@link #write} からやり直す。
   *
   * @param manager 配列を作る管理元
   * @param stream 呼び手のスレッドの今のストリーム（CPU の管理元なら null）
   * @throws IllegalStateException 領域に書いていないとき
   */
  void enqueue(NDManager manager, PtStream stream) {
    if (written == null) {
      throw new IllegalStateException("the staging has not been written to a region");
    }
    UploadRegion region = written;
    written = null;
    NDArray[] arrays = new NDArray[KINDS.length];
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size > 0) {
        PtNDArray array = (PtNDArray) manager.create(new Shape(size), kind.type());
        array.enqueueCopyFrom(region.pinned(kind));
        arrays[kind.ordinal()] = array;
      }
    }
    region.copied(stream);
    slice(arrays);
  }

  private void writeAll(Kind kind, ByteBuffer buffer) {
    for (Section section : sections) {
      if (section.kind == kind) {
        kind.write(buffer, section.offset, section.values);
      }
    }
  }

  private void slice(NDArray[] arrays) {
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
