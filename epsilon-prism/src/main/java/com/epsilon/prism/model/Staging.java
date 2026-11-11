package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtStream;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.prism.input.InputBlob.Element;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * ホストで並べた領域を、要素の型ごとに 1 本の配列にまとめてデバイスへ送る（各領域はその配列の切り出しになる）。1 回の順伝播の
 * インデックス（{@link BatchIndex}）、損失の入力（{@link LossInputs}）、推論の出力の中で各手が入る位置、前もって並べる学習の
 * マイクロバッチ（{@link StagedBatch}）が使う。送る値は、DJL がホストの配列から作る配列と同じバイト列になる（真偽は 1 と 0）。
 *
 * <p>送り方は 2 つある。どちらも、ページ固定のバッファ（{@link PinnedUploads}）の中の、型ごとのバッファに並べる。
 *
 * <ul>
 *   <li>{@link #send}: バッファに並べ、型ごとに同期で写す（写し終えてから返る）。
 *   <li>{@link #write} と {@link #enqueue}: 前もって（別のスレッドで）バッファに並べておき、後でレーンの写し用のストリームに、型ごとに
 *       1 回の非同期の写しを積む。{@link PinnedUploads} は写しが終わるまで、バッファを書き換えも解放もしない。切り出しを読むストリームは、
 *       読む前に写しの終わりを待つ（{@link PinnedUploads.Transfer#waitOnStream}）。
 * </ul>
 */
public final class Staging {

  /**
   * 領域の先頭をそろえる境界（バイト）。PyTorch のデバイスの caching allocator はメモリを 512 バイトの倍数で切り出すので、領域の組ごとに
   * 別の配列で送ると、組の先頭は 512 バイトにそろう。{@link #align} の後の組は、1 本の配列の中でも、各領域の先頭のアドレスを 512 で割った
   * 余りが別の配列のときと同じになる。そのため、先頭のそろい方で決まるカーネルのベクトル化の幅と、和を取る順が変わらない。
   */
  static final int ALIGNMENT = 512;

  /** 領域の要素の型。 */
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
    /** 符号なし 8 ビット（領域はバイト列のまま載せる）。 */
    BYTE(DataType.UINT8),
    /** bfloat16（領域はバイト列のまま載せる）。 */
    BFLOAT16(DataType.BFLOAT16);

    private final DataType type;

    Kind(DataType type) {
      this.type = type;
    }

    DataType type() {
      return type;
    }

    /** ネットワークの入力の要素の型に当たる型を返す。 */
    static Kind of(Element element) {
      return switch (element) {
        case U8 -> BYTE;
        case BF16 -> BFLOAT16;
        case FP32 -> FLOAT;
        case I32 -> INT;
      };
    }

    /** 領域の値（Java の配列か、ネイティブのバイト順のバイト列）を、バッファの {@code offset} 番目の要素の位置から書く。 */
    final void write(ByteBuffer buffer, int offset, Object values) {
      if (values instanceof ByteBuffer bytes) {
        buffer.put(offset * type.getNumOfBytes(), bytes, bytes.position(), bytes.remaining());
      } else {
        writeArray(buffer, offset, values);
      }
    }

    /** この型の Java の配列を、バッファの {@code offset} 番目の要素の位置から書く。 */
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

  /** 送る領域。{@link #send} か {@link #enqueue} が型ごとの 1 本の配列からの切り出しを付け、その後で {@link #array} で読む。 */
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
     * @throws IllegalStateException まだ送っていないとき
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

  /** 領域を書いたバッファ（{@link #write} から、{@link #enqueue} で写しを積むまでの間だけ持つ。それ以外は null）。 */
  private PinnedUploads written;

  /**
   * INT64 の領域を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 領域
   * @throws IllegalStateException 空のとき、または形が値の数と合わないとき
   */
  public Section add(long[] values, Shape shape) {
    return add(Kind.LONG, values, values.length, shape);
  }

  /**
   * INT32 の領域を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 領域
   * @throws IllegalStateException 空のとき、または形が値の数と合わないとき
   */
  public Section add(int[] values, Shape shape) {
    return add(Kind.INT, values, values.length, shape);
  }

  /**
   * FP32 の領域を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 領域
   * @throws IllegalStateException 空のとき、または形が値の数と合わないとき
   */
  public Section add(float[] values, Shape shape) {
    return add(Kind.FLOAT, values, values.length, shape);
  }

  /**
   * BOOLEAN の領域を載せる。
   *
   * @param values 値（送るまで変えない）
   * @param shape 送った配列の形
   * @return 領域
   * @throws IllegalStateException 空のとき、または形が値の数と合わないとき
   */
  public Section add(boolean[] values, Shape shape) {
    return add(Kind.BOOLEAN, values, values.length, shape);
  }

  /**
   * ホストのバイト列の領域を載せる（転送用のエンコードのまま、ネイティブのバイト順）。
   *
   * @param bytes 値（今の位置から残りの全てを読む。送るまで変えない）
   * @param kind 要素の型
   * @param shape 送った配列の形
   * @return 領域
   * @throws IllegalStateException 空のとき、バイト数が要素の大きさで割り切れないとき、または形がバイト数と合わないとき
   */
  Section add(ByteBuffer bytes, Kind kind, Shape shape) {
    int element = kind.type().getNumOfBytes();
    if (bytes.remaining() % element != 0) {
      throw new IllegalStateException(bytes.remaining() + " bytes are not " + kind + " elements");
    }
    return add(kind, bytes.slice(), bytes.remaining() / element, shape);
  }

  /** 整数の列を INT64 の領域にして載せる（空なら null を返す）。 */
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

  /** 次に載せる領域の先頭を、型ごとに {@link #ALIGNMENT} バイトの境界にそろえる（領域の組の区切りに使う）。 */
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
   * 型ごとに領域をバッファへ並べて 1 本の配列で送り（同期）、各領域に切り出しを付ける。
   *
   * @param uploads ページ固定のバッファ（呼び出し元のスレッドのもの）
   * @param manager 配列を作るマネージャー
   */
  public void send(PinnedUploads uploads, NDManager manager) {
    NDArray[] arrays = new NDArray[KINDS.length];
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size > 0) {
        arrays[kind.ordinal()] =
            uploads.send(manager, kind.type(), new Shape(size), buffer -> writeAll(kind, buffer));
      }
    }
    slice(arrays);
  }

  /**
   * 型ごとに領域をバッファへ並べる（ホストのスレッドで呼ぶ）。{@link PinnedUploads} は、前の写しが終わるのを待ってから書かせる。写しは
   * {@link #enqueue} が積む。
   *
   * @param region ページ固定のバッファ（写しを積むまで、この組だけが使う）
   */
  void write(PinnedUploads region) {
    for (Kind kind : KINDS) {
      int size = sizes[kind.ordinal()];
      if (size > 0) {
        writeAll(kind, region.write(kind.type(), size));
      }
    }
    written = region;
  }

  /**
   * バッファに書いてあり、まだ写しを積んでいないかを返す。
   *
   * @return {@link #write} の後、{@link #enqueue} の前なら true
   */
  boolean written() {
    return written != null;
  }

  /**
   * 書いたバッファから、型ごとに 1 回の非同期の写しをストリームに積み、各領域に切り出しを付ける（CPU の配列は同期で写す）。
   * {@link PinnedUploads} は写しのチケットを預かり、書き直す前と閉じる前に写しの終わりを待つ。切り出しを読むほかのストリームは、読む前に、
   * 返した写しの {@link PinnedUploads.Transfer#waitOnStream} を自分の上で呼ぶ。積んだ後に送り直すときは、{@link #write} からやり直す。
   *
   * @param manager 写し先の配列を作るマネージャー
   * @param stream 写しを積むストリーム（CPU のマネージャーなら null）
   * @return 積んだ写し
   * @throws IllegalStateException バッファにまだ書いていないとき
   */
  PinnedUploads.Transfer enqueue(NDManager manager, PtStream stream) {
    if (written == null) {
      throw new IllegalStateException("the staging has not been written to pinned uploads");
    }
    PinnedUploads region = written;
    written = null;
    PinnedUploads.Transfer transfer = region.enqueue(manager, stream);
    NDArray[] arrays = new NDArray[KINDS.length];
    for (Kind kind : KINDS) {
      if (sizes[kind.ordinal()] > 0) {
        arrays[kind.ordinal()] = transfer.array(kind.type());
      }
    }
    slice(arrays);
    return transfer;
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
