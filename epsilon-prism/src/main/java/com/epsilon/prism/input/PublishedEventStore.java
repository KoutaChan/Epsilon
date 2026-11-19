package com.epsilon.prism.input;

import java.lang.ref.WeakReference;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

/** 公開した列のバイトを語彙の幅に詰めて保管する。経路が返したページは次の経路が使う。 */
public final class PublishedEventStore {
  private static final Map<InputFormat, WeakReference<PublishedEventStore>> STORES =
      new HashMap<>();

  static synchronized PublishedEventStore of(InputFormat format) {
    var reference = STORES.get(format);
    var store = reference == null ? null : reference.get();
    if (store == null) {
      store = new PublishedEventStore(format);
      STORES.put(format, new WeakReference<>(store));
    }
    return store;
  }

  /** レーンの GPU に置けるイベント数を保管の上限に加える。レーンを閉じるときに返り値を一度実行する。 */
  public static synchronized Runnable register(InputFormat format, int events) {
    PublishedEventStore store = of(format);
    store.resize(1, events);
    return () -> unregister(format, store, events);
  }

  /** 最後のレーンを閉じたら、新しい方策は新しいプールを使う。提出済みの入力は元のプールを保持する。 */
  private static synchronized void unregister(
      InputFormat format, PublishedEventStore store, int events) {
    if (store.resize(-1, -events) == 0) STORES.remove(format);
  }

  static final int ROWS = 8;
  private static final int PAGES = 4096;
  private final int[] categories, relations;
  final int scalars;
  private final int rowBytes;
  private volatile ByteBuffer[] slabs = new ByteBuffer[4];
  private int fresh, free = -1, live, lanes;
  private long capacity;

  private synchronized int resize(int laneChange, int eventChange) {
    lanes += laneChange;
    capacity += eventChange;
    return lanes;
  }

  PublishedEventStore(InputFormat format) {
    categories = widths(format.event().columnVocabularies());
    relations = widths(format.relation().columnVocabularies());
    scalars = format.width(InputBlob.Tensor.EVENT_SCALAR);
    rowBytes = bytes(categories) + bytes(relations) + Short.BYTES * scalars;
  }

  private static int[] widths(int[] vocabularies) {
    for (int i = 0; i < vocabularies.length; i++) {
      vocabularies[i] = Integer.SIZE - Integer.numberOfLeadingZeros(vocabularies[i] - 1);
    }
    return vocabularies;
  }

  private static int bytes(int[] widths) {
    int bits = 0;
    for (int width : widths) bits += width;
    return (bits + 7) / 8;
  }

  /** GPU を開く前の最初の受付と単独の CPU の符号化には、スラブ 1 枚まで貸す。 */
  synchronized boolean take(int[] pages, int from, int to) {
    long limit = lanes == 0 ? PAGES : capacity / ROWS;
    if (to - from > limit - live) return false;
    for (int i = from; i < to; i++) pages[i] = takePage();
    return true;
  }

  private int takePage() {
    int page = free;
    if (page >= 0) {
      free = slabs[page / PAGES].getInt(offset(page, 0));
    } else {
      page = fresh++;
      int index = page / PAGES;
      if (index == slabs.length) slabs = Arrays.copyOf(slabs, index * 2);
      if (slabs[index] == null) {
        slabs[index] =
            ByteBuffer.allocateDirect(PAGES * ROWS * rowBytes).order(ByteOrder.LITTLE_ENDIAN);
      }
    }
    live++;
    return page;
  }

  synchronized void give(int[] pages, int count) {
    for (int i = 0; i < count; i++) {
      int page = pages[i];
      slabs[page / PAGES].putInt(offset(page, 0), free);
      free = page;
    }
    live -= count;
    if (live == 0) {
      ByteBuffer first = lanes > 0 && capacity >= (long) PAGES * ROWS ? slabs[0] : null;
      slabs = new ByteBuffer[4];
      slabs[0] = first;
      fresh = 0;
      free = -1;
    }
  }

  private int offset(int page, int row) {
    return ((page % PAGES) * ROWS + row) * rowBytes;
  }

  private static int pack(ByteBuffer to, int at, int[] widths, byte[] source, int from) {
    int word = 0, bits = 0;
    for (int width : widths) {
      word |= Byte.toUnsignedInt(source[from++]) << bits;
      bits += width;
      if (bits >= 8) {
        to.put(at++, (byte) word);
        word >>>= 8;
        bits -= 8;
      }
    }
    if (bits > 0) to.put(at++, (byte) word);
    return at;
  }

  private static int unpack(ByteBuffer from, int at, int[] widths, ByteBuffer to, int target) {
    int word = 0, bits = 0;
    for (int width : widths) {
      if (bits < width) {
        word |= Byte.toUnsignedInt(from.get(at++)) << bits;
        bits += 8;
      }
      to.put(target++, (byte) (word & ((1 << width) - 1)));
      word >>>= width;
      bits -= width;
    }
    return at;
  }

  void store(int page, int row, int event, int fixed, byte[] cats, short[] values, byte[] rels) {
    ByteBuffer slab = slabs[page / PAGES];
    int at = pack(slab, offset(page, row), categories, cats, event * categories.length);
    at = pack(slab, at, relations, rels, (fixed + event) * relations.length);
    for (int i = 0; i < scalars; i++)
      slab.putShort(at + Short.BYTES * i, values[event * scalars + i]);
  }

  void copy(int page, int row, ByteBuffer to, int cat, int scalar, int rel) {
    ByteBuffer slab = slabs[page / PAGES];
    int at = unpack(slab, offset(page, row), categories, to, cat);
    at = unpack(slab, at, relations, to, rel);
    to.put(scalar, slab, at, scalars * Short.BYTES);
  }
}
