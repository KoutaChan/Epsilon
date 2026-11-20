package com.epsilon.prism.backend;

import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.IntBuffer;
import java.util.Arrays;

/** スロット番号の列を直接バッファから貸す。分岐点より後の番号だけを各経路が所有する。 */
final class EventSlotPool {
  private static final int SLAB_WORDS = 1 << 15;
  private final int capacity;
  private final Bin[] bins = new Bin[Integer.SIZE];
  private final int stackWidth;
  private final int stackShift;
  private IntBuffer[] stack = new IntBuffer[4];
  private int returned, fresh;

  EventSlotPool(int capacity) {
    this.capacity = capacity;
    stackWidth = Math.min(SLAB_WORDS, Integer.highestOneBit(capacity));
    stackShift = Integer.numberOfTrailingZeros(stackWidth);
  }

  long available() {
    return (long) returned + capacity - fresh;
  }

  /** 同じ長さの列をスラブから切り出す。返した列の先頭の整数を空きリストの次の番号に使う。 */
  static final class Bin {
    final int width;
    private final int shift, mask;
    private IntBuffer[] slabs = new IntBuffer[4];
    private int fresh, free = -1;

    Bin(int width, int capacity) {
      this.width = width;
      int blocks = Math.max(1, Integer.highestOneBit(Math.min(capacity, SLAB_WORDS) / width));
      shift = Integer.numberOfTrailingZeros(blocks);
      mask = blocks - 1;
    }

    int take() {
      if (free >= 0) {
        int index = free;
        free = slab(index).get(offset(index));
        return index;
      }
      int index = fresh++;
      int chunk = index >>> shift;
      if (chunk == slabs.length) slabs = Arrays.copyOf(slabs, chunk * 2);
      if (slabs[chunk] == null) slabs[chunk] = allocate((mask + 1) * width);
      return index;
    }

    IntBuffer slab(int index) {
      return slabs[index >>> shift];
    }

    int offset(int index) {
      return (index & mask) * width;
    }

    void give(int index) {
      slab(index).put(offset(index), free);
      free = index;
    }
  }

  Bin bin(int length) {
    int power = Integer.SIZE - Integer.numberOfLeadingZeros(length - 1);
    Bin bin = bins[power];
    if (bin == null) bins[power] = bin = new Bin(1 << power, capacity);
    return bin;
  }

  private static IntBuffer allocate(int words) {
    return ByteBuffer.allocateDirect(Math.multiplyExact(words, Integer.BYTES))
        .order(ByteOrder.nativeOrder())
        .asIntBuffer();
  }

  /** 返された番号と、まだ貸したことのない範囲から追記する。 */
  void append(IntBuffer slots, int at, int count) {
    while (returned > 0 && count > 0) {
      int chunk = (returned - 1) >>> stackShift;
      int end = ((returned - 1) & (stackWidth - 1)) + 1;
      int n = Math.min(end, count);
      slots.put(at, stack[chunk], end - n, n);
      returned -= n;
      count -= n;
      at += n;
    }
    for (int end = at + count; at < end; at++) slots.put(at, fresh++);
  }

  void give(IntBuffer slots, int at, int count) {
    while (count > 0) {
      int chunk = returned >>> stackShift, offset = returned & (stackWidth - 1);
      if (chunk == stack.length) stack = Arrays.copyOf(stack, chunk * 2);
      if (stack[chunk] == null) stack[chunk] = allocate(stackWidth);
      int n = Math.min(count, stackWidth - offset);
      stack[chunk].put(offset, slots, at, n);
      returned += n;
      count -= n;
      at += n;
    }
  }

  void clear() {
    Arrays.fill(bins, null);
    Arrays.fill(stack, null);
    fresh = returned = 0;
  }
}
