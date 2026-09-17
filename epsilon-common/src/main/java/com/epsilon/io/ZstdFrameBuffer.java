package com.epsilon.io;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.ZstdCompressCtx;
import com.github.luben.zstd.ZstdDecompressCtx;
import com.github.luben.zstd.ZstdException;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.util.Objects;

/**
 * 直列化とZstdの圧縮・展開で使うダイレクトバッファを再利用する。
 *
 * <p>一つのスレッドから使用する。返されるストリームとバッファは次の操作まで有効で、呼び出し元はその後も保持してはならない。
 * 圧縮中はJavaヒープ上の配列をJNIのクリティカル領域で保持しない。
 */
public final class ZstdFrameBuffer implements AutoCloseable {
  private static final int INITIAL_CAPACITY = 1 << 16;

  private final int compressionLevel;
  private final BufferOutput output = new BufferOutput();
  private final DataOutputStream dataOutput = new DataOutputStream(output);
  private final BufferInput input = new BufferInput();
  private final DataInputStream dataInput = new DataInputStream(input);
  private ByteBuffer frame = ByteBuffer.allocateDirect(INITIAL_CAPACITY);
  private ZstdCompressCtx compressor;
  private ZstdDecompressCtx decompressor;

  public ZstdFrameBuffer(int compressionLevel) {
    this.compressionLevel = compressionLevel;
  }

  /** 次のフレームの直列化を開始する。 */
  public DataOutputStream resetOutput() {
    output.buffer.clear();
    return dataOutput;
  }

  /** 直列化済みの内容を圧縮し、position=0・limit=圧縮長の借用参照を返す。 */
  public ByteBuffer compress() throws IOException {
    int rawLength = output.buffer.position();
    frame = capacity(frame, Math.toIntExact(Zstd.compressBound(rawLength)));
    frame.clear();
    if (compressor == null) {
      compressor = new ZstdCompressCtx().setLevel(compressionLevel);
    }
    try {
      int length =
          compressor.compressDirectByteBuffer(
              frame, 0, frame.capacity(), output.buffer, 0, rawLength);
      frame.clear().limit(length);
      return frame.asReadOnlyBuffer();
    } catch (ZstdException e) {
      throw new IOException("Failed to compress Zstd frame", e);
    }
  }

  /** 指定範囲のフレームを読み込み、展開内容を借用ストリームとして返す。 */
  public DataInputStream read(FileChannel channel, long offset, int length) throws IOException {
    frame = capacity(frame, length);
    frame.clear().limit(length);
    while (frame.hasRemaining()) {
      if (channel.read(frame, offset + frame.position()) < 0) {
        throw new EOFException("Truncated Zstd frame");
      }
    }
    long rawLength = Zstd.getDirectByteBufferFrameContentSize(frame, 0, length);
    if (rawLength <= 0L || rawLength > Integer.MAX_VALUE) {
      throw new IOException("Invalid Zstd frame content size: " + rawLength);
    }
    input.buffer = capacity(input.buffer, (int) rawLength);
    input.buffer.clear();
    if (decompressor == null) {
      decompressor = new ZstdDecompressCtx();
    }
    try {
      int actualLength =
          decompressor.decompressDirectByteBuffer(
              input.buffer, 0, input.buffer.capacity(), frame, 0, length);
      if (actualLength != rawLength) {
        throw new IOException(
            "Zstd frame content size changed: expected=" + rawLength + " actual=" + actualLength);
      }
      input.buffer.clear().limit(actualLength);
      return dataInput;
    } catch (ZstdException e) {
      throw new IOException("Failed to decompress Zstd frame", e);
    }
  }

  @Override
  public void close() {
    if (compressor != null) {
      compressor.close();
      compressor = null;
    }
    if (decompressor != null) {
      decompressor.close();
      decompressor = null;
    }
  }

  private static ByteBuffer capacity(ByteBuffer buffer, int required) {
    return buffer.capacity() >= required
        ? buffer
        : ByteBuffer.allocateDirect(Math.max(required, buffer.capacity() * 2));
  }

  private static final class BufferOutput extends OutputStream {
    private ByteBuffer buffer = ByteBuffer.allocateDirect(INITIAL_CAPACITY);

    @Override
    public void write(int value) {
      prepare(1);
      buffer.put((byte) value);
    }

    @Override
    public void write(byte[] bytes, int offset, int length) {
      prepare(length);
      buffer.put(bytes, offset, length);
    }

    private void prepare(int length) {
      int required = Math.addExact(buffer.position(), length);
      if (required > buffer.capacity()) {
        ByteBuffer next = capacity(buffer, required);
        buffer.flip();
        next.put(buffer);
        buffer = next;
      }
    }
  }

  private static final class BufferInput extends InputStream {
    private ByteBuffer buffer = ByteBuffer.allocateDirect(INITIAL_CAPACITY);

    @Override
    public int read() {
      return buffer.hasRemaining() ? Byte.toUnsignedInt(buffer.get()) : -1;
    }

    @Override
    public int read(byte[] bytes, int offset, int length) {
      Objects.checkFromIndexSize(offset, length, bytes.length);
      if (length == 0) {
        return 0;
      }
      if (!buffer.hasRemaining()) {
        return -1;
      }
      int count = Math.min(length, buffer.remaining());
      buffer.get(bytes, offset, count);
      return count;
    }

    @Override
    public long skip(long count) {
      int skipped = (int) Math.min(Math.max(0L, count), buffer.remaining());
      buffer.position(buffer.position() + skipped);
      return skipped;
    }

    @Override
    public int available() {
      return buffer.remaining();
    }
  }
}
