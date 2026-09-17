package com.epsilon.io;

import com.github.luben.zstd.Zstd;
import com.github.luben.zstd.ZstdCompressCtx;
import java.io.ByteArrayOutputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/** バッファの拡張と再利用後もZstdの圧縮・展開結果が一致し、破損した圧縮データを拒否することを検証する。 */
public class ZstdFrameBufferTest {
  @DataProvider
  public Object[][] compressionLevels() {
    return new Object[][] {{0}, {1}, {3}};
  }

  @Test(dataProvider = "compressionLevels")
  public void directFramesMatchExistingHeapCompressionAfterGrowthAndReuse(int level)
      throws Exception {
    byte[] bulk = new byte[190_003];
    new Random(17L).nextBytes(bulk);
    try (var frame = new ZstdFrameBuffer(level);
        var reference = new ZstdCompressCtx().setLevel(level)) {
      for (int length : new int[] {bulk.length, 19, 80_001}) {
        ByteArrayOutputStream raw = new ByteArrayOutputStream();
        try (var expected = new DataOutputStream(raw)) {
          writePayload(expected, bulk, length);
        }
        writePayload(frame.resetOutput(), bulk, length);
        ByteBuffer compressed = frame.compress();
        Assert.assertTrue(compressed.isDirect());
        Assert.assertTrue(compressed.isReadOnly());
        Assert.assertEquals(compressed.position(), 0);
        byte[] actual = new byte[compressed.remaining()];
        compressed.get(actual);
        Assert.assertEquals(actual, reference.compress(raw.toByteArray()));
      }
    }
  }

  @Test
  public void readsExistingFramesAtOffsetsAndReusesStreamAfterEof() throws Exception {
    byte[] first = new byte[150_007];
    new Random(31L).nextBytes(first);
    byte[] second = {0, 127, (byte) 128, (byte) 255};
    byte[] firstFrame = Zstd.compress(first, 1);
    byte[] secondFrame = Zstd.compress(second, 1);
    Path path = Files.createTempFile("zstd-frame-", ".bin");
    try (FileChannel channel =
            FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE);
        var buffer = new ZstdFrameBuffer(1)) {
      channel.write(ByteBuffer.wrap(new byte[7]));
      channel.write(ByteBuffer.wrap(firstFrame));
      channel.write(ByteBuffer.wrap(secondFrame));
      try (var in = buffer.read(channel, 7L, firstFrame.length)) {
        byte[] decoded = new byte[first.length];
        in.readFully(decoded);
        Assert.assertEquals(decoded, first);
        Assert.assertEquals(in.read(), -1);
        Assert.assertEquals(in.read(decoded, 0, 0), 0);
      }
      try (var in = buffer.read(channel, 7L + firstFrame.length, secondFrame.length)) {
        Assert.assertEquals(in.available(), 4);
        Assert.assertEquals(in.skip(-1L), 0L);
        Assert.assertEquals(in.readUnsignedByte(), 0);
        Assert.assertEquals(in.skip(1L), 1L);
        Assert.assertEquals(in.readUnsignedByte(), 128);
        Assert.assertEquals(in.readUnsignedByte(), 255);
        Assert.assertEquals(in.read(), -1);
      }
      try (var in = buffer.read(channel, 7L, firstFrame.length)) {
        byte[] decoded = new byte[first.length];
        in.readFully(decoded);
        Assert.assertEquals(decoded, first);
      }
    } finally {
      Files.delete(path);
    }
  }

  @Test
  public void rejectsTruncatedCorruptAndUnknownSizeFrames() throws Exception {
    byte[] payload = new byte[4097];
    new Random(53L).nextBytes(payload);
    byte[] compressed = Zstd.compress(payload, 1);
    Path path = Files.createTempFile("zstd-invalid-", ".bin");
    try (FileChannel channel =
            FileChannel.open(path, StandardOpenOption.READ, StandardOpenOption.WRITE);
        var buffer = new ZstdFrameBuffer(1);
        var unknownSize = new ZstdCompressCtx().setContentSize(false)) {
      channel.write(ByteBuffer.wrap(compressed));
      Assert.expectThrows(
          EOFException.class, () -> buffer.read(channel, 0L, compressed.length + 1));
      Assert.expectThrows(IOException.class, () -> buffer.read(channel, 0L, compressed.length - 1));
      channel.write(ByteBuffer.wrap(new byte[] {0, 0, 0, 0}), 0L);
      Assert.expectThrows(IOException.class, () -> buffer.read(channel, 0L, compressed.length));
      byte[] withoutSize = unknownSize.compress(payload);
      channel.write(ByteBuffer.wrap(withoutSize), 0L);
      Assert.expectThrows(IOException.class, () -> buffer.read(channel, 0L, withoutSize.length));
    } finally {
      Files.delete(path);
    }
  }

  private static void writePayload(DataOutputStream out, byte[] bytes, int length)
      throws IOException {
    out.writeInt(length);
    out.writeLong(0x123456789abcdef0L);
    out.writeDouble(-13.625);
    out.writeBoolean(true);
    out.writeUTF("牌譜・direct buffer");
    out.write(bytes, 0, length);
    out.writeByte(255);
    out.flush();
  }
}
