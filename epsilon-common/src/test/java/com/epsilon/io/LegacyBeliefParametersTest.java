package com.epsilon.io;

import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;
import org.testng.annotations.Test;

/** モデルを生成せずに旧形式のBeliefから隠れ層の幅を読み取り、異なる構成や途中で切れたデータを拒否することを検証する。 */
public class LegacyBeliefParametersTest {
  @Test
  public void readsStoredHiddenWithoutAllocatingModel() throws Exception {
    Path file = fixture(20, 8);
    try {
      Assert.assertEquals(LegacyBeliefParameters.readHidden(file), 20);
    } finally {
      Files.delete(file);
    }
  }

  @Test
  public void rejectsChangedPrefixTopologyAndTruncatedTensor() throws Exception {
    Path wrong = fixture(20, 7);
    Path truncated = fixture(20, 8);
    try {
      Assert.expectThrows(IOException.class, () -> LegacyBeliefParameters.readHidden(wrong));
      try (var file = new java.io.RandomAccessFile(truncated.toFile(), "rw")) {
        file.setLength(file.length() - 1);
      }
      Assert.expectThrows(IOException.class, () -> LegacyBeliefParameters.readHidden(truncated));
    } finally {
      Files.delete(wrong);
      Files.delete(truncated);
    }
  }

  private static Path fixture(int hidden, int categoryWidth) throws IOException {
    Path path = Files.createTempFile("belief-metadata-", ".params");
    try (var out = new DataOutputStream(Files.newOutputStream(path))) {
      out.writeInt(0x444a4c40);
      out.writeInt(1);
      out.writeUTF("belief");
      out.writeUTF("FLOAT32");
      out.writeInt(0);
      out.writeInt(0);
      block(out);
      block(out);
      block(out);
      embedding(out, 36, categoryWidth);
      block(out);
      embedding(out, 35, hidden);
    }
    return path;
  }

  private static void block(DataOutputStream out) throws IOException {
    out.writeByte(1);
    out.writeInt(0);
  }

  private static void embedding(DataOutputStream out, int rows, int columns) throws IOException {
    out.writeChar('P');
    out.writeByte(1);
    out.writeUTF("embedding");
    out.writeUTF("NDAR");
    out.writeInt(3);
    out.writeByte(0);
    out.writeUTF("DENSE");
    out.writeUTF("FLOAT32");
    out.writeInt(2);
    out.writeLong(rows);
    out.writeLong(columns);
    out.writeInt(0);
    out.writeByte('<');
    out.writeInt(rows * columns * 4);
    out.write(new byte[rows * columns * 4]);
  }
}
