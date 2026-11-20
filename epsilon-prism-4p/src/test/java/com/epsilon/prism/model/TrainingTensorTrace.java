package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.types.DataType;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;
import org.testng.Assert;

/** テンソルを呼び出し順に記録し、基準を二度実行して測った変動と照合する。 */
final class TrainingTensorTrace implements AutoCloseable {
  private final DataOutputStream out;

  TrainingTensorTrace(Path path) throws IOException {
    out =
        new DataOutputStream(
            new BufferedOutputStream(
                new GZIPOutputStream(Files.newOutputStream(path, StandardOpenOption.CREATE_NEW))));
  }

  void tensor(String name, NDArray array) throws IOException {
    NDArray floats = array.toType(DataType.FLOAT32, false);
    try {
      write(out, name, array.getDataType() + array.getShape().toString(), floats.toFloatArray());
    } finally {
      if (floats != array) floats.close();
    }
  }

  static void write(DataOutputStream target, String name, String type, float[] values)
      throws IOException {
    target.writeUTF(name);
    target.writeUTF(type);
    target.writeInt(values.length);
    for (float v : values) {
      if (!Float.isFinite(v)) throw new AssertionError(name + " contains a non-finite value");
      target.writeInt(Float.floatToRawIntBits(v));
    }
  }

  @Override
  public void close() throws IOException {
    try (out) {
      out.writeUTF("");
    }
  }

  static void compare(Path first, Path repeat, Path candidate) throws IOException {
    try (var a = open(first);
        var b = open(repeat);
        var c = open(candidate)) {
      compare(a, b, c);
    }
  }

  private static DataInputStream open(Path path) throws IOException {
    return new DataInputStream(
        new BufferedInputStream(new GZIPInputStream(Files.newInputStream(path))));
  }

  static void compare(DataInputStream a, DataInputStream b, DataInputStream c) throws IOException {
    for (; ; ) {
      String name = a.readUTF();
      Assert.assertEquals(b.readUTF(), name);
      Assert.assertEquals(c.readUTF(), name);
      if (name.isEmpty()) break;
      String type = a.readUTF();
      Assert.assertEquals(b.readUTF(), type, name);
      Assert.assertEquals(c.readUTF(), type, name);
      int n = a.readInt();
      Assert.assertTrue(n >= 0, name);
      Assert.assertEquals(b.readInt(), n, name);
      Assert.assertEquals(c.readInt(), n, name);
      boolean stable = true, equal = true;
      double variance = 0, error = 0, norm = 0;
      for (int i = 0; i < n; i++) {
        int one = a.readInt(), two = b.readInt(), actual = c.readInt();
        float x = Float.intBitsToFloat(one),
            y = Float.intBitsToFloat(two),
            z = Float.intBitsToFloat(actual);
        if (!Float.isFinite(x) || !Float.isFinite(y) || !Float.isFinite(z))
          throw new AssertionError(name + " contains a non-finite value at " + i);
        stable &= one == two;
        equal &= one == actual;
        double delta = (double) y - x, difference = (double) z - x;
        variance += delta * delta;
        error += difference * difference;
        norm += (double) x * x;
      }
      if (name.startsWith("start/"))
        Assert.assertTrue(stable, name + " baseline initial weights differ");
      if (stable) Assert.assertTrue(equal, name + " differs in raw bits");
      else {
        double divisor = Math.max(norm, Double.MIN_NORMAL);
        Assert.assertTrue(
            error <= variance,
            name
                + " relative L2 "
                + Math.sqrt(error / divisor)
                + " exceeds baseline "
                + Math.sqrt(variance / divisor));
      }
    }
    Assert.assertEquals(a.read(), -1);
    Assert.assertEquals(b.read(), -1);
    Assert.assertEquals(c.read(), -1);
  }
}
