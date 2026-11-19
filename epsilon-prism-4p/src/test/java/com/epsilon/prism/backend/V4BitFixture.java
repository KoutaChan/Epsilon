package com.epsilon.prism.backend;

import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import org.testng.Assert;

/** Raw FP32 fixtures captured by the identical test code on the K1 tree. */
public final class V4BitFixture {
  private V4BitFixture() {}

  public record Difference(
      long elements, long differing, double squaredError, double squaredReference, float max) {
    public Difference plus(Difference other) {
      return new Difference(
          elements + other.elements,
          differing + other.differing,
          squaredError + other.squaredError,
          squaredReference + other.squaredReference,
          Math.max(max, other.max));
    }

    public double relativeL2() {
      return Math.sqrt(squaredError / Math.max(squaredReference, 1e-30));
    }
  }

  public static Difference compare(String label, String field, float[] values) {
    String directory = System.getProperty("prism.v4.bitFixtures");
    Difference same = new Difference(values.length, 0, 0.0, 0.0, 0f);
    if (directory == null) return same;
    Path path = Path.of(directory, label, field + ".f32");
    try {
      if (Boolean.getBoolean("prism.v4.writeBitFixtures")) {
        Files.createDirectories(path.getParent());
        ByteBuffer bytes =
            ByteBuffer.allocate(Math.multiplyExact(values.length, Float.BYTES))
                .order(ByteOrder.LITTLE_ENDIAN);
        for (float value : values) bytes.putInt(Float.floatToRawIntBits(value));
        Files.write(path, bytes.array());
        return same;
      }
      ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(path)).order(ByteOrder.LITTLE_ENDIAN);
      Assert.assertEquals(
          bytes.remaining(), Math.multiplyExact(values.length, Float.BYTES), path.toString());
      long differing = 0;
      double squaredError = 0;
      double squaredReference = 0;
      float max = 0f;
      for (float value : values) {
        int bits = bytes.getInt();
        if (bits != Float.floatToRawIntBits(value)) differing++;
        float reference = Float.intBitsToFloat(bits);
        double error = (double) value - reference;
        squaredError += error * error;
        squaredReference += (double) reference * reference;
        max = Math.max(max, (float) Math.abs(error));
      }
      return new Difference(values.length, differing, squaredError, squaredReference, max);
    } catch (IOException error) {
      throw new UncheckedIOException(path.toString(), error);
    }
  }
}
