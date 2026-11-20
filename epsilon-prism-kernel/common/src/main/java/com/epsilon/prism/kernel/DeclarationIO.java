package com.epsilon.prism.kernel;

import java.io.ByteArrayOutputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.function.Supplier;

final class DeclarationIO {
  private DeclarationIO() {}

  static void positive(int n) {
    if (n <= 0) throw new IllegalArgumentException("Expected a positive dimension");
  }

  static void size(int n) {
    if (n < 0) throw new IllegalArgumentException("Negative size");
  }

  static void scale(float n) {
    if (!(n > 0) || !Float.isFinite(n)) throw new IllegalArgumentException("Invalid scale");
  }

  static void text(String s) {
    if (s.isEmpty() || !StandardCharsets.US_ASCII.newEncoder().canEncode(s))
      throw new IllegalArgumentException("Expected a nonempty ASCII identifier or description");
  }

  static final class Output {
    final ByteArrayOutputStream bytes = new ByteArrayOutputStream();

    void word(int v) {
      bytes.write(v);
      bytes.write(v >>> 8);
      bytes.write(v >>> 16);
      bytes.write(v >>> 24);
    }

    void positive(int v) {
      DeclarationIO.positive(v);
      word(v);
    }

    void size(int v) {
      DeclarationIO.size(v);
      word(v);
    }

    void scale(float v) {
      DeclarationIO.scale(v);
      word(Float.floatToRawIntBits(v));
    }

    void text(String s) {
      DeclarationIO.text(s);
      size(s.length());
      bytes.writeBytes(s.getBytes(StandardCharsets.US_ASCII));
    }
  }

  static final class Input {
    final ByteBuffer buffer;

    Input(ByteBuffer src) {
      buffer = src.slice().order(ByteOrder.LITTLE_ENDIAN);
    }

    int word() {
      if (buffer.remaining() < 4) throw new IllegalArgumentException("Truncated declaration");
      return buffer.getInt();
    }

    int positive() {
      int n = word();
      DeclarationIO.positive(n);
      return n;
    }

    int size() {
      int n = word();
      DeclarationIO.size(n);
      return n;
    }

    float scale() {
      float f = Float.intBitsToFloat(word());
      DeclarationIO.scale(f);
      return f;
    }

    String text() {
      int n = size();
      if (n > buffer.remaining()) throw new IllegalArgumentException("Truncated string");
      byte[] a = new byte[n];
      buffer.get(a);
      String s = new String(a, StandardCharsets.US_ASCII);
      DeclarationIO.text(s);
      return s;
    }

    <T> List<T> list(Supplier<T> read) {
      int n = size();
      if (n > buffer.remaining() / Integer.BYTES)
        throw new IllegalArgumentException("Truncated list");
      var items = new ArrayList<T>();
      for (int i = 0; i < n; i++) items.add(read.get());
      return items;
    }
  }
}
