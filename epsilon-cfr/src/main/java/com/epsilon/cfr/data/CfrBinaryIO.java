package com.epsilon.cfr.data;

import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FilterInputStream;
import java.io.IOException;
import java.io.InputStream;

/** 半荘とリプレイで共通する判断の形式と、読み込み長の境界を扱う。 */
final class CfrBinaryIO {
  private CfrBinaryIO() {}

  static void writeDecisionBody(DataOutputStream out, int[] actions, int selected, double[] policy)
      throws IOException {
    out.writeInt(actions.length);
    out.writeInt(selected);
    for (int i = 0; i < actions.length; i++) {
      out.writeInt(actions[i]);
      out.writeDouble(policy[i]);
    }
  }

  static DecisionBody readDecisionBody(DataInputStream in) throws IOException {
    int count = readCount(in, CfrDataChecks.MAX_ACTIONS, "actions");
    int selected = in.readInt();
    int[] actions = new int[count];
    double[] policy = new double[count];
    for (int i = 0; i < count; i++) {
      actions[i] = in.readInt();
      policy[i] = in.readDouble();
    }
    return new DecisionBody(actions, selected, policy);
  }

  static int readCount(DataInputStream in, int maximum, String description) throws IOException {
    int count = in.readInt();
    if (count < 0 || count > maximum)
      throw new IOException("Invalid " + description + ": " + count);
    return count;
  }

  record DecisionBody(int[] actions, int selected, double[] policy) {}

  static final class LimitedInputStream extends FilterInputStream {
    private long remaining;

    LimitedInputStream(InputStream input, long remaining) {
      super(input);
      this.remaining = remaining;
    }

    long remaining() {
      return remaining;
    }

    @Override
    public int read() throws IOException {
      if (remaining == 0) return -1;
      int value = super.read();
      if (value < 0) throw new EOFException();
      remaining--;
      return value;
    }

    @Override
    public int read(byte[] bytes, int offset, int length) throws IOException {
      if (length == 0) return 0;
      if (remaining == 0) return -1;
      int count = super.read(bytes, offset, (int) Math.min(length, remaining));
      if (count < 0) throw new EOFException();
      remaining -= count;
      return count;
    }

    @Override
    public long skip(long count) throws IOException {
      long requested = Math.min(Math.max(count, 0), remaining);
      if (requested == 0) return 0;
      // DigestInputStream.skipでは読み飛ばした部分がハッシュに含まれないため、readで読み進める。
      byte[] buffer = new byte[(int) Math.min(8192, requested)];
      long skipped = 0;
      while (skipped < requested) {
        skipped += read(buffer, 0, (int) Math.min(buffer.length, requested - skipped));
      }
      return skipped;
    }
  }
}
