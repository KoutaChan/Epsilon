package com.epsilon.cfr.data;

import com.epsilon.cfr.data.CfrBinaryIO.DecisionBody;
import com.epsilon.io.AtomicFiles;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.DigestInputStream;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;

/** リプレイの教師行と世代索引を保存する。公開と検証の単位はリプレイストアが管理する。 */
final class CfrReplayCodec {
  private static final int MAGIC = 0x43465231;
  private static final int VERSION = 3;
  static final int REPLAY = 3;
  static final int REPLAY_POINTER = 4;

  private CfrReplayCodec() {}

  static void writeRow(DataOutputStream out, CfrTrainingRow row) throws IOException {
    writeId(out, row.id());
    CfrBinaryIO.writeDecisionBody(
        out, row.legalActionIds(), row.selectedSlot(), row.targetPolicy());
    out.writeDouble(row.utility());
    out.writeDouble(row.logRegretWeight());
    out.writeDouble(row.logAverageWeight());
    row.observation().write(out);
    double[] targets = row.explicitRegretTargets();
    out.writeBoolean(targets != null);
    if (targets != null) {
      for (double target : targets) out.writeDouble(target);
    }
  }

  static CfrTrainingRow readRow(DataInputStream in) throws IOException {
    CfrSampleId id = readId(in);
    DecisionBody body = CfrBinaryIO.readDecisionBody(in);
    double utility = in.readDouble();
    double logRegret = in.readDouble();
    double logAverage = in.readDouble();
    ObservationPayload observation = ObservationPayload.read(in);
    double[] targets = null;
    if (in.readBoolean()) {
      targets = new double[body.actions().length];
      for (int i = 0; i < targets.length; i++) targets[i] = in.readDouble();
    }
    return new CfrTrainingRow(
        id,
        body.actions(),
        body.selected(),
        body.policy(),
        utility,
        logRegret,
        logAverage,
        observation,
        targets);
  }

  static void writeId(DataOutputStream out, CfrSampleId id) throws IOException {
    out.writeLong(id.iteration());
    out.writeLong(id.episodeId());
    out.writeLong(id.decisionId());
    out.writeInt(id.seat());
  }

  static CfrSampleId readId(DataInputStream in) throws IOException {
    return new CfrSampleId(in.readLong(), in.readLong(), in.readLong(), in.readInt());
  }

  static void writeEnvelope(Path destination, int kind, Writer writer) throws IOException {
    AtomicFiles.write(
        destination,
        stream -> {
          MessageDigest digest = sha256();
          DigestOutputStream hashing = new DigestOutputStream(stream, digest);
          DataOutputStream out = new DataOutputStream(new BufferedOutputStream(hashing));
          out.writeInt(MAGIC);
          out.writeInt(VERSION);
          out.writeInt(kind);
          writer.write(out);
          out.flush();
          byte[] checksum = digest.digest();
          hashing.on(false);
          out.write(checksum);
          out.flush();
        });
  }

  static <T> T readEnvelope(Path path, int kind, Reader<T> reader) throws IOException {
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      long size = channel.size();
      if (size < 44) throw new IOException("Truncated CFR file: " + path);
      MessageDigest digest = sha256();
      InputStream source = Channels.newInputStream(channel);
      DataInputStream in =
          new DataInputStream(
              new BufferedInputStream(
                  new CfrBinaryIO.LimitedInputStream(
                      new DigestInputStream(source, digest), size - 32),
                  64 * 1024));
      if (in.readInt() != MAGIC || in.readInt() != VERSION || in.readInt() != kind) {
        throw new IOException("Unsupported CFR binary format: " + path);
      }
      T result;
      try {
        result = reader.read(in);
      } catch (IllegalArgumentException | NullPointerException error) {
        throw new IOException("Invalid CFR binary contents: " + path, error);
      }
      try {
        if (in.read() != -1) throw new IOException("Unexpected trailing CFR data: " + path);
        byte[] expected = new byte[32];
        new DataInputStream(source).readFully(expected);
        if (!MessageDigest.isEqual(digest.digest(), expected))
          throw new IOException("CFR checksum mismatch: " + path);
        return result;
      } catch (IOException failure) {
        if (result instanceof AutoCloseable closeable) {
          try {
            closeable.close();
          } catch (Exception closeFailure) {
            failure.addSuppressed(closeFailure);
          }
        }
        throw failure;
      }
    }
  }

  static MessageDigest sha256() {
    try {
      return MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException error) {
      throw new IllegalStateException("SHA-256 unavailable", error);
    }
  }

  @FunctionalInterface
  interface Writer {
    void write(DataOutputStream out) throws IOException;
  }

  @FunctionalInterface
  interface Reader<T> {
    T read(DataInputStream in) throws IOException;
  }
}
