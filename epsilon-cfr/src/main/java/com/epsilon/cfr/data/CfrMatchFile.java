package com.epsilon.cfr.data;

import com.epsilon.cfr.data.CfrBinaryIO.DecisionBody;
import com.github.luben.zstd.RecyclingBufferPool;
import com.github.luben.zstd.ZstdInputStream;
import com.github.luben.zstd.ZstdOutputStream;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.FilterInputStream;
import java.io.FilterOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.channels.Channels;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.CRC32;

/** 局ごとの圧縮経路を追記し、半荘の完了時に一度だけ同期して公開する保存形式。 */
public final class CfrMatchFile {
  private static final int MAGIC = 0x43464d31;
  private static final int VERSION = 1;
  private static final int FRAME = 0x46524d31;
  private static final int FOOTER = 0x454e4431;
  private static final int HEADER_BYTES = 16;
  private static final int FRAME_BYTES = 20;
  private static final int FOOTER_BYTES = 36;
  private static final int MAX_TRAJECTORIES = 100_000;
  private static final int MAX_DECISIONS = 1_000_000;
  private static final long MAX_EPISODE_BYTES = 1L << 30;

  private CfrMatchFile() {}

  /** 局の観測を展開せずに読める、完了済み半荘の集計値。 */
  public record Summary(long gameId, int trajectories, long decisions) {}

  public static Summary summary(Path path) throws IOException {
    requireCompletedPath(path);
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      return summary(channel);
    }
  }

  /** 半荘内の経路を局順に復元する。圧縮データ、連番、判断IDと終端の集計値を検証する。 */
  public static List<CfrEpisode> read(Path path) throws IOException {
    requireCompletedPath(path);
    try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
      Summary expected = summary(channel);
      long contentEnd = channel.size() - FOOTER_BYTES;
      channel.position(HEADER_BYTES);
      var episodes = new ArrayList<CfrEpisode>(Math.min(expected.trajectories(), 256));
      long decisions = 0;
      long previousDecision = -1;
      for (int index = 0; index < expected.trajectories(); index++) {
        if (contentEnd - channel.position() < FRAME_BYTES)
          throw new IOException("Truncated CFR match frame");
        ByteBuffer header = read(channel, FRAME_BYTES);
        int marker = header.getInt();
        int ordinal = header.getInt();
        int flags = header.getInt();
        long bytes = header.getLong();
        if (marker != FRAME
            || ordinal != index
            || flags != (index + 1 == expected.trajectories() ? 1 : 0)
            || bytes <= 0
            || bytes > contentEnd - channel.position())
          throw new IOException("Invalid CFR match frame header");
        var compressed =
            new CfrBinaryIO.LimitedInputStream(Channels.newInputStream(channel), bytes);
        CfrEpisode episode = readCompressedEpisode(compressed);
        if (compressed.remaining() != 0 || episode.episodeId() != expected.gameId())
          throw new IOException("CFR match frame identity or length mismatch");
        previousDecision = lastDecision(episode, previousDecision);
        decisions = Math.addExact(decisions, episode.decisions().size());
        episodes.add(episode);
      }
      if (channel.position() != contentEnd || decisions != expected.decisions())
        throw new IOException("CFR match summary mismatch");
      return List.copyOf(episodes);
    }
  }

  private static void requireCompletedPath(Path path) throws IOException {
    if (path.getFileName().toString().endsWith(".part"))
      throw new IOException("Unpublished CFR match file");
  }

  private static Summary summary(FileChannel channel) throws IOException {
    long size = channel.size();
    if (size < HEADER_BYTES + FRAME_BYTES + FOOTER_BYTES)
      throw new IOException("Truncated CFR match file");
    channel.position(0);
    ByteBuffer header = read(channel, HEADER_BYTES);
    if (header.getInt() != MAGIC || header.getInt() != VERSION)
      throw new IOException("Unsupported CFR match format");
    long gameId = header.getLong();
    channel.position(size - FOOTER_BYTES);
    ByteBuffer footer = read(channel, FOOTER_BYTES);
    CRC32 checksum = new CRC32();
    checksum.update(footer.array(), 0, FOOTER_BYTES - Integer.BYTES);
    if (footer.getInt() != FOOTER || footer.getLong() != gameId)
      throw new IOException("CFR match footer identity mismatch");
    int trajectories = footer.getInt();
    long decisions = footer.getLong();
    long contentEnd = footer.getLong();
    if (gameId < 0
        || trajectories < 1
        || trajectories > MAX_TRAJECTORIES
        || decisions < 0
        || contentEnd != size - FOOTER_BYTES
        || contentEnd < HEADER_BYTES + (long) trajectories * FRAME_BYTES
        || footer.getInt() != (int) checksum.getValue())
      throw new IOException("Invalid CFR match summary");
    return new Summary(gameId, trajectories, decisions);
  }

  private static long lastDecision(CfrEpisode episode, long previous) throws IOException {
    if (episode.decisions().isEmpty()) return previous;
    if (episode.decisions().getFirst().decisionId() <= previous)
      throw new IOException("CFR match decision IDs must increase across trajectories");
    return episode.decisions().getLast().decisionId();
  }

  private static ByteBuffer read(FileChannel channel, int bytes) throws IOException {
    ByteBuffer buffer = ByteBuffer.allocate(bytes);
    while (buffer.hasRemaining()) {
      if (channel.read(buffer) < 0) throw new EOFException("Truncated CFR match file");
    }
    return buffer.flip();
  }

  private static void write(FileChannel channel, ByteBuffer buffer) throws IOException {
    buffer.flip();
    while (buffer.hasRemaining()) channel.write(buffer);
  }

  /** 同じ半荘への書き込みを直列化して使う追記状態。ファイルも観測payloadも呼び出しの間には保持しない。 */
  static final class Appender {
    private final Path target;
    private final Path partial;
    private final long gameId;
    private int trajectories;
    private long decisions;
    private long previousDecision = -1;
    private long position = HEADER_BYTES;
    private boolean completed;

    Appender(Path target, long gameId) {
      this.target = target.toAbsolutePath().normalize();
      this.partial = this.target.resolveSibling(this.target.getFileName() + ".part");
      this.gameId = gameId;
    }

    void append(CfrEpisodeFrame frame) throws IOException {
      CfrEpisode episode = frame.episode();
      if (completed
          || frame.index() != trajectories
          || episode.episodeId() != gameId
          || trajectories == MAX_TRAJECTORIES)
        throw new IOException("Invalid CFR match append order");
      if (Files.exists(target)) throw new IOException("CFR match is already published: " + target);
      long nextDecision = lastDecision(episode, previousDecision);
      long nextCount = Math.addExact(decisions, episode.decisions().size());
      if (trajectories == 0) Files.createDirectories(target.getParent());
      // 半荘を最初から再収集したときは、以前の未完了ファイルをここで置き換える。
      var options =
          trajectories == 0
              ? new StandardOpenOption[] {
                StandardOpenOption.WRITE,
                StandardOpenOption.CREATE,
                StandardOpenOption.TRUNCATE_EXISTING
              }
              : new StandardOpenOption[] {StandardOpenOption.WRITE};
      try (FileChannel channel = FileChannel.open(partial, options)) {
        if (trajectories == 0)
          write(
              channel,
              ByteBuffer.allocate(HEADER_BYTES).putInt(MAGIC).putInt(VERSION).putLong(gameId));
        if (channel.size() != position) throw new IOException("CFR partial match length changed");
        channel.position(position);
        long frameStart = position;
        write(
            channel,
            ByteBuffer.allocate(FRAME_BYTES)
                .putInt(FRAME)
                .putInt(trajectories)
                .putInt(frame.last() ? 1 : 0)
                .putLong(0));
        writeCompressedEpisode(Channels.newOutputStream(channel), episode);
        long end = channel.position();
        channel.position(frameStart + 3 * Integer.BYTES);
        write(channel, ByteBuffer.allocate(Long.BYTES).putLong(end - frameStart - FRAME_BYTES));
        channel.position(end);
        if (frame.last()) {
          ByteBuffer footer =
              ByteBuffer.allocate(FOOTER_BYTES)
                  .putInt(FOOTER)
                  .putLong(gameId)
                  .putInt(trajectories + 1)
                  .putLong(nextCount)
                  .putLong(end);
          CRC32 checksum = new CRC32();
          checksum.update(footer.array(), 0, footer.position());
          footer.putInt((int) checksum.getValue());
          write(channel, footer);
          channel.force(true);
        }
        position = end;
      }
      if (frame.last()) {
        Files.move(partial, target, StandardCopyOption.ATOMIC_MOVE);
        completed = true;
      }
      trajectories++;
      decisions = nextCount;
      previousDecision = nextDecision;
    }
  }

  /** 局の内容を一つのチェックサム付き圧縮フレームとして書く。外側のストリームは呼び出し側が所有する。 */
  private static void writeCompressedEpisode(OutputStream destination, CfrEpisode episode)
      throws IOException {
    if (episode.decisions().size() > MAX_DECISIONS)
      throw new IOException("Too many CFR episode decisions");
    try (DataOutputStream out =
        new DataOutputStream(
            new BufferedOutputStream(
                new BoundedOutputStream(
                    new ZstdOutputStream(nonClosing(destination), RecyclingBufferPool.INSTANCE, 1)
                        .setChecksum(true)
                        .setWindowLog(23),
                    MAX_EPISODE_BYTES),
                64 * 1024))) {
      out.writeLong(episode.episodeId());
      for (double utility : episode.utilities()) out.writeDouble(utility);
      out.writeInt(episode.decisions().size());
      for (CfrDecision decision : episode.decisions()) {
        out.writeLong(decision.decisionId());
        out.writeInt(decision.seat());
        CfrBinaryIO.writeDecisionBody(
            out, decision.legalActionIds(), decision.selectedSlot(), decision.targetPolicy());
        out.writeDouble(decision.behaviorProbability());
        decision.observation().write(out);
      }
      out.writeInt(episode.branchOutcomes().size());
      for (CfrBranchOutcome branch : episode.branchOutcomes()) {
        out.writeLong(branch.decisionId());
        out.writeDouble(branch.mainInclusionProbability());
        out.writeInt(branch.alternativeSlot());
        out.writeDouble(branch.alternativeInclusionProbability());
        out.writeDouble(branch.alternativeUtility());
        out.writeDouble(branch.alternativeLogSuffixWeight());
      }
    }
  }

  /** 外側で長さを制限した圧縮フレームから、形状とチェックサムを検証して局を復元する。 */
  private static CfrEpisode readCompressedEpisode(InputStream source) throws IOException {
    try (DataInputStream in =
        new DataInputStream(
            new BufferedInputStream(
                new BoundedInputStream(
                    new ZstdInputStream(nonClosing(source)).setLongMax(23), MAX_EPISODE_BYTES),
                64 * 1024))) {
      long episodeId = in.readLong();
      double[] utilities = new double[4];
      for (int i = 0; i < 4; i++) utilities[i] = in.readDouble();
      int count = CfrBinaryIO.readCount(in, MAX_DECISIONS, "decisions");
      List<CfrDecision> decisions = new ArrayList<>(Math.min(count, 4096));
      for (int i = 0; i < count; i++) {
        long id = in.readLong();
        int seat = in.readInt();
        DecisionBody body = CfrBinaryIO.readDecisionBody(in);
        decisions.add(
            new CfrDecision(
                id,
                seat,
                body.actions(),
                body.selected(),
                body.policy(),
                in.readDouble(),
                ObservationPayload.read(in)));
      }
      int branchCount = CfrBinaryIO.readCount(in, count, "branch outcomes");
      List<CfrBranchOutcome> branches = new ArrayList<>(branchCount);
      for (int i = 0; i < branchCount; i++) {
        branches.add(
            new CfrBranchOutcome(
                in.readLong(),
                in.readDouble(),
                in.readInt(),
                in.readDouble(),
                in.readDouble(),
                in.readDouble()));
      }
      if (in.read() != -1) throw new IOException("Unexpected trailing CFR episode data");
      return new CfrEpisode(episodeId, decisions, utilities, branches);
    } catch (IllegalArgumentException | NullPointerException invalid) {
      throw new IOException("Invalid CFR episode contents", invalid);
    }
  }

  private static OutputStream nonClosing(OutputStream output) {
    return new FilterOutputStream(output) {
      @Override
      public void write(byte[] bytes, int offset, int length) throws IOException {
        out.write(bytes, offset, length);
      }

      @Override
      public void close() throws IOException {
        flush();
      }
    };
  }

  private static InputStream nonClosing(InputStream input) {
    return new FilterInputStream(input) {
      @Override
      public void close() {}
    };
  }

  /** 圧縮ファイルのサイズや各行の制限とは別に、展開後のデータ全体にもサイズ上限を設ける。 */
  private static final class BoundedInputStream extends FilterInputStream {
    private long remaining;

    BoundedInputStream(InputStream input, long maximum) {
      super(input);
      remaining = maximum;
    }

    @Override
    public int read() throws IOException {
      int value = in.read();
      if (value >= 0) consume(1);
      return value;
    }

    @Override
    public int read(byte[] bytes, int offset, int length) throws IOException {
      int count = in.read(bytes, offset, (int) Math.min(length, remaining + 1));
      if (count > 0) consume(count);
      return count;
    }

    private void consume(int count) throws IOException {
      if (count > remaining) throw new IOException("CFR episode exceeds decoded byte limit");
      remaining -= count;
    }
  }

  private static final class BoundedOutputStream extends FilterOutputStream {
    private long remaining;

    BoundedOutputStream(OutputStream output, long maximum) {
      super(output);
      remaining = maximum;
    }

    @Override
    public void write(int value) throws IOException {
      consume(1);
      out.write(value);
    }

    @Override
    public void write(byte[] bytes, int offset, int length) throws IOException {
      consume(length);
      out.write(bytes, offset, length);
    }

    private void consume(int count) throws IOException {
      if (count > remaining) throw new IOException("CFR episode exceeds decoded byte limit");
      remaining -= count;
    }
  }
}
