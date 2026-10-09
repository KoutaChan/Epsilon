package com.epsilon.cfr.data;

import com.epsilon.io.ZstdFrameBuffer;
import com.github.luben.zstd.Zstd;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.security.DigestInputStream;
import java.security.DigestOutputStream;
import java.security.MessageDigest;
import java.util.HashSet;
import java.util.HexFormat;
import java.util.LinkedHashMap;
import java.util.Set;
import java.util.UUID;

/** 共通の学習データを行単位のチェックサム付きで追記保存する。復号した行のキャッシュには容量上限を設ける。 */
final class CfrPayloadStore implements AutoCloseable {
  private static final long SEGMENT_BYTES = 128L * 1024 * 1024;
  private final Path directory;
  private final long cacheBudget;
  private final ZstdFrameBuffer frames = new ZstdFrameBuffer(1);
  private final LinkedHashMap<Ref, CfrTrainingRow> cache = new LinkedHashMap<>(16, .75f, true);
  private long cachedBytes;
  private long storedBytes;
  private FileChannel writer;
  private String writerFile;

  CfrPayloadStore(Path directory, long cacheBudget) throws IOException {
    if (cacheBudget < 0) throw new IllegalArgumentException("Negative cache budget");
    this.directory = directory.toAbsolutePath().normalize();
    this.cacheBudget = cacheBudget;
    Files.createDirectories(this.directory);
    try (var files = Files.list(this.directory)) {
      for (Path path :
          files.filter(p -> p.getFileName().toString().startsWith("payload-")).toList()) {
        storedBytes = Math.addExact(storedBytes, Files.size(path));
      }
    }
  }

  synchronized Ref append(CfrTrainingRow row) throws IOException {
    synchronized (frames) {
      MessageDigest digest = CfrReplayCodec.sha256();
      DataOutputStream output =
          new DataOutputStream(new DigestOutputStream(frames.resetOutput(), digest));
      CfrReplayCodec.writeRow(output, row);
      output.flush();
      ByteBuffer frame = frames.compress();
      int length = frame.remaining();
      prepareWriter(length);
      long offset = writer.position();
      while (frame.hasRemaining()) writer.write(frame);
      storedBytes = Math.addExact(storedBytes, length);
      return new Ref(writerFile, offset, length, HexFormat.of().formatHex(digest.digest()));
    }
  }

  /** 呼び出し側で復号した行を検証した後、保存済みのフレームをそのまま複製する。 */
  synchronized Ref copyFrame(Ref source) throws IOException {
    try (FileChannel input = FileChannel.open(resolve(source.file()), StandardOpenOption.READ)) {
      if (source.offset() > input.size() - source.length())
        throw new IOException("Truncated CFR compaction source");
      prepareWriter(source.length());
      long offset = writer.position();
      long copied = 0;
      ByteBuffer fallback = null;
      while (copied < source.length()) {
        long count = input.transferTo(source.offset() + copied, source.length() - copied, writer);
        if (count == 0) {
          if (fallback == null) fallback = ByteBuffer.allocateDirect(64 * 1024);
          fallback.clear().limit((int) Math.min(fallback.capacity(), source.length() - copied));
          count = input.read(fallback, source.offset() + copied);
          if (count <= 0) throw new IOException("Could not read CFR compaction source");
          fallback.flip();
          while (fallback.hasRemaining()) writer.write(fallback);
        }
        copied += count;
      }
      storedBytes = Math.addExact(storedBytes, source.length());
      return new Ref(writerFile, offset, source.length(), source.sha256());
    }
  }

  private void prepareWriter(int length) throws IOException {
    if (writer == null || writer.position() + length > SEGMENT_BYTES) {
      seal();
      writerFile = "payload-" + UUID.randomUUID() + ".bin";
      writer =
          FileChannel.open(
              directory.resolve(writerFile),
              StandardOpenOption.CREATE_NEW,
              StandardOpenOption.WRITE);
    }
  }

  static long maximumCompressedBytes(CfrTrainingRow row) {
    return Zstd.compressBound(row.serializedBytes());
  }

  synchronized void force() throws IOException {
    if (writer != null) writer.force(true);
  }

  synchronized void seal() throws IOException {
    if (writer != null) {
      writer.force(true);
      writer.close();
      writer = null;
      writerFile = null;
    }
  }

  CfrTrainingRow read(Ref ref) throws IOException {
    return read(ref, false);
  }

  CfrTrainingRow readVerified(Ref ref) throws IOException {
    return read(ref, true);
  }

  private CfrTrainingRow read(Ref ref, boolean verifyDisk) throws IOException {
    synchronized (cache) {
      CfrTrainingRow cached = cache.get(ref);
      if (!verifyDisk && cached != null) return cached;
    }
    Path path = resolve(ref.file());
    CfrTrainingRow row;
    synchronized (frames) {
      try (FileChannel channel = FileChannel.open(path, StandardOpenOption.READ)) {
        if (ref.offset() > channel.size() - ref.length())
          throw new IOException("Truncated CFR payload: " + path);
        long limit = CfrDataChecks.MAX_OBSERVATION_BYTES + 65L + 20L * CfrDataChecks.MAX_ACTIONS;
        DataInputStream input = frames.read(channel, ref.offset(), ref.length(), limit);
        MessageDigest digest = verifyDisk ? CfrReplayCodec.sha256() : null;
        if (verifyDisk) input = new DataInputStream(new DigestInputStream(input, digest));
        row = CfrReplayCodec.readRow(input);
        if (input.read() != -1) throw new IOException("Trailing CFR payload data");
        if (verifyDisk
            && !MessageDigest.isEqual(digest.digest(), HexFormat.of().parseHex(ref.sha256()))) {
          throw new IOException("CFR payload checksum mismatch: " + path);
        }
      } catch (IllegalArgumentException error) {
        throw new IOException("Invalid CFR payload", error);
      }
    }
    long cost = cacheCost(row);
    if (cost <= cacheBudget) {
      synchronized (cache) {
        if (!cache.containsKey(ref)) {
          while (cachedBytes + cost > cacheBudget && !cache.isEmpty()) {
            var iterator = cache.entrySet().iterator();
            cachedBytes -= cacheCost(iterator.next().getValue());
            iterator.remove();
          }
          cache.put(ref, row);
          cachedBytes += cost;
        }
      }
    }
    return row;
  }

  void validate(Ref ref) throws IOException {
    Path path = resolve(ref.file());
    if (!Files.isRegularFile(path) || ref.offset() > Files.size(path) - ref.length()) {
      throw new IOException("Missing or truncated CFR payload: " + path);
    }
  }

  synchronized long storedBytes() {
    return storedBytes;
  }

  synchronized long garbageCollect(Set<String> referencedFiles) throws IOException {
    Set<String> retained = new HashSet<>(referencedFiles);
    if (writerFile != null) retained.add(writerFile);
    long removed = 0;
    try (var files = Files.list(directory)) {
      for (Path path :
          files
              .filter(p -> p.getFileName().toString().matches("payload-[a-f0-9-]+\\.bin"))
              .toList()) {
        if (!retained.contains(path.getFileName().toString())) {
          removed += Files.size(path);
          Files.delete(path);
        }
      }
    }
    storedBytes -= removed;
    synchronized (cache) {
      cache.entrySet().removeIf(entry -> !retained.contains(entry.getKey().file()));
      cachedBytes = cache.values().stream().mapToLong(CfrPayloadStore::cacheCost).sum();
    }
    return removed;
  }

  long cachedBytes() {
    synchronized (cache) {
      return cachedBytes;
    }
  }

  private Path resolve(String file) throws IOException {
    if (!file.matches("payload-[a-f0-9-]+\\.bin"))
      throw new IOException("Invalid CFR segment reference");
    return directory.resolve(file);
  }

  private static long cacheCost(CfrTrainingRow row) {
    return row.serializedBytes() + 256;
  }

  @Override
  public void close() throws IOException {
    try {
      seal();
    } finally {
      synchronized (frames) {
        frames.close();
      }
      synchronized (cache) {
        cache.clear();
        cachedBytes = 0;
      }
    }
  }

  record Ref(String file, long offset, int length, String sha256) {
    Ref {
      if (file == null
          || !file.matches("payload-[a-f0-9-]+\\.bin")
          || offset < 0
          || length < 1
          || length
              > Zstd.compressBound(
                  CfrDataChecks.MAX_OBSERVATION_BYTES + 65L + 20L * CfrDataChecks.MAX_ACTIONS)
          || sha256 == null
          || !sha256.matches("[a-f0-9]{64}")) {
        throw new IllegalArgumentException("Invalid CFR payload reference");
      }
    }
  }
}
