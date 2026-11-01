package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.github.luben.zstd.Zstd;
import java.io.EOFException;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.util.Arrays;
import java.util.Objects;
import java.util.zip.CRC32C;

/**
 * 封印したコホートの運の網 Ω の標本を読む。開くときに目録の形の指紋と shard の大きさを照らし、合わなければ読まない（互換の道は
 * 持たない）。
 *
 * <p>標本は {@link Cursor} で shard と枠の順に読む。枠ごとに展開して CRC32C を照らし、標本の数・長さ・O° の形・値の範囲が合わなければ
 * {@link IOException} にする。カーソルは O° と Y を使い回しの配列に読むので、標本ごとの割り当てをしない（{@link Cursor#sample} だけが
 * 写しを作る）。
 */
public final class LuckSampleStore {

  private final Path directory;
  private final LuckSampleWriter.Manifest manifest;

  private LuckSampleStore(Path directory, LuckSampleWriter.Manifest manifest) {
    this.directory = directory;
    this.manifest = manifest;
  }

  /**
   * 封印したコホートの標本を開く。
   *
   * @param cohort コホートのディレクトリ
   * @return 標本の読み手
   * @throws IOException 封印されていないか、形の指紋・shard の大きさ・数が目録と合わないとき
   */
  public static LuckSampleStore open(Path cohort) throws IOException {
    Path directory = LuckSampleWriter.directory(cohort);
    LuckSampleWriter.Manifest manifest = LuckSampleWriter.readManifest(directory);
    String expected = LuckSampleFormat.fingerprint();
    if (!expected.equals(manifest.fingerprint())) {
      throw new IOException(
          "Luck sample fingerprint " + manifest.fingerprint() + " is not " + expected);
    }
    long samples = 0;
    long bytes = 0;
    for (LuckSampleWriter.Shard shard : manifest.shards()) {
      Path path = directory.resolve(shard.file());
      if (!Files.isRegularFile(path) || Files.size(path) != shard.bytes()) {
        throw new IOException("Luck sample shard size mismatch: " + path);
      }
      samples += shard.samples();
      bytes += shard.bytes();
    }
    if (samples != manifest.samples() || bytes != manifest.bytes()) {
      throw new IOException("Luck sample manifest totals do not match its shards: " + directory);
    }
    return new LuckSampleStore(directory, manifest);
  }

  public LuckSampleWriter.Manifest manifest() {
    return manifest;
  }

  public long samples() {
    return manifest.samples();
  }

  /**
   * 全ての標本を頭から読むカーソルを作る。
   *
   * @return カーソル（使い終えたら閉じる）
   */
  public Cursor cursor() {
    return new Cursor();
  }

  /** 標本を順に読むカーソル。1 つのスレッドだけが使う。 */
  public final class Cursor implements AutoCloseable {

    private final long[] words = new long[LuckSchema.MAX_LONGS];
    private final float[] y = new float[GameState.NUM_PLAYERS];
    private final LuckObservation observation = new LuckObservation();
    private final ByteBuffer header =
        ByteBuffer.allocate(LuckSampleFormat.FRAME_HEADER).order(ByteOrder.LITTLE_ENDIAN);
    private final CRC32C crc = new CRC32C();
    private FileChannel channel;
    private int shard = -1;
    private long shardSamples;
    private ByteBuffer frame = ByteBuffer.allocate(0);
    private int frameSamples;
    private long read;
    private long game;
    private int stratum;
    private float weight;
    private boolean current;

    private Cursor() {}

    /**
     * 次の標本に進む。
     *
     * @return 標本があれば true、全て読んだら false
     * @throws IOException shard が壊れているとき
     */
    public boolean next() throws IOException {
      current = false;
      while (frameSamples == 0) {
        if (frame.hasRemaining()) {
          throw new IOException("Luck sample frame has trailing bytes in " + shardName());
        }
        if (!nextFrame()) {
          if (read != manifest.samples()) {
            throw new IOException("Luck samples end after " + read + " of " + manifest.samples());
          }
          return false;
        }
      }
      readSample();
      frameSamples--;
      read++;
      current = true;
      return true;
    }

    /** 今の標本の半荘 ID。 */
    public long game() {
      requireCurrent();
      return game;
    }

    /** 今の標本の O°（読み手は次の {@link #next} まで有効）。 */
    public LuckObservation observation() {
      requireCurrent();
      return observation;
    }

    /** 今の標本の O° を置いた配列（先頭は 0、長さは {@link LuckObservation#longs}）。次の {@link #next} で書き換わる。 */
    public long[] words() {
      requireCurrent();
      return words;
    }

    /** 今の標本の相対席 {@code seat} の Y。 */
    public float y(int seat) {
      requireCurrent();
      return y[Objects.checkIndex(seat, y.length)];
    }

    /** 今の標本の層の鍵。 */
    public int stratum() {
      requireCurrent();
      return stratum;
    }

    /** 今の標本の重み。 */
    public float weight() {
      requireCurrent();
      return weight;
    }

    /** 今の標本の写し。 */
    public LuckSample sample() {
      requireCurrent();
      return new LuckSample(game, Arrays.copyOf(words, observation.longs()), y, stratum, weight);
    }

    @Override
    public void close() throws IOException {
      current = false;
      if (channel != null) {
        channel.close();
        channel = null;
      }
    }

    /** 次の枠を展開する。shard の終わりでは次の shard へ移る。全て読んだら false。 */
    private boolean nextFrame() throws IOException {
      while (channel == null || channel.position() == channel.size()) {
        if (channel != null) {
          if (shardSamples != manifest.shards().get(shard).samples()) {
            throw new IOException("Luck sample count mismatch in " + shardName());
          }
          channel.close();
          channel = null;
        }
        if (shard + 1 == manifest.shards().size()) {
          return false;
        }
        shard++;
        shardSamples = 0;
        channel = FileChannel.open(directory.resolve(shardName()), StandardOpenOption.READ);
      }
      header.clear();
      readFully(header);
      header.flip();
      int compressedLength = header.getInt();
      int rawLength = header.getInt();
      int expected = header.getInt();
      int samples = header.getInt();
      if (compressedLength < 0
          || compressedLength > channel.size() - channel.position()
          || rawLength < LuckSampleFormat.SAMPLE_FIXED
          || rawLength > LuckSampleFormat.BLOCK
          || samples < 1) {
        throw new IOException("Corrupt luck sample frame header in " + shardName());
      }
      ByteBuffer compressed = ByteBuffer.allocate(compressedLength);
      readFully(compressed);
      byte[] raw;
      try {
        raw = Zstd.decompress(compressed.array(), rawLength);
      } catch (RuntimeException corrupt) {
        throw new IOException("Corrupt luck sample frame in " + shardName(), corrupt);
      }
      if (raw.length != rawLength) {
        throw new IOException("Luck sample frame length mismatch in " + shardName());
      }
      crc.reset();
      crc.update(raw);
      if ((int) crc.getValue() != expected) {
        throw new IOException("Luck sample frame CRC mismatch in " + shardName());
      }
      frame = ByteBuffer.wrap(raw).order(ByteOrder.LITTLE_ENDIAN);
      frameSamples = samples;
      shardSamples += samples;
      return true;
    }

    /** 枠から標本を 1 件読み、形と値の範囲を照らす。 */
    private void readSample() throws IOException {
      if (frame.remaining() < LuckSampleFormat.SAMPLE_FIXED) {
        throw new IOException("Truncated luck sample in " + shardName());
      }
      game = frame.getLong();
      int longs = Short.toUnsignedInt(frame.getShort());
      if (longs > LuckSchema.MAX_LONGS
          || frame.remaining()
              < longs * Long.BYTES + LuckSampleFormat.SAMPLE_FIXED - Long.BYTES - Short.BYTES) {
        throw new IOException("Corrupt luck sample length " + longs + " in " + shardName());
      }
      for (int index = 0; index < longs; index++) {
        words[index] = frame.getLong();
      }
      for (int seat = 0; seat < y.length; seat++) {
        y[seat] = frame.getFloat();
      }
      stratum = Short.toUnsignedInt(frame.getShort());
      weight = frame.getFloat();
      try {
        if (observation.bind(words, 0).longs() != longs) {
          throw new IOException("Luck sample does not match its schema in " + shardName());
        }
        LuckSample.requireValues(y, stratum, weight);
      } catch (RuntimeException corrupt) {
        throw new IOException("Corrupt luck sample in " + shardName(), corrupt);
      }
    }

    private void readFully(ByteBuffer target) throws IOException {
      while (target.hasRemaining()) {
        if (channel.read(target) < 0) {
          throw new EOFException("Luck sample frame past the end of " + shardName());
        }
      }
    }

    private String shardName() {
      return shard < 0 ? "(none)" : manifest.shards().get(shard).file();
    }

    private void requireCurrent() {
      if (!current) {
        throw new IllegalStateException("cursor has no current sample");
      }
    }
  }
}
