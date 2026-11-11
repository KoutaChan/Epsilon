package com.epsilon.classic.data;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * 牌譜から作った学習の行。1 行が 1 つの判断で、判断の記録 O、牌譜で選ばれた手（合法手の何番目か）、半荘の番号、局の番号を持つ。半荘ごとには、
 * 牌譜 id、検証用かどうか、各局の開始時の卓、終局の点を持つ。ネットワークの入力は学習のときに O から作るので、入力のフィールドを変えても
 * 牌譜を再生し直さなくてよい。
 *
 * <p>ファイルは DataOutputStream（ビッグエンディアン）で書いて gzip で圧縮したもので、中身は下の順に並ぶ。バージョン番号は持たない。
 * 途中で切れたファイルは、gzip の CRC と長さの検査で弾く。書くときは一時ファイルに書いてから名前を変える。O の long の総数を先に書くので、
 * 読むときは O のバッファをちょうどの大きさで 1 回だけ確保する。半荘が多いときは、行をいくつかのファイルに分けて置く（{@link RowChunks}）。
 *
 * <pre>
 * UTF Variant の名前  i32 半荘の数
 * 半荘ごと: UTF 牌譜 id  u8 検証用なら 1  u16 局の数  （局ごと: u8 局  u16 本場  u16 供託  i32 × S 点）  i32 × S 終局の点
 * i32 行の数  i32 O の long の総数
 * 行ごと: i32 半荘の番号  u16 局の番号  u8 選んだ手  u16 O の long の数  i64 × その数
 * </pre>
 */
public final class ClassicRows {

  /**
   * 牌譜の半荘 1 つ。
   *
   * @param id 牌譜 id
   * @param validation 検証用なら true（学習には使わない）
   * @param rounds 各局の開始時の卓（局の順）
   * @param finalScores 終局の点（絶対席の順）
   */
  public record Game(String id, boolean validation, List<RoundStart> rounds, int[] finalScores) {

    public Game {
      Objects.requireNonNull(id, "id");
      rounds = List.copyOf(rounds);
      if (rounds.isEmpty()) {
        throw new IllegalArgumentException("Game " + id + " has no round");
      }
      finalScores = finalScores.clone();
    }

    /** 席の終局の順位を返す。0 始まりで、同点なら起家に近い席が上。 */
    public int finalRank(int seat) {
      return Placement.rank(finalScores, seat);
    }

    @Override
    public int[] finalScores() {
      return finalScores.clone();
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Game game
          && id.equals(game.id)
          && validation == game.validation
          && rounds.equals(game.rounds)
          && Arrays.equals(finalScores, game.finalScores);
    }

    @Override
    public int hashCode() {
      return Objects.hash(id, validation, rounds, Arrays.hashCode(finalScores));
    }
  }

  private static final int BUFFER = 1 << 16;
  private static final int MAX_UNSIGNED_BYTE = 0xFF;
  private static final int MAX_UNSIGNED_SHORT = 0xFFFF;

  private final Variant variant;
  private final List<Game> games;
  private final ObservationBatch observations;
  private final int[] game;
  private final short[] round;
  private final byte[] chosen;

  private ClassicRows(Builder builder) {
    variant = builder.variant;
    games = List.copyOf(builder.games);
    observations = builder.observations;
    game = Arrays.copyOf(builder.game, builder.size);
    round = Arrays.copyOf(builder.round, builder.size);
    chosen = Arrays.copyOf(builder.chosen, builder.size);
  }

  public static Builder builder(Variant variant) {
    return new Builder(variant);
  }

  public Variant variant() {
    return variant;
  }

  /** 行の数。 */
  public int size() {
    return game.length;
  }

  public List<Game> games() {
    return games;
  }

  /** 全ての行の O を合わせた long の数。 */
  public int longs() {
    return usedLongs(observations);
  }

  private static int usedLongs(ObservationBatch batch) {
    int last = batch.size() - 1;
    return last < 0 ? 0 : batch.start(last) + batch.longs(last);
  }

  /** リーダーを付け替えて、行の判断の記録を読めるようにする。 */
  public Observation bind(int row, Observation reader) {
    return reader.bind(observations, row);
  }

  /** 行の半荘。 */
  public Game game(int row) {
    return games.get(game[row]);
  }

  /** 行の半荘の番号（{@link #games} の何番目か）。 */
  public int gameIndex(int row) {
    return game[row];
  }

  /** 行の局の番号（半荘の何局目か）。 */
  public int round(int row) {
    return round[row];
  }

  /** 牌譜で選ばれた手（合法手の何番目か）。 */
  public int chosen(int row) {
    return Byte.toUnsignedInt(chosen[row]);
  }

  /** 検証用の半荘の行なら true。 */
  public boolean validation(int row) {
    return games.get(game[row]).validation();
  }

  /** ファイルに書く。 */
  public void write(Path file) throws IOException {
    Path absolute = file.toAbsolutePath();
    if (absolute.getParent() != null) {
      Files.createDirectories(absolute.getParent());
    }
    Path temporary = absolute.resolveSibling(absolute.getFileName() + ".tmp");
    try (DataOutputStream out =
        new DataOutputStream(
            new BufferedOutputStream(new GZIPOutputStream(Files.newOutputStream(temporary), BUFFER), BUFFER))) {
      out.writeUTF(variant.name());
      out.writeInt(games.size());
      for (Game entry : games) {
        out.writeUTF(entry.id());
        out.writeByte(entry.validation() ? 1 : 0);
        out.writeShort(bounded(entry.rounds().size(), MAX_UNSIGNED_SHORT, "rounds"));
        for (RoundStart start : entry.rounds()) {
          out.writeByte(bounded(start.round(), MAX_UNSIGNED_BYTE, "round"));
          out.writeShort(bounded(start.honba(), MAX_UNSIGNED_SHORT, "honba"));
          out.writeShort(bounded(start.kyotaku(), MAX_UNSIGNED_SHORT, "kyotaku"));
          for (int seat = 0; seat < variant.seats(); seat++) {
            out.writeInt(start.score(seat));
          }
        }
        for (int seat = 0; seat < variant.seats(); seat++) {
          out.writeInt(entry.finalScores[seat]);
        }
      }
      out.writeInt(size());
      out.writeInt(longs());
      long[] words = observations.words();
      for (int row = 0; row < size(); row++) {
        out.writeInt(game[row]);
        out.writeShort(round[row]);
        out.writeByte(chosen(row));
        int longs = observations.longs(row);
        out.writeShort(bounded(longs, MAX_UNSIGNED_SHORT, "observation longs"));
        int start = observations.start(row);
        for (int index = 0; index < longs; index++) {
          out.writeLong(words[start + index]);
        }
      }
    } catch (IOException | RuntimeException failure) {
      Files.deleteIfExists(temporary);
      throw failure;
    }
    Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /** ファイルを読む。読めないとき、途中で切れているとき、中身が形式に合わないときは IOException を投げる。 */
  public static ClassicRows read(Path file) throws IOException {
    try (DataInputStream in =
        new DataInputStream(new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      Variant variant = Variant.valueOf(in.readUTF());
      int seats = variant.seats();
      int games = in.readInt();
      List<Game> entries = new ArrayList<>(Math.max(0, games));
      for (int index = 0; index < games; index++) {
        String id = in.readUTF();
        boolean validation = in.readUnsignedByte() != 0;
        int count = in.readUnsignedShort();
        List<RoundStart> rounds = new ArrayList<>(count);
        for (int round = 0; round < count; round++) {
          int number = in.readUnsignedByte();
          int honba = in.readUnsignedShort();
          int kyotaku = in.readUnsignedShort();
          rounds.add(new RoundStart(variant, number, honba, kyotaku, scores(in, seats)));
        }
        entries.add(new Game(id, validation, rounds, scores(in, seats)));
      }
      int rows = in.readInt();
      int totalLongs = in.readInt();
      if (rows < 0 || totalLongs < 0) {
        throw new IOException("Negative row or long count in " + file + ": " + rows + ", " + totalLongs);
      }
      Builder builder = new Builder(variant, Math.max(1, totalLongs), Math.max(1, rows));
      entries.forEach(builder::game);
      Observation reader = new Observation(variant);
      long[] words = new long[MAX_UNSIGNED_SHORT];
      for (int row = 0; row < rows; row++) {
        int gameIndex = in.readInt();
        int round = in.readUnsignedShort();
        int chosen = in.readUnsignedByte();
        int longs = in.readUnsignedShort();
        for (int index = 0; index < longs; index++) {
          words[index] = in.readLong();
        }
        Observation observation = reader.bind(words, 0);
        if (observation.longs() != longs) {
          throw new IOException("Row " + row + " of " + file + " holds " + longs + " longs, its O " + observation.longs());
        }
        builder.row(gameIndex, round, chosen, observation);
      }
      if (builder.longs() != totalLongs) {
        throw new IOException("Rows of " + file + " hold " + builder.longs() + " longs, the header " + totalLongs);
      }
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the last row of " + file);
      }
      return builder.build();
    } catch (EOFException truncated) {
      throw new IOException("Truncated rows " + file, truncated);
    } catch (IllegalArgumentException | IndexOutOfBoundsException malformed) {
      throw new IOException("Malformed rows " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  private static int[] scores(DataInputStream in, int seats) throws IOException {
    int[] scores = new int[seats];
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = in.readInt();
    }
    return scores;
  }

  private static int bounded(int value, int max, String name) {
    if (value < 0 || value > max) {
      throw new IllegalArgumentException(name + " does not fit the rows: " + value);
    }
    return value;
  }

  /** 半荘と行を順に足していくビルダー。行を足すたびに、その行の O をバッチにコピーする。 */
  public static final class Builder {

    private final Variant variant;
    private final List<Game> games = new ArrayList<>();
    private final ObservationBatch observations;
    private int[] game;
    private short[] round;
    private byte[] chosen;
    private int size;

    private Builder(Variant variant) {
      this(variant, 1 << 16, 1 << 12);
    }

    /**
     * @param initialLongs 最初に確保する O の long の数
     * @param initialRows 最初に確保する行の数
     */
    private Builder(Variant variant, int initialLongs, int initialRows) {
      this.variant = Objects.requireNonNull(variant, "variant");
      observations = new ObservationBatch(variant, initialLongs);
      game = new int[initialRows];
      round = new short[initialRows];
      chosen = new byte[initialRows];
    }

    public Variant variant() {
      return variant;
    }

    /** これまでに足した O の long の数。 */
    int longs() {
      return usedLongs(observations);
    }

    /** 半荘を足し、その番号を返す。 */
    public int game(Game entry) {
      if (entry.rounds().getFirst().variant() != variant) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + variant + " but found " + entry.rounds().getFirst().variant());
      }
      games.add(entry);
      return games.size() - 1;
    }

    /** これまでに足した行の数。 */
    public int size() {
      return size;
    }

    /**
     * リーダーが読んでいる O をバッチにコピーして、行を足す。
     *
     * @throws IllegalArgumentException 選んだ手が 0〜255 の外のとき
     * @throws IndexOutOfBoundsException 半荘か局の番号が範囲の外のとき
     */
    public void row(int gameIndex, int roundIndex, int choice, Observation observation) {
      Objects.checkIndex(gameIndex, games.size());
      Objects.checkIndex(roundIndex, games.get(gameIndex).rounds().size());
      if (choice < 0 || choice > MAX_UNSIGNED_BYTE) {
        throw new IllegalArgumentException("Choice out of range: " + choice);
      }
      observations.add(observation);
      if (size == game.length) {
        int capacity = 2 * game.length;
        game = Arrays.copyOf(game, capacity);
        round = Arrays.copyOf(round, capacity);
        chosen = Arrays.copyOf(chosen, capacity);
      }
      game[size] = gameIndex;
      round[size] = (short) roundIndex;
      chosen[size] = (byte) choice;
      size++;
    }

    public ClassicRows build() {
      if (observations.size() != size) {
        throw new IllegalStateException(observations.size() + " observations for " + size + " rows");
      }
      return new ClassicRows(this);
    }
  }
}
