package com.epsilon.prism.grp;

import java.io.BufferedInputStream;
import java.io.BufferedOutputStream;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.EOFException;
import java.io.IOException;
import java.nio.channels.FileChannel;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * 1 コホートの主経路の対局の記録 grp/games/c&lt;t&gt;.bin.gz の形（{@link CohortGames}）。先頭にコホートの Φ と挙動（{@link GameLogHeader}）、
 * 続いて半荘ごとに半荘 ID・半荘が続く局末ごとの次局の整数（局・本場・供託・4 人の点数）・最終の点数。符号化と復号はこのクラスだけ。
 *
 * <p>gzip の中は {@link DataOutputStream}（big endian）の並び:
 *
 * <pre>
 * i32 MAGIC  i32 FORMAT
 * i32 cohort  UTF phiId  UTF phiSha256  f64 epsilonB  f64 exploreTemperature  i64 mainDecisions  i64 mainNonGreedy
 * i32 半荘の数
 * 半荘ごと: i64 gameId  u16 半荘が続く局末の数 k  （k 回: u8 局番号  u16 本場  u16 供託  i32×4 点数）  i32×4 最終の点数
 * i32 END（書き終えた印。gzip の CRC と合わせて、途中で切れた記録を拒む）
 * </pre>
 */
public final class GameLog {

  /** 先頭の印（"PGL1"）。 */
  private static final int MAGIC = 0x50474C31;

  /** 形式の番号（変えたら古い記録は読まない）。 */
  private static final int FORMAT = 1;

  /** 末尾の印（"END!"）。 */
  private static final int END = 0x454E4421;

  private static final int BUFFER = 1 << 16;

  private GameLog() {}

  /**
   * file に書く（一時ファイルに書いて fsync し、rename で差し替える）。
   *
   * @throws IOException 書けないとき
   */
  public static void write(Path file, CohortGames games) throws IOException {
    Files.createDirectories(file.toAbsolutePath().getParent());
    Path temporary = file.resolveSibling(file.getFileName() + ".tmp");
    try {
      try (var out =
          new DataOutputStream(
              new BufferedOutputStream(
                  new GZIPOutputStream(Files.newOutputStream(temporary), BUFFER), BUFFER))) {
        encode(out, games);
      }
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        channel.force(true);
      }
    } catch (IOException | RuntimeException failure) {
      Files.deleteIfExists(temporary);
      throw failure;
    }
    Files.move(temporary, file, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /**
   * file を読む。
   *
   * @throws IOException 読めないか、形が壊れているとき
   */
  public static CohortGames read(Path file) throws IOException {
    try (var in =
        new DataInputStream(
            new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      return decode(in, file);
    } catch (EOFException truncated) {
      throw new IOException("Truncated game log " + file, truncated);
    } catch (IllegalArgumentException malformed) {
      throw new IOException("Malformed game log " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  private static void encode(DataOutputStream out, CohortGames games) throws IOException {
    GameLogHeader header = games.header();
    out.writeInt(MAGIC);
    out.writeInt(FORMAT);
    out.writeInt(header.cohort());
    out.writeUTF(header.phiId());
    out.writeUTF(header.phiSha256());
    out.writeDouble(header.epsilonB());
    out.writeDouble(header.exploreTemperature());
    out.writeLong(header.mainDecisions());
    out.writeLong(header.mainNonGreedy());
    out.writeInt(games.games().size());
    for (RecordedGame game : games.games()) {
      out.writeLong(game.gameId());
      List<HandStart> starts = game.starts();
      out.writeShort(bounded(starts.size(), 0xFFFF, "hand ends of game " + game.gameId()));
      for (HandStart start : starts) {
        out.writeByte(bounded(start.kyokuIndex(), 0xFF, "kyoku index"));
        out.writeShort(bounded(start.honba(), 0xFFFF, "honba"));
        out.writeShort(bounded(start.kyotaku(), 0xFFFF, "kyotaku"));
        writeScores(out, start.scores());
      }
      writeScores(out, game.finalScores());
    }
    out.writeInt(END);
  }

  private static CohortGames decode(DataInputStream in, Path file) throws IOException {
    if (in.readInt() != MAGIC) throw new IOException("Not a game log: " + file);
    int format = in.readInt();
    if (format != FORMAT) throw new IOException("Game log format " + format + " is not " + FORMAT + ": " + file);
    var header =
        new GameLogHeader(
            in.readInt(), in.readUTF(), in.readUTF(), in.readDouble(), in.readDouble(), in.readLong(), in.readLong());
    int count = in.readInt();
    if (count < 0) throw new IOException("Negative game count in " + file);
    List<RecordedGame> games = new ArrayList<>(count);
    for (int g = 0; g < count; g++) {
      long gameId = in.readLong();
      int hands = in.readUnsignedShort();
      List<HandStart> starts = new ArrayList<>(hands);
      for (int k = 0; k < hands; k++)
        starts.add(
            new HandStart(in.readUnsignedByte(), in.readUnsignedShort(), in.readUnsignedShort(), readScores(in)));
      games.add(new RecordedGame(gameId, starts, readScores(in)));
    }
    if (in.readInt() != END) throw new IOException("Game log without its end mark: " + file);
    if (in.read() != -1) throw new IOException("Trailing bytes after the end mark: " + file);
    return new CohortGames(header, games);
  }

  /** 符号なしの欄に入る値（入らなければ書かずに拒む）。 */
  private static int bounded(int value, int max, String name) {
    if (value < 0 || value > max) throw new IllegalArgumentException(name + " out of range: " + value);
    return value;
  }

  private static void writeScores(DataOutputStream out, int[] scores) throws IOException {
    for (int score : scores) out.writeInt(score);
  }

  private static int[] readScores(DataInputStream in) throws IOException {
    return new int[] {in.readInt(), in.readInt(), in.readInt(), in.readInt()};
  }
}
