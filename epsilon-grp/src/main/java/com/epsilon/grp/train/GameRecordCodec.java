package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.Variant;
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
 * GameCohort を 1 つのファイル（gzip で圧縮）に読み書きする。中身は DataOutputStream（ビッグエンディアン）で次の順に並べる:
 *
 * <pre>
 * UTF Variant の名前  i32 コホート  UTF Φ の項目の id  i32 半荘の数
 * 半荘ごと: i64 半荘 ID  u16 終局しなかった局末の数  （局末ごと: u8 次局  u16 本場  u16 供託の本数  i32 × S 点）  i32 × S 最終の点
 * </pre>
 *
 * バージョンの番号もマジックナンバーも持たない。読むときは最後の半荘の後もファイルの終わりまで読む。そのため途中で切れたファイルは、
 * gzip の CRC と長さの検査で拒まれる。書くときは一時ファイルに書いて fsync し、リネームで差し替える。
 */
public final class GameRecordCodec {

  private static final int BUFFER = 1 << 16;
  private static final int MAX_UNSIGNED_BYTE = 0xFF;
  private static final int MAX_UNSIGNED_SHORT = 0xFFFF;

  private GameRecordCodec() {}

  /** u8・u16 のフィールドに収まらない値があれば IllegalArgumentException を投げ、ファイルは変えない。 */
  public static void write(Path file, GameCohort cohort) throws IOException {
    Path absolute = file.toAbsolutePath();
    Files.createDirectories(absolute.getParent());
    Path temporary = absolute.resolveSibling(absolute.getFileName() + ".tmp");
    try {
      try (DataOutputStream out = new DataOutputStream(
          new BufferedOutputStream(new GZIPOutputStream(Files.newOutputStream(temporary), BUFFER), BUFFER))) {
        encode(out, cohort);
      }
      try (FileChannel channel = FileChannel.open(temporary, StandardOpenOption.WRITE)) {
        channel.force(true);
      }
    } catch (IOException | RuntimeException failure) {
      Files.deleteIfExists(temporary);
      throw failure;
    }
    Files.move(temporary, absolute, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
  }

  /** 読めないとき、途中で切れているとき、中身が卓として正しくないときは IOException を投げる。 */
  public static GameCohort read(Path file) throws IOException {
    try (DataInputStream in = new DataInputStream(
        new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      GameCohort cohort = decode(in);
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the last game of " + file);
      }
      return cohort;
    } catch (EOFException truncated) {
      throw new IOException("Truncated game records " + file, truncated);
    } catch (IllegalArgumentException malformed) {
      throw new IOException("Malformed game records " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  private static void encode(DataOutputStream out, GameCohort cohort) throws IOException {
    out.writeUTF(cohort.variant().name());
    out.writeInt(cohort.cohort());
    out.writeUTF(cohort.phiId());
    out.writeInt(cohort.games().size());
    for (GameRecord game : cohort.games()) {
      out.writeLong(game.gameId());
      List<GrpState> handEnds = game.handEnds();
      out.writeShort(bounded(handEnds.size(), MAX_UNSIGNED_SHORT, "hand ends of game " + game.gameId()));
      for (GrpState state : handEnds) {
        out.writeByte(bounded(state.round(), MAX_UNSIGNED_BYTE, "round"));
        out.writeShort(bounded(state.honba(), MAX_UNSIGNED_SHORT, "honba"));
        out.writeShort(bounded(state.sticks(), MAX_UNSIGNED_SHORT, "sticks"));
        for (int seat = 0; seat < cohort.variant().seats(); seat++) {
          out.writeInt(state.score(seat));
        }
      }
      for (int seat = 0; seat < cohort.variant().seats(); seat++) {
        out.writeInt(game.finalScore(seat));
      }
    }
  }

  private static GameCohort decode(DataInputStream in) throws IOException {
    Variant variant = Variant.valueOf(in.readUTF());
    int cohort = in.readInt();
    String phiId = in.readUTF();
    int count = in.readInt();
    if (count < 0) {
      throw new IllegalArgumentException("Negative game count " + count);
    }
    // ヘッダーの数が壊れたファイルでは、その数で先に領域を取ると OutOfMemoryError になる。読めた半荘だけを足す。
    List<GameRecord> games = new ArrayList<>();
    for (int game = 0; game < count; game++) {
      long gameId = in.readLong();
      int hands = in.readUnsignedShort();
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        int round = in.readUnsignedByte();
        int honba = in.readUnsignedShort();
        int sticks = in.readUnsignedShort();
        handEnds.add(new GrpState(variant, round, honba, sticks, scores(in, variant)));
      }
      games.add(new GameRecord(variant, gameId, handEnds, scores(in, variant)));
    }
    return new GameCohort(variant, cohort, phiId, games);
  }

  private static int[] scores(DataInputStream in, Variant variant) throws IOException {
    int[] scores = new int[variant.seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = in.readInt();
    }
    return scores;
  }

  private static int bounded(int value, int max, String name) {
    if (value < 0 || value > max) {
      throw new IllegalArgumentException(name + " does not fit the record: " + value);
    }
    return value;
  }
}
