package com.epsilon.grp.train;

import com.epsilon.mahjong.rules.TableState;
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
import java.util.zip.GZIPInputStream;
import java.util.zip.GZIPOutputStream;

/**
 * CountedHands を数のファイル（gzip で圧縮）に読み書きする。中身は DataOutputStream（ビッグエンディアン）で次の順に並べる:
 *
 * <pre>
 * UTF Variant の名前  i32 コホート  f64 主経路の探索の温度
 * UTF Φ のモードの名前  f64 Φ の打ち切りの値（数え上げないモードでは NaN）  i32 半荘の数
 * 半荘ごと: i64 半荘 ID  u16 局の数  （局ごと: u8 局  u16 本場  u16 供託の本数  i32 × S 点  i64 結果のコード）
 * </pre>
 *
 * Φ は主経路の局末の値に使ったもの（{@link PhiStamp}）。局ごとの卓はその局の始まりの卓で、半荘が終わった局も入る。半荘は半荘 ID の
 * 昇順に並ぶ。バージョンの番号もマジックナンバーも持たない。
 * 読むときは最後の半荘の後もファイルの終わりまで読む。そのため
 * 途中で切れたファイルは、gzip の CRC と長さの検査で拒まれる。書くときは一時ファイルに書いて fsync し、リネームで差し替える。
 */
public final class CountFile {

  private static final int BUFFER = 1 << 16;

  private CountFile() {}

  public static void write(Path file, CountedHands hands) throws IOException {
    Path absolute = file.toAbsolutePath();
    Files.createDirectories(absolute.getParent());
    Path temporary = absolute.resolveSibling(absolute.getFileName() + ".tmp");
    try {
      try (DataOutputStream out = new DataOutputStream(
          new BufferedOutputStream(new GZIPOutputStream(Files.newOutputStream(temporary), BUFFER), BUFFER))) {
        encode(out, hands);
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

  /** 読めないとき、途中で切れているとき、中身が決まりを破るときは IOException を投げる。 */
  public static CountedHands read(Path file) throws IOException {
    try (DataInputStream in = new DataInputStream(
        new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      CountedHands hands = decode(in);
      if (in.read() != -1) {
        throw new IOException("Trailing bytes after the last game of " + file);
      }
      return hands;
    } catch (EOFException truncated) {
      throw new IOException("Truncated counts " + file, truncated);
    } catch (IllegalArgumentException | IllegalStateException malformed) {
      throw new IOException("Malformed counts " + file + ": " + malformed.getMessage(), malformed);
    }
  }

  private static void encode(DataOutputStream out, CountedHands hands) throws IOException {
    Variant variant = hands.variant();
    int seats = variant.seats();
    out.writeUTF(variant.name());
    out.writeInt(hands.cohort());
    out.writeDouble(hands.exploreTemperature());
    PhiStamp phi = hands.phi();
    out.writeUTF(phi.mode());
    out.writeDouble(phi.cutoff());
    out.writeInt(hands.games());
    for (int game = 0; game < hands.games(); game++) {
      out.writeLong(hands.gameId(game));
      out.writeShort(hands.firstHand(game + 1) - hands.firstHand(game));
      for (int hand = hands.firstHand(game); hand < hands.firstHand(game + 1); hand++) {
        out.writeByte(hands.round(hand));
        out.writeShort(hands.honba(hand));
        out.writeShort(hands.kyotaku(hand));
        for (int seat = 0; seat < seats; seat++) {
          out.writeInt(hands.score(hand, seat));
        }
        out.writeLong(hands.code(hand));
      }
    }
  }

  private static CountedHands decode(DataInputStream in) throws IOException {
    Variant variant = Variant.valueOf(in.readUTF());
    int seats = variant.seats();
    int cohort = in.readInt();
    double exploreTemperature = in.readDouble();
    String mode = in.readUTF();
    double cutoff = in.readDouble();
    PhiStamp phi = new PhiStamp(mode, cutoff);
    int games = in.readInt();
    if (games < 0) {
      throw new IllegalArgumentException("Negative game count " + games);
    }
    CountedHands.Builder builder = CountedHands.builder(variant, cohort, exploreTemperature, phi);
    TableState start = new TableState(variant);
    int[] scores = new int[seats];
    long previous = 0;
    for (int game = 0; game < games; game++) {
      long gameId = in.readLong();
      if (game > 0 && gameId <= previous) {
        throw new IllegalArgumentException("Game " + gameId + " is out of order after game " + previous);
      }
      previous = gameId;
      int count = in.readUnsignedShort();
      if (count == 0) {
        throw new IllegalArgumentException("Game " + gameId + " has no hands");
      }
      for (int hand = 0; hand < count; hand++) {
        int round = in.readUnsignedByte();
        int honba = in.readUnsignedShort();
        int kyotaku = in.readUnsignedShort();
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] = in.readInt();
        }
        long code = in.readLong();
        start.set(round, honba, kyotaku, scores);
        builder.add(gameId, start, code);
      }
    }
    return builder.build();
  }
}
