package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.Variant;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * 主経路の記録（prism の GameLog の形式 1、四人打ち）を読む。ベンチが実際の卓を読むのに使う。形式は GameRecordCodec とは違う。
 * gzip の中身は、先頭に i32 "PGL1"・i32 形式 1・i32 コホート・UTF Φ の項目の id・UTF その SHA-256・f64 ε_b・f64 探索の温度・
 * i64 × 2・i32 半荘の数が並ぶ。次に半荘ごとに、i64 半荘 ID・u16 局末の数・局末ごとの（u8 局・u16 本場・u16 供託・i32 × 4 点数）・
 * i32 × 4 終局の点数が並び、最後に i32 "END!" が来る。
 */
final class OldGameLog {

  private static final int MAGIC = 0x50474C31;
  private static final int END = 0x454E4421;
  private static final int BUFFER = 1 << 16;
  private static final Variant VARIANT = Variant.TENHOU_4P;

  private OldGameLog() {}

  /** ファイルを読んで 1 コホートの半荘を返す。形式が違えば IOException を投げる。 */
  static GameCohort read(Path file) throws IOException {
    try (DataInputStream in = new DataInputStream(
        new BufferedInputStream(new GZIPInputStream(Files.newInputStream(file), BUFFER), BUFFER))) {
      if (in.readInt() != MAGIC || in.readInt() != 1) {
        throw new IOException("Not a game log of format 1: " + file);
      }
      int cohort = in.readInt();
      String phiId = in.readUTF();
      in.readUTF();
      in.readDouble();
      in.readDouble();
      in.readLong();
      in.readLong();
      int count = in.readInt();
      List<GameRecord> games = new ArrayList<>(count);
      for (int game = 0; game < count; game++) {
        long gameId = in.readLong();
        int hands = in.readUnsignedShort();
        List<GrpState> handEnds = new ArrayList<>(hands);
        for (int hand = 0; hand < hands; hand++) {
          int round = in.readUnsignedByte();
          int honba = in.readUnsignedShort();
          int sticks = in.readUnsignedShort();
          handEnds.add(new GrpState(VARIANT, round, honba, sticks, scores(in)));
        }
        games.add(new GameRecord(VARIANT, gameId, handEnds, scores(in)));
      }
      if (in.readInt() != END) {
        throw new IOException("Game log without its end mark: " + file);
      }
      return new GameCohort(VARIANT, cohort, phiId, games);
    }
  }

  private static int[] scores(DataInputStream in) throws IOException {
    int[] scores = new int[VARIANT.seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = in.readInt();
    }
    return scores;
  }
}
