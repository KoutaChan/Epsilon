package com.epsilon.prism.luck;

import com.epsilon.ai.model.MahjongTileRelation;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.prism.luck.LuckInputSchema.Token;
import java.nio.ByteBuffer;

/**
 * 運の網 Ω の関係の偏りの符号。トークンの対 (問い合わせ i, 鍵 j) ごとに 7 区画の 4 bit の符号を作り、下位の区画から 4 bit ずつ INT16 の 2 語に
 * 詰める（打つ網の背骨の {@code EncoderLayer} の表 [7 × 16, 頭] と同じ詰め方。区画 s の符号 c は表の行 16s + c を引く。Ω は自分の重みを持つ
 * ので、行の意味は打つ網と違ってよい）。
 *
 * <ol start="0">
 *   <li>型の対: 4·型(i) + 型(j)（局 0・席 1・牌 2）。
 *   <li>牌の関係の前半: 両方が牌で、{@link MahjongTileRelation} の類 c が 15 未満なら c + 1。
 *   <li>牌の関係の後半: 両方が牌で、c が 15 以上なら c − 14。
 *   <li>席: 両方が席なら 1 + (席(j) − 席(i)) mod 4、j だけが席なら 5 + 席(j)、i だけが席なら 9 + 席(i)。
 *   <li>所持と待ち（行ごと）: 席 s → 牌 t は 1 + k、牌 t → 席 s は 7 + k。k = min(s の手の t の枚数, 2) + 3·[t は s の待ち]。
 *   <li>河（行ごと）: 席 s → 牌 t は [t が s の河にある]、牌 t → 席 s は 2·[t が s の河にある]。
 *   <li>ドラの表示: 牌 i が牌 j を指す（j = ドラ(i)）なら 1、牌 j が牌 i を指すなら 2。
 * </ol>
 *
 * <p>牌は正準の色の順に並ぶので、牌どうしの区画（1・2・6）と型・席の区画は全ての行で同じ（{@link #BASE}）。行ごとに変わるのは席と牌の対の
 * 区画 4・5 だけで、それは上の語にある。当てはまらない区画の符号は 0。
 */
final class LuckRelation {

  /** 定義の記述（入力の指紋に入れる）。 */
  static final String DEFINITION =
      "type(4i+j) | tileRelation<15 | tileRelation>=15 | seat(1+(j-i)%4,5+j,9+i)"
          + " | hold+wait(1+k seat->tile,7+k tile->seat,k=min(n,2)+3w) | river(1,2) | doraOf(1,2)";

  /** 区画の数。 */
  static final int SEGMENTS = 7;

  /** 区画あたりの符号の数。 */
  static final int ENTRIES = 16;

  /** 1 対の INT16 の語の数。 */
  static final int WORDS = 2;

  /** 1 行の符号のバイト数。 */
  static final int ROW_BYTES = Token.ROW_TOKENS * Token.ROW_TOKENS * WORDS * Short.BYTES;

  private static final int TOKENS = Token.ROW_TOKENS;
  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int FIRST_HALF = ENTRIES - 1;
  private static final int HOLD_TO_TILE = 1;
  private static final int HOLD_TO_SEAT = 7;
  private static final int WAIT_STEP = 3;
  private static final int HELD_CAP = 2;

  /** 全ての行で同じ符号 [i, j, 語]（区画 4・5 は 0）。 */
  private static final short[] BASE = base();

  static {
    if (MahjongTileRelation.values().length > FIRST_HALF + ENTRIES - 1) {
      throw new IllegalStateException("tile relations do not fit two segments");
    }
    if (HOLD_TO_SEAT + HELD_CAP + WAIT_STEP >= ENTRIES) {
      throw new IllegalStateException("hold and wait codes do not fit one segment");
    }
  }

  private LuckRelation() {}

  /**
   * 1 行の符号を書く。
   *
   * @param target 書く先（ネイティブのバイト順）
   * @param at 行の先頭のバイト位置
   * @param held 相対席ごとの物理の牌の手の枚数
   * @param waits 相対席ごとの待ちの物理の牌のマスク
   * @param rivers 相対席ごとの河の物理の牌のマスク
   * @param suits 色の正準の写し
   */
  static void write(
      ByteBuffer target, int at, int[][] held, long[] waits, long[] rivers, LuckSuits suits) {
    for (int index = 0; index < BASE.length; index++) {
      target.putShort(at + index * Short.BYTES, BASE[index]);
    }
    int firstSeat = Token.SEAT.first();
    int firstTile = Token.TILE.first();
    for (int seat = 0; seat < SEATS; seat++) {
      int p = firstSeat + seat;
      for (int canonical = 0; canonical < TILES; canonical++) {
        int tile = suits.physical(canonical);
        long bit = 1L << tile;
        int k = Math.min(held[seat][tile], HELD_CAP) + ((waits[seat] & bit) != 0 ? WAIT_STEP : 0);
        int river = (rivers[seat] & bit) != 0 ? 1 : 0;
        int q = firstTile + canonical;
        target.putShort(at + high(p, q), (short) (HOLD_TO_TILE + k + ENTRIES * river));
        target.putShort(at + high(q, p), (short) (HOLD_TO_SEAT + k + ENTRIES * 2 * river));
      }
    }
  }

  /** 対 (i, j) の上の語のバイト位置（行の先頭から）。 */
  private static int high(int i, int j) {
    return ((i * TOKENS + j) * WORDS + 1) * Short.BYTES;
  }

  private static short[] base() {
    short[] codes = new short[TOKENS * TOKENS * WORDS];
    for (int i = 0; i < TOKENS; i++) {
      for (int j = 0; j < TOKENS; j++) {
        Token typeI = type(i);
        Token typeJ = type(j);
        int typePair = 4 * typeI.ordinal() + typeJ.ordinal();
        int tileFirst = 0;
        int tileSecond = 0;
        int dora = 0;
        if (typeI == Token.TILE && typeJ == Token.TILE) {
          int a = i - Token.TILE.first();
          int b = j - Token.TILE.first();
          int relation = MahjongTileRelation.between(a, b).ordinal();
          if (relation < FIRST_HALF) {
            tileFirst = relation + 1;
          } else {
            tileSecond = relation - (FIRST_HALF - 1);
          }
          if (Tile.doraFrom(a) == b) {
            dora = 1;
          } else if (Tile.doraFrom(b) == a) {
            dora = 2;
          }
        }
        int seat = 0;
        boolean seatI = typeI == Token.SEAT;
        boolean seatJ = typeJ == Token.SEAT;
        int ownerI = i - Token.SEAT.first();
        int ownerJ = j - Token.SEAT.first();
        if (seatI && seatJ) {
          seat = 1 + Math.floorMod(ownerJ - ownerI, SEATS);
        } else if (seatJ) {
          seat = 5 + ownerJ;
        } else if (seatI) {
          seat = 9 + ownerI;
        }
        int low = typePair + ENTRIES * tileFirst + ENTRIES * ENTRIES * tileSecond;
        low += ENTRIES * ENTRIES * ENTRIES * seat;
        int at = (i * TOKENS + j) * WORDS;
        codes[at] = (short) low;
        codes[at + 1] = (short) (ENTRIES * ENTRIES * dora);
      }
    }
    return codes;
  }

  /** 行の中の位置の型。 */
  private static Token type(int position) {
    if (position < Token.SEAT.first()) {
      return Token.ROUND;
    }
    return position < Token.TILE.first() ? Token.SEAT : Token.TILE;
  }
}
