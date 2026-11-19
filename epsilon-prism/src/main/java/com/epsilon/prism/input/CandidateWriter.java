package com.epsilon.prism.input;

import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.CandidateCodes.Digit;
import com.epsilon.prism.input.CandidateCodes.Slot;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.TileField;
import java.nio.ByteBuffer;
import java.util.stream.IntStream;

/**
 * 候補の牌コードと、局面に共通する区分の桁を合わせて書く。
 *
 * <p>区分の符号は、牌の公開の値（判断ごとに 1 つ）と、遷移の牌のコードの桁で決まる。席に依らない区分は受け入れの桁を、席の区分は
 * 待ちと打った牌の桁を読み、枚数の桁は読まない。判断の手牌のコードはどの桁も 0 なので、手牌のコードで作った区分の行を
 * {@link #prepare} で 1 回作り、遷移ごとにはそれを写して、遷移のコードを書いたときに桁が立った牌（{@link #mark}）の区分だけを書き直す。
 */
final class CandidateWriter {
  private final int tiles;
  private final int keys;
  private final int tileWidth;
  private final int seatWidth;
  private final int riverColumn;
  private final int safetyColumn;
  private final int riichiColumn;
  private final Slot[] slots;
  private final byte[][] publicCodes;
  private final byte[][] transitionCodes;
  private final byte[] row;

  /** 判断の手牌のコードで作った区分の行（キーの列の後ろだけを使う）。 */
  private final byte[] handRow;

  /** 遷移ごとに、受け入れの桁が立った牌と、待ちか打った牌の桁が立った牌（牌のトークンのビット）。 */
  private final long[] publicMarks = new long[Limits.MAX_TRANSITIONS];

  private final long[] seatMarks = new long[Limits.MAX_TRANSITIONS];

  /** 牌のコードごとの、区分が読む桁（{@link #UKEIRE}・{@link #WAIT}・{@link #DISCARDED} のビット）。 */
  private final byte[] read = new byte[256];

  /** 区分ごとの、符号を書く最初の列。 */
  private final int[] columns;

  /** 席に依らない区分と、席の区分の番号。 */
  private final int[] publicSlots;

  private final int[] seatSlots;

  private static final int UKEIRE = 1;
  private static final int WAIT = 2;
  private static final int DISCARDED = 4;

  CandidateWriter(InputFormat format) {
    tiles = format.tiles();
    keys = format.fixedTokens();
    tileWidth = format.tile().width();
    seatWidth = format.seat().width();
    riverColumn = format.tile().column(TileField.OWN_RIVER);
    safetyColumn = format.tile().column(TileField.SAFETY, 0);
    riichiColumn = format.seat().column(SeatField.RIICHI);
    var layout = format.candidateCodes();
    slots = layout.slots().toArray(Slot[]::new);
    row = new byte[layout.codeWidth()];
    handRow = new byte[layout.codeWidth()];
    publicCodes = new byte[slots.length][tiles];
    transitionCodes = new byte[slots.length][256];
    var key = format.tileKeyCode();
    var digits = key.space();
    for (int s = 0; s < slots.length; s++) {
      var slot = slots[s];
      for (int code = 0; code < digits.size(); code++) {
        int upper;
        if (slot.seat() < 0) {
          upper = slot.space().part(Digit.UKEIRE, digits.digit(code, TileKeyCode.Digit.UKEIRE));
        } else {
          int wait = digits.digit(code, TileKeyCode.Digit.WAIT_CLASS) == 0 ? 0 : 1;
          upper =
              slot.space().part(Digit.WAIT, wait)
                  + slot.space()
                      .part(Digit.DISCARDED, digits.digit(code, TileKeyCode.Digit.DISCARDED));
        }
        transitionCodes[s][key.first() + code] = (byte) upper;
      }
    }
    for (int code = 0; code < digits.size(); code++) {
      int ukeire = digits.digit(code, TileKeyCode.Digit.UKEIRE);
      int discarded = digits.digit(code, TileKeyCode.Digit.DISCARDED);
      if (ukeire > 1 || discarded > 1) {
        throw new IllegalStateException(
            "candidate digits must be flags: " + ukeire + ", " + discarded);
      }
      int wait = digits.digit(code, TileKeyCode.Digit.WAIT_CLASS) == 0 ? 0 : WAIT;
      read[key.first() + code] = (byte) (ukeire * UKEIRE | wait | discarded * DISCARDED);
    }
    columns = IntStream.range(0, slots.length).map(s -> slots[s].column()).toArray();
    publicSlots = IntStream.range(0, slots.length).filter(s -> slots[s].seat() < 0).toArray();
    seatSlots = IntStream.range(0, slots.length).filter(s -> slots[s].seat() >= 0).toArray();
  }

  /**
   * 判断に共通する牌の公開の値と、判断の手牌のコードで作った区分の行を作る。
   *
   * @param handCodes 判断の手牌の牌のコード（牌のトークンの順。遷移のコードの下書きになるもの）
   * @throws IllegalStateException 手牌のコードに区分が読む桁が立っているとき
   */
  void prepare(RowFacts facts, byte[] tileDraft, byte[] seatDraft, byte[] handCodes) {
    publicCodes(facts, tileDraft, seatDraft);
    for (int t = 0; t < tiles; t++) {
      if (read[Byte.toUnsignedInt(handCodes[t])] != 0) {
        throw new IllegalStateException(
            "hand code of tile " + t + " carries candidate digits: " + handCodes[t]);
      }
    }
    for (int s = 0; s < slots.length; s++) {
      byte[] lower = publicCodes[s];
      byte[] upper = transitionCodes[s];
      for (int t = 0; t < tiles; t++) {
        handRow[columns[s] + t] = (byte) (lower[t] + upper[Byte.toUnsignedInt(handCodes[t])]);
      }
    }
  }

  /** 遷移のコードを書き始める。桁が立った牌の印を消す。 */
  void start(int transition) {
    publicMarks[transition] = 0L;
    seatMarks[transition] = 0L;
  }

  /** 遷移の牌のコードを書いた。区分が読む桁が立っていれば、その牌に印を付ける。 */
  void mark(int transition, int tile, int code) {
    int digits = read[code];
    publicMarks[transition] |= (long) (digits & UKEIRE) << tile;
    seatMarks[transition] |= (long) ((digits >> 1 | digits >> 2) & 1) << tile;
  }

  private void publicCodes(RowFacts facts, byte[] tileDraft, byte[] seatDraft) {
    for (int s = 0; s < slots.length; s++) {
      var slot = slots[s];
      byte[] target = publicCodes[s];
      switch (slot.segment()) {
        case RIVER -> {
          for (int t = 0; t < tiles; t++) target[t] = tileDraft[t * tileWidth + riverColumn];
        }
        case DORA -> {
          for (int t = 0; t < tiles; t++) target[t] = (byte) facts.tiles.dora(facts.physicalOf[t]);
        }
        case SEAT -> {
          int seat = slot.seat();
          int riichi = seatDraft[seat * seatWidth + riichiColumn] > 1 ? 1 : 0;
          int weight = slot.space().weight(Digit.SAFETY);
          if (seat == 0) {
            long river = facts.safety.mask(0, TileSafety.Safety.GENBUTSU);
            long after = facts.safety.mask(0, TileSafety.Safety.PASSED_AFTER_RIICHI);
            long cycle = facts.safety.mask(0, TileSafety.Safety.PASSED_THIS_CYCLE);
            for (int t = 0; t < tiles; t++) {
              long bit = Tiles.bit(facts.physicalOf[t]);
              int safety =
                  (bit & river) != 0 ? 1 : (bit & after) != 0 ? 2 : (bit & cycle) != 0 ? 3 : 0;
              target[t] = (byte) (safety * weight + riichi);
            }
          } else {
            int col = safetyColumn + seat - 1;
            for (int t = 0; t < tiles; t++)
              target[t] = (byte) (tileDraft[t * tileWidth + col] * weight + riichi);
          }
        }
      }
    }
  }

  /**
   * 遷移ごとに、基本のキーのコードと区分の符号を書く。区分の行は {@link #prepare} の行を写し、印の付いた牌の区分だけを遷移のコードで
   * 書き直す。
   */
  void write(ByteBuffer target, int start, byte[] basic, int transitions) {
    for (int tr = 0; tr < transitions; tr++) {
      int at = tr * keys;
      System.arraycopy(basic, at, row, 0, keys);
      System.arraycopy(handRow, keys, row, keys, row.length - keys);
      int tileCodes = at + keys - tiles;
      patch(publicSlots, publicMarks[tr], basic, tileCodes);
      patch(seatSlots, seatMarks[tr], basic, tileCodes);
      target.put(start + tr * row.length, row, 0, row.length);
    }
  }

  private void patch(int[] chosen, long marks, byte[] basic, int tileCodes) {
    for (long rest = marks; rest != 0L; rest &= rest - 1) {
      int t = Long.numberOfTrailingZeros(rest);
      int code = Byte.toUnsignedInt(basic[tileCodes + t]);
      for (int s : chosen) {
        row[columns[s] + t] = (byte) (publicCodes[s][t] + transitionCodes[s][code]);
      }
    }
  }
}
