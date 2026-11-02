package com.epsilon.prism.input;

import com.epsilon.calculate.shape.HandShapeAnalysisBuffer;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeView;
import com.epsilon.core.Action;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import java.util.Arrays;

/**
 * 後の手の形（向聴 4 種・受け入れ・形の待ち・受け入れの生きた枚数と、1〜3 ツモ以内に受け入れを引く確率）。{@link HandShapeAnalyzer#analyzeInto}
 * を 1 回呼び、エンジンの解析（{@code HandAnalyzer}）と同じく、副露がある手では自分の副露と手で 5 枚目になる牌を受け入れと待ちから除く。引く確率は
 * 見えない牌から引く超幾何分布で、エンジンの改善率と同じ式・同じ順で計算する。
 *
 * <p>同じ後の手は 1 判断で 1 回だけ解析する。形は枚数と副露の牌だけで決まり、赤を見ない。だから次の遷移は枠（{@link #slot}）を共有する。
 *
 * <ul>
 *   <li>手番の打牌: 捨てる牌の 34 枠（双子・赤の選び・立直は枚数を変えない）。
 *   <li>判断の時点の手: 1 枠（{@link #CURRENT}。応答の見送りと 13 枚の判断の牌の欄）。
 *   <li>チー・ポンの後の打牌: 副露の種類と使う牌（ポン、チーの鳴いた位置 3 つ）× 捨てる牌の枠（{@link #callSlot}。赤の選びだけが違う鳴きが
 *       共有する）。
 * </ul>
 *
 * <p>ほかの遷移（大明槓・暗槓・加槓）は共有しない（{@link #UNSHARED}）。解析器には手の具体の形（{@link AfterHand#shape}）を渡す。割り当てを
 * しない。
 */
final class ShapeAnalysis {

  /** 判断の時点の手の枠。 */
  static final int CURRENT = Tile.NUM_TILE_TYPES;

  /** チー・ポンの後の打牌の最初の枠。 */
  private static final int CALL_FIRST = CURRENT + 1;

  /** チー・ポンの枠の組（ポン 1 つ ＋ チーの鳴いた牌の位置 3 つ）。 */
  private static final int CALL_GROUPS = 4;

  /** 共有しない遷移の枠。 */
  static final int UNSHARED = CALL_FIRST + CALL_GROUPS * Tile.NUM_TILE_TYPES;

  /** 引く確率を出すツモの数（1・2・3）。 */
  static final int DRAWS = 3;

  private static final int SLOTS = UNSHARED + 1;
  private static final long ALL_TILES = (1L << Tile.NUM_TILE_TYPES) - 1;

  private final HandShapeAnalyzer analyzer = new HandShapeAnalyzer();
  private final HandShapeAnalysisBuffer buffer = new HandShapeAnalysisBuffer();
  private final byte[] minimum = new byte[SLOTS];
  private final byte[] standard = new byte[SLOTS];
  private final byte[] chiitoi = new byte[SLOTS];
  private final byte[] kokushi = new byte[SLOTS];
  private final boolean[] special = new boolean[SLOTS];
  private final long[] ukeire = new long[SLOTS];
  private final long[] waits = new long[SLOTS];
  private final int[] liveUkeire = new int[SLOTS];
  private final float[] hits = new float[SLOTS * DRAWS];
  private final long[] analyzed = new long[(SLOTS + Long.SIZE - 1) / Long.SIZE];
  private int slot;

  /** 判断の初めに、共有の枠を捨てる。 */
  void reset() {
    Arrays.fill(analyzed, 0L);
  }

  /**
   * チー・ポンの後の打牌の枠を返す。チーは鳴いた牌と順子の最小の牌から使う手の牌が決まるので、赤の選びだけが違うチー・ポンは同じ枠になる。
   *
   * @param call チーかポンの合法手
   * @param calledTile 鳴いた牌（直前の打牌の牌）
   * @param discardTile 鳴いた後に捨てる牌
   * @return 枠
   * @throws IllegalArgumentException チーでもポンでもないとき
   */
  static int callSlot(Action call, int calledTile, int discardTile) {
    int group =
        switch (call.type()) {
          case PON -> 0;
          case CHI -> 1 + calledTile - call.chiSequenceBaseTileType();
          default -> throw new IllegalArgumentException("not a chi or pon: " + call);
        };
    return CALL_FIRST + group * Tile.NUM_TILE_TYPES + discardTile;
  }

  /**
   * 後の手を解析し、その枠を選ぶ（共有の枠に結果があれば解析しない）。
   *
   * @param hand 後の手（13 枚相当）
   * @param tiles 牌ごとの枚数（生きた枚数を数える）
   * @param slot 捨てる牌（手番の打牌）、{@link #CURRENT}、{@link #callSlot} の枠、または {@link #UNSHARED}
   */
  void analyze(AfterHand hand, TileCounts tiles, int slot) {
    this.slot = slot;
    int word = slot >>> 6;
    long bit = 1L << slot;
    if (slot != UNSHARED && (analyzed[word] & bit) != 0L) {
      return;
    }
    analyzer.analyzeInto(hand.shape(), buffer);
    long improving = buffer.ukeireTileTypeMask();
    long agari = buffer.agariTileTypeMask();
    if (hand.meldCount() != 0) {
      long possible = possibleDraws(hand.shape(), hand);
      improving &= possible;
      agari &= possible;
    }
    minimum[slot] = (byte) buffer.minimumShanten();
    standard[slot] = (byte) buffer.standardShanten();
    chiitoi[slot] = (byte) buffer.chiitoitsuShanten();
    kokushi[slot] = (byte) buffer.kokushiShanten();
    special[slot] = buffer.specialHandsAvailable();
    ukeire[slot] = improving;
    waits[slot] = agari;
    int live = tiles.liveCopies(improving);
    liveUkeire[slot] = live;
    loadHits(live, tiles.totalUnseen());
    if (slot != UNSHARED) {
      analyzed[word] |= bit;
    }
  }

  /** 選んだ枠（{@link #analyze} に渡した枠）。 */
  int slot() {
    return slot;
  }

  /** 最小の向聴。 */
  int minimum() {
    return minimum[slot];
  }

  /** 通常形の向聴。 */
  int standard() {
    return standard[slot];
  }

  /** 七対子の向聴（{@link #special} のときだけ意味を持つ）。 */
  int chiitoi() {
    return chiitoi[slot];
  }

  /** 国士の向聴（{@link #special} のときだけ意味を持つ）。 */
  int kokushi() {
    return kokushi[slot];
  }

  /** 副露がなく、七対子と国士を数える手。 */
  boolean special() {
    return special[slot];
  }

  /** 受け入れの牌（5 枚目になる牌を除く）。 */
  long ukeire() {
    return ukeire[slot];
  }

  /** 形の待ちの牌（13 枚相当で聴牌のときだけ。5 枚目になる牌を除く）。 */
  long waits() {
    return waits[slot];
  }

  /** 受け入れの生きた枚数。 */
  int liveUkeire() {
    return liveUkeire[slot];
  }

  /** draws（1〜{@value #DRAWS}）ツモ以内に受け入れを引く確率。 */
  float hitWithin(int draws) {
    return hits[slot * DRAWS + draws - 1];
  }

  /**
   * 1〜3 ツモ以内に受け入れを引く確率を書く。ツモの数ごとに初めから掛け直すエンジンの式と、掛ける順も値も同じになるように、1 ツモずつ掛けて
   * いく。
   */
  private void loadHits(int targets, int total) {
    double miss = 1.0;
    boolean certain = false;
    for (int draw = 0; draw < DRAWS; draw++) {
      float hit = 0f;
      if (targets != 0 && total != 0) {
        if (!certain && draw < total) {
          int remaining = total - targets - draw;
          if (remaining <= 0) {
            certain = true;
          } else {
            miss *= remaining / (double) (total - draw);
          }
        }
        hit = certain ? 1f : (float) (1.0 - miss);
      }
      hits[slot * DRAWS + draw] = hit;
    }
  }

  /**
   * 手の枚数 {@code shape} と副露 {@code hand} を合わせて 5 枚目になる牌を除いたマスクを返す（エンジンの {@code PossibleDrawsLookup} と同じ
   * 決まり）。前線の探索では、引いて捨てた後の枚数（{@code shape}）に後の手の副露を合わせる。
   *
   * @param shape 手の枚数
   * @param hand 副露を読む手
   * @return 引ける牌のマスク
   */
  static long possibleDraws(HandShapeView shape, AfterHand hand) {
    long possible = ALL_TILES;
    for (long pairs = shape.concealedPairTileTypeMask(); pairs != 0L; pairs &= pairs - 1) {
      int tile = Long.numberOfTrailingZeros(pairs);
      if (shape.count(tile) >= Tile.TILES_PER_TYPE) {
        possible &= ~(1L << tile);
      }
    }
    int meldCount = hand.meldCount();
    long fixed = 0L;
    for (int index = 0; index < meldCount; index++) {
      fixed |= hand.meld(index).tileTypeMask();
    }
    for (; fixed != 0L; fixed &= fixed - 1) {
      int tile = Long.numberOfTrailingZeros(fixed);
      long bit = 1L << tile;
      int owned = shape.count(tile);
      for (int index = 0; index < meldCount; index++) {
        Meld meld = hand.meld(index);
        if ((meld.tileTypeMask() & bit) != 0L) {
          owned += meld.type() == Meld.Type.CHI ? 1 : meld.size();
        }
      }
      if (owned >= Tile.TILES_PER_TYPE) {
        possible &= ~bit;
      }
    }
    return possible;
  }
}
