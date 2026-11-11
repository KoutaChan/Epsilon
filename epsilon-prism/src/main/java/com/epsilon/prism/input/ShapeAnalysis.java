package com.epsilon.prism.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;

/**
 * 手を選んだ後の手牌の形を求める。求めるのは、4 種類の向聴、受け入れ、形の上の待ち、受け入れの残り枚数、1〜3 ツモ以内に受け入れを
 * 引く確率。形は {@link ShapeAnalyzer} で求める。面子を晒した手か北を抜いた手では、晒した面子・抜いた北と合わせて 5 枚目になる牌を
 * 受け入れと待ちから除く（{@link #possibleDraws}）。引く確率は、見えていない牌から引くとして超幾何分布で求める。
 *
 * <p>1 回の判断の中では、手を選んだ後の手が同じなら解析は 1 回だけにする。形は牌の枚数と面子だけで決まり、赤は関係しない。だから次の
 * 遷移は結果のスロット（{@link #slot}）を共有する。
 *
 * <ul>
 *   <li>手番の打牌: 捨てる牌の種類ごとに 34 スロット。同じ結果になる手どうしや、赤の選び方・立直の有無だけが違う手は、枚数が
 *       変わらないので同じスロットを使う。
 *   <li>判断の時点の手: 1 スロット（{@link #CURRENT}）。応答を見送るときと、13 枚の手で判断するときの牌のフィールドに使う。
 *   <li>チー・ポンの後の打牌: 面子の種類と使う牌（ポン、チーで鳴いた牌の位置 3 つ）× 捨てる牌（{@link #callSlot}）。赤の選び方だけが違う
 *       鳴きは同じスロットを使う。
 * </ul>
 *
 * <p>ほかの遷移（槓・北抜き）は共有しない（{@link #UNSHARED}）。メモリを割り当てない。
 */
public final class ShapeAnalysis {

  /** 判断の時点の手のスロット。 */
  public static final int CURRENT = Tiles.KINDS;

  /** チー・ポンの後の打牌の最初のスロット。 */
  private static final int CALL_FIRST = CURRENT + 1;

  /** チー・ポンのスロットの組の数（ポン 1 つ ＋ チーで鳴いた牌の位置 3 つ）。 */
  private static final int CALL_GROUPS = 1 + Meld.Type.CHI.tiles();

  /** 共有しない遷移のスロット。 */
  public static final int UNSHARED = CALL_FIRST + CALL_GROUPS * Tiles.KINDS;

  /** 引く確率を出すツモの数（1・2・3）。 */
  public static final int DRAWS = 3;

  private static final int SLOTS = UNSHARED + 1;

  private final ShapeAnalyzer analyzer;
  private final ShapeAnalyzer.Analysis result = new ShapeAnalyzer.Analysis();
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

  public ShapeAnalysis(Variant variant) {
    analyzer = new ShapeAnalyzer(variant);
  }

  /** 判断の初めに、共有していた結果を捨てる。 */
  public void reset() {
    Arrays.fill(analyzed, 0L);
  }

  /**
   * チー・ポンの後の打牌のスロットを返す。チーで手から出す牌は、鳴いた牌と順子のいちばん小さい牌で決まる。だから赤の選び方だけが
   * 違うチー・ポンは同じスロットになる。
   *
   * @param call チーかポンの合法手
   * @param calledTile 鳴いた牌（直前に捨てられた牌）
   * @param discardTile 鳴いた後に捨てる牌
   * @return スロット
   * @throws IllegalArgumentException チーでもポンでもないとき
   */
  public static int callSlot(Action call, int calledTile, int discardTile) {
    int group =
        switch (call.type()) {
          case PON -> 0;
          case CHI -> 1 + calledTile - call.base();
          default -> throw new IllegalArgumentException("not a chi or pon: " + call);
        };
    return CALL_FIRST + group * Tiles.KINDS + discardTile;
  }

  /**
   * 手を選んだ後の手を解析し、そのスロットを選ぶ。共有するスロットに結果が既にあれば、解析しない。
   *
   * @param hand 手を選んだ後の手（13 枚相当）
   * @param tiles 種類ごとの枚数（受け入れの残り枚数を数えるのに使う）
   * @param slot 捨てる牌の種類（手番の打牌）、{@link #CURRENT}、{@link #callSlot} が返すスロット、または {@link #UNSHARED}
   */
  public void analyze(AfterHand hand, TileCounts tiles, int slot) {
    this.slot = slot;
    int word = slot >>> 6;
    long bit = 1L << slot;
    if (slot != UNSHARED && (analyzed[word] & bit) != 0L) {
      return;
    }
    HandShape shape = hand.shape();
    analyzer.analyze(shape, result);
    int shanten = result.shanten();
    long improving = result.improving();
    long agari = shanten == 0 ? improving : 0L;
    if (hand.meldCount() != 0 || hand.nuki() != 0) {
      long possible = possibleDraws(shape, hand);
      improving &= possible;
      agari &= possible;
    }
    boolean concealed = shape.melds() == 0;
    minimum[slot] = (byte) shanten;
    standard[slot] = (byte) result.standardShanten();
    chiitoi[slot] = (byte) (concealed ? result.chiitoitsuShanten() : 0);
    kokushi[slot] = (byte) (concealed ? result.kokushiShanten() : 0);
    special[slot] = concealed;
    ukeire[slot] = improving;
    waits[slot] = agari;
    int live = tiles.live(improving);
    liveUkeire[slot] = live;
    loadHits(live, tiles.totalUnseen());
    if (slot != UNSHARED) {
      analyzed[word] |= bit;
    }
  }

  /** {@link #analyze} で選んだスロット。 */
  public int slot() {
    return slot;
  }

  /** 最小の向聴。 */
  public int minimum() {
    return minimum[slot];
  }

  /** 通常形の向聴。 */
  public int standard() {
    return standard[slot];
  }

  /** 七対子の向聴（{@link #special} が真のときだけ意味がある）。 */
  public int chiitoi() {
    return chiitoi[slot];
  }

  /** 国士の向聴（{@link #special} が真のときだけ意味がある）。 */
  public int kokushi() {
    return kokushi[slot];
  }

  /** 晒した面子がなく、七対子と国士の向聴も数える手か。 */
  public boolean special() {
    return special[slot];
  }

  /** 受け入れになる牌（5 枚目になる牌を除く）。 */
  public long ukeire() {
    return ukeire[slot];
  }

  /** 形の上の待ち牌。13 枚相当で聴牌のときだけ持ち、5 枚目になる牌を除く。 */
  public long waits() {
    return waits[slot];
  }

  /** 受け入れの残り枚数（見えていない枚数）。 */
  public int liveUkeire() {
    return liveUkeire[slot];
  }

  /** {@code draws} ツモ（1〜{@value #DRAWS}）以内に受け入れを引く確率。 */
  public float hitWithin(int draws) {
    return hits[slot * DRAWS + draws - 1];
  }

  /** 1〜3 ツモ以内に受け入れを引く確率を書く。引けない確率を 1 ツモずつ掛け合わせて求める。 */
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
   * 引ける牌のマスクを返す。手牌の枚数と晒した面子と抜いた北を合わせて、5 枚目になる牌を除く。フロンティアの探索では、引いて捨てた後の
   * 枚数と、手を選んだ後の手を渡す（面子と抜いた北はそちらから読む）。
   *
   * @param shape 手牌の種類ごとの枚数
   * @param hand 晒した面子と抜いた北の数を読む手
   * @return 引ける牌のマスク
   */
  public static long possibleDraws(HandShape shape, AfterHand hand) {
    long possible = Tiles.ALL_KINDS;
    for (long pairs = shape.pairs(); pairs != 0L; pairs &= pairs - 1) {
      int tile = Long.numberOfTrailingZeros(pairs);
      if (shape.count(tile) >= Tiles.COPIES) {
        possible &= ~Tiles.bit(tile);
      }
    }
    int meldCount = hand.meldCount();
    int nuki = hand.nuki();
    long fixed = nuki == 0 ? 0L : Tiles.bit(Tiles.PEI);
    for (int index = 0; index < meldCount; index++) {
      fixed |= hand.meld(index).kinds();
    }
    for (; fixed != 0L; fixed &= fixed - 1) {
      int tile = Long.numberOfTrailingZeros(fixed);
      long bit = Tiles.bit(tile);
      int owned = shape.count(tile) + (tile == Tiles.PEI ? nuki : 0);
      for (int index = 0; index < meldCount; index++) {
        Meld meld = hand.meld(index);
        if ((meld.kinds() & bit) != 0L) {
          owned += meld.type() == Meld.Type.CHI ? 1 : meld.size();
        }
      }
      if (owned >= Tiles.COPIES) {
        possible &= ~bit;
      }
    }
    return possible;
  }
}
