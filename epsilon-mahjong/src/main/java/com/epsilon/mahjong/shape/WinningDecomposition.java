package com.epsilon.mahjong.shape;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.shape.HandShape.Block;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Objects;

/**
 * 純手牌を通常の和了の形（雀頭 1 つと面子）に分ける方法を、形の表を使って 1 つずつ数え上げる。手全体の分解は、ブロックごとの
 * 分解の組み合わせ。萬・筒・索・字のブロックを桁と見て、字牌のブロックを一番下の桁にして繰り上げながら数える。使い回す可変の
 * オブジェクトで、同時に使えるのは 1 スレッドだけ。メモリは確保しない。
 *
 * <p>ここでの面子は純手牌の中の面子だけで、副露と暗槓は入らない。面子は種類の昇順に並び、同じ種類では刻子が順子より先。分解を
 * 読むメソッドは、{@link #next} が true を返した後の分解を返す。
 */
public final class WinningDecomposition {

  /** その分解で、和了牌がどの待ちの形を完成させたか。 */
  public enum Wait {
    /** 雀頭の単騎待ち。 */
    TANKI,
    /** 刻子の双碰待ち。 */
    SHANPON,
    RYANMEN,
    /** 順子の真ん中を待つ嵌張。 */
    KANCHAN,
    /** 12 で 3、89 で 7 を待つ辺張。 */
    PENCHAN;

    /** {@link WinningDecomposition#waits} が返す値の中で、この形を表すビット。 */
    public int bit() {
      return 1 << ordinal();
    }
  }

  private static final Block[] BLOCKS = Block.values();

  /** 色の中で最後の順子（789）が始まる番号（0 始まり）。 */
  private static final int LAST_RUN_START = Tiles.SUIT_SIZE - HandShape.MELD_SIZE;

  /** ブロックごとに、分解の番号の範囲と、次に読む分解。 */
  private final int[] start = new int[BLOCKS.length];
  private final int[] end = new int[BLOCKS.length];
  private final int[] next = new int[BLOCKS.length];

  /** 面子ごとに、先頭の種類と、順子かどうか。 */
  private final int[] meldKinds = new int[Variant.MAX_MELDS];
  private final boolean[] meldRuns = new boolean[Variant.MAX_MELDS];

  private boolean more;
  private int pair;
  private int meldCount;
  private long triplets;
  private long runs;

  /** 14 枚の手の分解を、最初から数え始める。手は変えない。 */
  public void bind(HandShape hand) {
    if (hand.size() != HandShape.MAX_SIZE) {
      throw new IllegalArgumentException("A decomposition needs a hand of 14 tiles: " + hand);
    }
    load(hand, -1);
  }

  /**
   * 13 枚の手に和了牌を 1 枚足した手の分解を、最初から数え始める。手は変えない。
   *
   * @param winningKind 足す和了牌の種類
   */
  public void bind(HandShape hand, int winningKind) {
    if (hand.size() != HandShape.MAX_SIZE - 1 || hand.count(winningKind) == Tiles.COPIES) {
      throw new IllegalArgumentException("Cannot add kind " + winningKind + " to " + hand);
    }
    load(hand, winningKind);
  }

  /** 次の分解を読む。もう無ければ false。 */
  public boolean next() {
    if (!more) {
      return false;
    }
    meldCount = 0;
    triplets = 0;
    runs = 0;
    for (Block block : BLOCKS) {
      int decomposition = next[block.ordinal()];
      if (ShapeTables.hasPair(decomposition)) {
        pair = block.first() + ShapeTables.pair(decomposition);
      }
      for (int slot = 0; slot < Variant.MAX_MELDS; slot++) {
        int local = ShapeTables.meldStart(decomposition, slot);
        if (local < 0) {
          break;
        }
        boolean run = ShapeTables.meldRun(decomposition, slot);
        int kind = block.first() + local;
        meldKinds[meldCount] = kind;
        meldRuns[meldCount] = run;
        meldCount++;
        if (run) {
          runs |= Tiles.bit(kind);
        } else {
          triplets |= Tiles.bit(kind);
        }
      }
    }
    advance();
    return true;
  }

  public int pair() {
    return pair;
  }

  /** 純手牌の中の面子の数（4 − 副露と暗槓の数）。 */
  public int meldCount() {
    return meldCount;
  }

  /** 指定した面子の牌の種類。刻子ならその牌、順子なら先頭の牌。 */
  public int meld(int index) {
    return meldKinds[Objects.checkIndex(index, meldCount)];
  }

  public boolean run(int index) {
    return meldRuns[Objects.checkIndex(index, meldCount)];
  }

  /** 刻子の種類の集合。 */
  public long triplets() {
    return triplets;
  }

  /** 順子の先頭の種類の集合。同じ順子が 2 つあっても、ビットは 1 つ。 */
  public long runs() {
    return runs;
  }

  /**
   * 和了牌がこの分解で完成させうる待ちの形を、{@link Wait#bit} の論理和で返す。和了牌を含む雀頭も面子も無ければ 0。
   *
   * @param kind 和了牌の種類
   */
  public int waits(int kind) {
    int waits = pair == kind ? Wait.TANKI.bit() : 0;
    for (int i = 0; i < meldCount; i++) {
      int first = meldKinds[i];
      if (!meldRuns[i]) {
        waits |= first == kind ? Wait.SHANPON.bit() : 0;
        continue;
      }
      int position = kind - first;
      if (position >= 0 && position < HandShape.MELD_SIZE) {
        waits |= runWait(Tiles.number(first), position).bit();
      }
    }
    return waits;
  }

  /**
   * 順子のどの位置に和了牌が入ったかから、待ちの形を決める。
   *
   * @param start 順子の先頭が色の中で何番目か
   * @param position 和了牌が順子の中で何番目か（0 始まり）
   */
  private static Wait runWait(int start, int position) {
    if (position == 1) {
      return Wait.KANCHAN;
    }
    boolean edge = position == 0 ? start == LAST_RUN_START : start == 0;
    return edge ? Wait.PENCHAN : Wait.RYANMEN;
  }

  /**
   * ブロックごとに分解の範囲を引く。雀頭がちょうど 1 つのブロックにある組み合わせだけを数える。
   *
   * @param added 足す種類。足さなければ −1
   */
  private void load(HandShape hand, int added) {
    int pairs = 0;
    more = true;
    for (Block block : BLOCKS) {
      int b = block.ordinal();
      int code = hand.code(block) + (added >= 0 && Block.of(added) == block ? Block.weight(added) : 0);
      int state = ShapeTables.state(block, code);
      start[b] = ShapeTables.first(state);
      end[b] = ShapeTables.first(state + 1);
      next[b] = start[b];
      if (start[b] == end[b]) {
        more = false;
      } else if (ShapeTables.hasPair(start[b])) {
        pairs++;
      }
    }
    more &= pairs == 1;
  }

  /** 次の分解へ進める。字牌のブロックから進め、尽きたブロックは最初に戻して前のブロックへ繰り上げる。全て尽きたら終わり。 */
  private void advance() {
    for (int b = BLOCKS.length - 1; b >= 0; b--) {
      if (++next[b] < end[b]) {
        return;
      }
      next[b] = start[b];
    }
    more = false;
  }
}
