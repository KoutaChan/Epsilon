package com.epsilon.grp;

import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;

/**
 * FinalRound の作業領域。次のものを持つ: 2 つのネットワークの作業領域、精算と局の進行に使う卓と結果、葉のネットワークの値のメモ、
 * 数え上げの回数、最後に求めた卓で半荘が終わる組の確率の和。メモは GrpKey のキーで引くダイレクトマップ方式。葉のネットワークは
 * 変わらないので、メモに当たっても外れても値はビット単位で同じ。FinalRound ごとに作り、同時に使えるのは 1 スレッドだけ。
 */
public final class FinalRoundWorkspace {

  /** 葉のメモの既定のスロットの数。 */
  public static final int LEAF_SLOTS = 1 << 10;

  final FinalRound finalRound;
  final OutcomeWorkspace outcomes;
  final GrpWorkspace leaf;

  /** GrpState をコピーした卓。 */
  final TableState start;

  final TableState after;
  final RoundOutcome outcome;
  final int[] delta;
  final int[] scores;

  /** 長さ S! の配列。葉のネットワークの出力と、Φ を求めるときの p(π|s)。 */
  final double[] leafOut;
  final double[] mixture;

  /** 空のスロットのキーは GrpKey.NONE。値は [スロット × S! ＋ 順列] に並ぶ。 */
  final long[] memoKeys;
  final double[] memoValues;
  final int memoMask;

  /**
   * 今の終わり方で、立直の有無を確率に入れる席の数。続く 3 つの配列には、その i 番目の席が立直した確率、立直していない確率、
   * 絶対席のビットを置く。
   */
  int factors;
  final double[] declared;
  final double[] quiet;
  final int[] riichiBits;

  long settled;
  long terminalLeaves;
  long networkLeaves;
  long memoHits;
  double terminalMass;

  private FinalRoundWorkspace(FinalRound finalRound, OutcomeWorkspace outcomes, GrpWorkspace leaf, int leafSlots) {
    if (leafSlots < 1 || Integer.bitCount(leafSlots) != 1) {
      throw new IllegalArgumentException("Leaf memo slots must be a power of two: " + leafSlots);
    }
    this.finalRound = finalRound;
    this.outcomes = outcomes;
    this.leaf = leaf;
    Variant variant = finalRound.variant();
    int seats = variant.seats();
    int count = RankPermutations.of(variant).count();
    start = new TableState(variant);
    after = new TableState(variant);
    outcome = new RoundOutcome(variant);
    delta = new int[seats];
    scores = new int[seats];
    leafOut = new double[count];
    mixture = new double[count];
    memoKeys = new long[leafSlots];
    Arrays.fill(memoKeys, GrpKey.NONE);
    memoValues = new double[leafSlots * count];
    memoMask = leafSlots - 1;
    declared = new double[seats];
    quiet = new double[seats];
    riichiBits = new int[seats];
  }

  /** 葉のメモは既定のスロットの数（{@link #LEAF_SLOTS}）で作る。 */
  public static FinalRoundWorkspace of(FinalRound finalRound) {
    return of(finalRound, LEAF_SLOTS);
  }

  /** @param leafSlots 葉のメモのスロットの数 */
  public static FinalRoundWorkspace of(FinalRound finalRound, int leafSlots) {
    return new FinalRoundWorkspace(
        finalRound,
        OutcomeWorkspace.of(finalRound.outcomes().shape()),
        GrpWorkspace.of(finalRound.leaf().shape()),
        leafSlots);
  }

  /** 葉のメモのスロットの数。 */
  public int leafSlots() {
    return memoKeys.length;
  }

  /** この作業領域を作ってから、精算して局を進めた回数。 */
  public long settled() {
    return settled;
  }

  /** 半荘が終わった葉の数。 */
  public long terminalLeaves() {
    return terminalLeaves;
  }

  /** 葉のネットワークを呼んだ回数。メモに無かった卓とキーに収まらない卓で呼んだ分を数え、打ち切った確率を埋める呼び出しは数えない。 */
  public long networkLeaves() {
    return networkLeaves;
  }

  /** 葉のネットワークの値をメモから読んだ回数。 */
  public long memoHits() {
    return memoHits;
  }

  /**
   * 最後に求めた卓の p のうち、その局で半荘が終わる組に置いた確率の和。値が厳密に求まった部分にあたる。打ち切った組の確率と、
   * 半荘が続く組の確率は入らない。数え上げない卓では 0。
   */
  public double terminalMass() {
    return terminalMass;
  }

  /**
   * 渡した卓を作業用の卓にコピーする。ルールが違えば IllegalArgumentException を投げる。
   *
   * @return コピーした卓。次の呼び出しまで有効
   */
  TableState load(GrpState state) {
    state.copyInto(start);
    return start;
  }
}
