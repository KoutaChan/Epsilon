package com.epsilon.grp.train;

import com.epsilon.grp.GrpPhase;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.IntPredicate;
import java.util.stream.IntStream;

/**
 * 半荘の列から、半荘が続く局末を全て並べたもの。Φ に依らない情報だけを持つ。局末 h は半荘の順に並び、同じ半荘の中では局の順に
 * 並ぶ。席は絶対席。
 *
 * <p>区分は、その局末で終わった局で決める。東 1 局を 0 とし、南場の最後の局まである。区分の数は南場の最後の局の番号 ＋ 1 で、
 * 四人打ちは E1〜E4・S1〜S3・S4+、三人打ちは E1〜E3・S1・S2・S3+ になる。南場の最後の局より後の局は、最後の区分に入る。段階は
 * 次の局の GrpPhase。今の順位と点差は、次の局の始まりの卓の点で決める。点差は 1 つ下の順位の席との差で、最下位だけは 1 つ上の
 * 席との差。最終順位と効用 u は最終の点から決める（同点なら起家に近い席が上）。
 *
 * <p>一部の局末だけを選んだ列（{@link #select}）も同じ形で持つ。選んだ列は、元の列と、局末ごとに元の列での番号を覚えている。
 */
public final class HandEnds {

  final Variant variant;
  final RankUtility utility;
  final int seats;
  final int groups;
  final long[] gameIds;

  /** 半荘ごとの最初の局末。最後の要素は局末の総数。 */
  final int[] firstHand;

  final GrpState[] states;
  final byte[] group;
  final byte[] phase;
  final byte[] dealer;

  /** 実際の順列の番号。順列は相対席で、番号の付け方は RankPermutations に従う。 */
  final byte[] permutation;

  /** 今の順位。配列の位置は h × S ＋ 席で、続く点差と最終順位も同じ。 */
  final byte[] rank;
  final int[] gap;
  final byte[] finalRank;

  /** 順位ごとの効用 u。 */
  final double[] utilities;

  /** 選んで作った列なら、選ぶ前の元の列。半荘から作った列では null。 */
  final HandEnds origin;

  /** 局末ごとの、元の列での番号。半荘から作った列では null。 */
  final int[] sources;

  private HandEnds(Variant variant, RankUtility utility, List<GameRecord> games) {
    this.variant = variant;
    this.utility = utility;
    origin = null;
    sources = null;
    seats = variant.seats();
    groups = variant.lastRound() + 1;
    int handEnds = 0;
    for (GameRecord game : games) {
      if (game.variant() != variant) {
        throw new IllegalArgumentException("A " + game.variant() + " game among " + variant + " games");
      }
      handEnds += game.handEnds().size();
    }
    gameIds = new long[games.size()];
    firstHand = new int[games.size() + 1];
    states = new GrpState[handEnds];
    group = new byte[handEnds];
    phase = new byte[handEnds];
    dealer = new byte[handEnds];
    permutation = new byte[handEnds];
    rank = new byte[handEnds * seats];
    gap = new int[handEnds * seats];
    finalRank = new byte[handEnds * seats];
    utilities = new double[seats];
    for (int place = 0; place < seats; place++) {
      utilities[place] = utility.value(variant, place);
    }
    int[] seatAt = new int[seats];
    int h = 0;
    for (int g = 0; g < games.size(); g++) {
      GameRecord game = games.get(g);
      gameIds[g] = game.gameId();
      firstHand[g] = h;
      for (int k = 0; k < game.handEnds().size(); k++, h++) {
        GrpState state = game.handEnds().get(k);
        states[h] = state;
        group[h] = (byte) Math.min(game.endedRound(k), variant.lastRound());
        phase[h] = (byte) GrpPhase.of(variant, state.round()).ordinal();
        dealer[h] = (byte) state.dealer();
        permutation[h] = (byte) game.realizedPermutation(k);
        for (int seat = 0; seat < seats; seat++) {
          seatAt[state.rank(seat)] = seat;
        }
        for (int seat = 0; seat < seats; seat++) {
          int now = state.rank(seat);
          int neighbour = seatAt[now < seats - 1 ? now + 1 : seats - 2];
          int j = h * seats + seat;
          rank[j] = (byte) now;
          gap[j] = Math.abs(state.score(seat) - state.score(neighbour));
          finalRank[j] = (byte) game.finalRank(seat);
        }
      }
    }
    firstHand[games.size()] = h;
  }

  /**
   * 元の列から局末を選んだ列を作る。
   *
   * @param chosen 選んだ局末の、元の列での番号（昇順）
   */
  private HandEnds(HandEnds origin, int[] chosen) {
    this.origin = origin;
    sources = chosen;
    variant = origin.variant;
    utility = origin.utility;
    seats = origin.seats;
    groups = origin.groups;
    utilities = origin.utilities;
    int handEnds = chosen.length;
    states = new GrpState[handEnds];
    group = new byte[handEnds];
    phase = new byte[handEnds];
    dealer = new byte[handEnds];
    permutation = new byte[handEnds];
    rank = new byte[handEnds * seats];
    gap = new int[handEnds * seats];
    finalRank = new byte[handEnds * seats];
    long[] ids = new long[origin.games()];
    int[] first = new int[origin.games() + 1];
    int games = 0;
    int index = 0;
    for (int game = 0; game < origin.games() && index < handEnds; game++) {
      int start = index;
      while (index < handEnds && chosen[index] < origin.firstHand[game + 1]) {
        int h = chosen[index];
        states[index] = origin.states[h];
        group[index] = origin.group[h];
        phase[index] = origin.phase[h];
        dealer[index] = origin.dealer[h];
        permutation[index] = origin.permutation[h];
        System.arraycopy(origin.rank, h * seats, rank, index * seats, seats);
        System.arraycopy(origin.gap, h * seats, gap, index * seats, seats);
        System.arraycopy(origin.finalRank, h * seats, finalRank, index * seats, seats);
        index++;
      }
      if (index > start) {
        ids[games] = origin.gameIds[game];
        first[games] = start;
        games++;
      }
    }
    gameIds = Arrays.copyOf(ids, games);
    firstHand = Arrays.copyOf(first, games + 1);
    firstHand[games] = handEnds;
  }

  /** 半荘の列から作る。ルールの違う半荘があれば IllegalArgumentException を投げる。 */
  public static HandEnds of(Variant variant, RankUtility utility, List<GameRecord> games) {
    return new HandEnds(Objects.requireNonNull(variant, "variant"), Objects.requireNonNull(utility, "utility"), games);
  }

  /**
   * 選んだ局末だけを、元の順に並べた列を作る。選んだ局末を 1 つも持たない半荘は入れない。区分・段階・今の順位・点差・最終順位は
   * 元の列のままなので、この列で測った指標は、元の列で同じ局末を選んで測った指標と同じになる。元の列について求めた値は、
   * {@code HandEndValues.select} でこの列の分だけコピーできる。
   *
   * @param selected 残す局末を、元の列での番号で選ぶ条件
   */
  public HandEnds select(IntPredicate selected) {
    return new HandEnds(this, IntStream.range(0, handEnds()).filter(selected).toArray());
  }

  public Variant variant() {
    return variant;
  }

  public RankUtility utility() {
    return utility;
  }

  public int seats() {
    return seats;
  }

  /** 区分の数（南場の最後の局の番号 ＋ 1）。 */
  public int groups() {
    return groups;
  }

  public int games() {
    return gameIds.length;
  }

  public int handEnds() {
    return states.length;
  }

  public long seatHands() {
    return (long) states.length * seats;
  }

  public long gameId(int game) {
    return gameIds[game];
  }

  /** 半荘の最初の局末。半荘 g の局末は、{@code firstHand(g)} から {@code firstHand(g + 1)} の手前まで。 */
  public int firstHand(int game) {
    return firstHand[game];
  }

  /** 局末が属する半荘の番号。 */
  public int game(int handEnd) {
    Objects.checkIndex(handEnd, states.length);
    int found = Arrays.binarySearch(firstHand, handEnd);
    if (found < 0) {
      return -found - 2;
    }
    while (firstHand[found + 1] == handEnd) {
      found++;
    }
    return found;
  }

  public GrpState state(int handEnd) {
    return states[handEnd];
  }

  /**
   * 半荘の始めから、その局末で終わった局までの各局の始まりの卓を、渡した履歴に積み直す。半荘の始めの卓に続けて、同じ半荘でそれより
   * 前の局末の卓（次の局の始まりの卓）を順に並べる。選んだ列では、元の列の半荘から作る。
   */
  public void history(int handEnd, RoundHistory into) {
    if (origin != null) {
      origin.history(sources[handEnd], into);
      return;
    }
    into.clear();
    into.add(new TableState(variant));
    for (int earlier = firstHand[game(handEnd)]; earlier < handEnd; earlier++) {
      into.add(states[earlier]);
    }
  }

  public int group(int handEnd) {
    return group[handEnd];
  }

  public int permutation(int handEnd) {
    return permutation[handEnd];
  }

  public int rank(int handEnd, int seat) {
    return rank[handEnd * seats + seat];
  }

  public int finalRank(int handEnd, int seat) {
    return finalRank[handEnd * seats + seat];
  }

  public double utility(int handEnd, int seat) {
    return utilities[finalRank[handEnd * seats + seat]];
  }

  /** 効用の最小値（最大値は {@link #highestUtility}）。 */
  double lowestUtility() {
    double lowest = Double.POSITIVE_INFINITY;
    for (double value : utilities) {
      lowest = Math.min(lowest, value);
    }
    return lowest;
  }

  double highestUtility() {
    double highest = Double.NEGATIVE_INFINITY;
    for (double value : utilities) {
      highest = Math.max(highest, value);
    }
    return highest;
  }

  /** その局末で、今その順位にいる席。 */
  int seatAt(int handEnd, int place) {
    for (int seat = 0; seat < seats; seat++) {
      if (rank[handEnd * seats + seat] == place) {
        return seat;
      }
    }
    throw new IllegalStateException("No seat holds rank " + place);
  }
}
