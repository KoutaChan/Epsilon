package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.RoundOutcome;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.IntPredicate;
import java.util.function.Supplier;
import java.util.stream.IntStream;

/**
 * 局末ごとに、次の局で実際に起きた結果を持つ。局末の卓から次の局の卓（その局で半荘が終わるなら最終の点）への遷移を
 * TransitionSearch で再現し、どの種類の結果で再現できたか（{@link Kind}）で局末を分ける。局の結果のネットワークが表せる結果で
 * 再現できた局末（種類が {@code OUTCOME} か {@code ENDING}）は、遷移を再現する結果を全て持つ。次の局の後の順位
 * （半荘が終わるなら最終順位）も持つ。結果の席は次局の親を 0 とする相対席で、順位の席は絶対席。局を数えたファイルから作るとき
 * （{@link #of(HandEnds, CountedHands)}）は、遷移を探さずに、記録した結果を読む。
 */
public final class NextHands {

  /** 並列に調べるとき、1 チャンクに入れる局末の数。 */
  private static final int CHUNK = 4_096;

  private static final Kind[] KINDS = Kind.values();

  /** 今の順位のまま終わる率の誤差を、分けて見る量の数。測る値の誤差、比べる値の誤差、その差の 3 つ。 */
  private static final int QUANTITIES = 3;

  /** 遷移をどの種類の結果で再現できたか。上の種類から順に調べ、再現できた所で止める。 */
  public enum Kind {
    /** 局の結果のネットワークが表せる結果。 */
    OUTCOME,
    /** 包のある役満の和了。 */
    PAO,
    /** 流し満貫。 */
    NAGASHI,
    /**
     * 局の結果のネットワークが表せる結果で再現できるが、半荘が終わるかどうかだけは、記録を作ったエンジンの判定に合わせる必要がある。
     * そのエンジンは、和了者が場の供託を受け取る前の点で終わるかを判定し、西 4 局はどの終わり方でも半荘を終える。持つ結果は、その
     * エンジンの判定で再現した結果。
     */
    ENDING,
    /** どれでも再現できない。 */
    MISMATCH
  }

  /**
   * 種類 1 つについて、局末の数と、順位が入れ替わった率をまとめたもの。率は局末あたりか（席, 局末）の組あたりで、標準誤差は
   * 半荘をクラスタとして求める。
   *
   * @param kind 種類
   * @param handEnds 局末の数
   * @param share 選んだ局末のうち、この種類の局末の割合
   * @param nextChanged 次の局の後の順位が、今の順位と 1 席でも違う局末の率
   * @param finalChanged 最終順位が、今の順位と 1 席でも違う局末の率
   * @param seatChanged 最終順位が今の順位と違う席の率。率は（席, 局末）の組あたり
   */
  public record KindCount(
      Kind kind, long handEnds, double share, Paired nextChanged, Paired finalChanged, Paired seatChanged) {}

  /**
   * （席, 局末）の組ごとの量 1 つの平均を、表せる局末と表せない局末に分けたもの。量は、誤差か、2 つの値の誤差の差。
   * 表せる局末とは、次の局への遷移を、局の結果のネットワークが出す結果と Progression の判定で表せる局末で、種類が
   * {@code OUTCOME} のもの。それ以外を表せない局末と呼ぶ。標準誤差は半荘をクラスタとして求める。
   *
   * @param all 全ての組での平均
   * @param represented 表せる局末での平均
   * @param missing 表せない局末での平均
   * @param explained 表せない局末があるせいで全体の平均がずれる分。全体の平均 − 表せる局末の平均 ＝
   *     m·(表せない局末の平均 − 表せる局末の平均)。標準誤差は線形化して求める
   */
  public record Split(Paired all, Paired represented, Paired missing, Paired explained) {}

  /**
   * 選んだ（席, 局末）の組について、今の順位のまま終わる率の誤差を、表せる局末と表せない局末に分けたもの。誤差は、今の順位の
   * まま終わったか（1 か 0）から、予測の確率を引いた値。局末を実際の結果で分けるので、分けた平均には分け方による偏りが入る。この
   * 偏りはどの値にもほぼ同じだけ入る。そこで、測る値に固有の分を見るために、同じ組での比べる値の誤差を引いた差も分ける。
   *
   * @param seatHands 選んだ（席, 局末）の組の数
   * @param missingShare 組のうち、表せない局末にあるものの割合 m
   * @param representedKept 表せる局末で、今の順位のまま終わった率
   * @param missingKept 表せない局末で、今の順位のまま終わった率
   * @param candidate 測る値の誤差
   * @param reference 比べる値の誤差
   * @param difference 測る値の誤差 − 比べる値の誤差（＝ 比べる値の予測 − 測る値の予測）
   */
  public record KeepSplit(
      long seatHands,
      double missingShare,
      double representedKept,
      double missingKept,
      Split candidate,
      Split reference,
      Split difference) {}

  private final HandEnds ends;
  private final byte[] kinds;

  /** 次の局で半荘が終わるか。 */
  private final boolean[] over;

  /** 次の局の後の順位。並びは [h × S ＋ 席]。 */
  private final byte[] ranksAfter;

  /** 局末ごとに、その局末の最初の結果の番号。最後の要素は結果の総数。 */
  private final int[] firstFound;

  /** 結果ごとに、終わり方、打点の番号 2 つ（無ければ −1）、立直した席のビット（相対席）を持つ。 */
  private final byte[] mains;
  private final byte[] bases;
  private final byte[] others;
  private final byte[] riichi;

  private NextHands(HandEnds ends, byte[] kinds, boolean[] over, byte[] ranksAfter, int[] firstFound, Found found) {
    this.ends = ends;
    this.kinds = kinds;
    this.over = over;
    this.ranksAfter = ranksAfter;
    this.firstFound = firstFound;
    mains = Arrays.copyOf(found.mains, found.size);
    bases = Arrays.copyOf(found.bases, found.size);
    others = Arrays.copyOf(found.others, found.size);
    riichi = Arrays.copyOf(found.riichi, found.size);
  }

  /**
   * 局末の列について、次の局で実際に起きた結果を調べる。局末はチャンクごとに並列に調べるが、結果は局末の順に並べるので、
   * スレッドの実行順に依らない。
   *
   * @param ends 半荘の列から作った局末の列か、そこから選んだ列（{@code HandEnds.select}）
   * @param games 局末の列を作った半荘（選んだ列なら、元の列を作った半荘）。並びも同じであること
   * @return 局末ごとの、次の局の結果
   * @throws IllegalArgumentException 半荘が局末の列の半荘と食い違うとき
   */
  public static NextHands of(HandEnds ends, List<GameRecord> games) {
    HandEnds root = ends.origin == null ? ends : ends.origin;
    if (games.size() != root.games()) {
      throw new IllegalArgumentException(games.size() + " games for hand ends of " + root.games() + " games");
    }
    for (int game = 0; game < games.size(); game++) {
      if (games.get(game).gameId() != root.gameIds[game]) {
        throw new IllegalArgumentException(
            "Game " + games.get(game).gameId() + " is not hand-end game " + root.gameIds[game]);
      }
    }
    int count = ends.handEnds();
    int seats = ends.seats;
    byte[] kinds = new byte[count];
    boolean[] over = new boolean[count];
    byte[] ranksAfter = new byte[count * seats];
    int[] foundCounts = new int[count];
    ThreadLocal<TransitionSearch> searches = ThreadLocal.withInitial(() -> new TransitionSearch(ends.variant));
    List<Found> chunks = IntStream.range(0, Math.ceilDiv(count, CHUNK))
        .parallel()
        .mapToObj(chunk -> {
          TransitionSearch search = searches.get();
          Found found = new Found();
          int end = Math.min(count, (chunk + 1) * CHUNK);
          for (int index = chunk * CHUNK; index < end; index++) {
            int h = ends.sources == null ? index : ends.sources[index];
            int game = root.game(h);
            GameRecord record = games.get(game);
            int next = h - root.firstHand[game] + 1;
            search.from(root.states[h]);
            int hits;
            if (next < record.handEnds().size()) {
              GrpState state = record.handEnds().get(next);
              hits = search.next(state);
              for (int seat = 0; seat < seats; seat++) {
                ranksAfter[index * seats + seat] = (byte) state.rank(seat);
              }
            } else {
              int[] finalScores = record.finalScores();
              hits = search.last(finalScores);
              over[index] = true;
              for (int seat = 0; seat < seats; seat++) {
                ranksAfter[index * seats + seat] = (byte) Placement.rank(finalScores, seat);
              }
            }
            kinds[index] = (byte) kind(search, hits).ordinal();
            int reproducing = search.found();
            foundCounts[index] = reproducing;
            for (int k = 0; k < reproducing; k++) {
              found.add(search.main(k), search.base(k), search.other(k), search.riichi(k));
            }
          }
          return found;
        })
        .toList();
    int[] firstFound = new int[count + 1];
    for (int index = 0; index < count; index++) {
      firstFound[index + 1] = firstFound[index] + foundCounts[index];
    }
    Found all = new Found();
    for (Found chunk : chunks) {
      all.append(chunk);
    }
    return new NextHands(ends, kinds, over, ranksAfter, firstFound, all);
  }

  /**
   * 局末の列について、次の局の結果を、局を数えたファイルから読む。半荘の k 番目の局末（0 から数える）の次の局は、ファイルの中で
   * 同じ半荘 ID を持つ半荘の k ＋ 1 番目の局（0 番目は東 1 局）とみなす。その局の始まりの卓は局末の卓と同じで、半荘の局の数は
   * 局末の数 ＋ 1 であること。
   *
   * <p>記録した結果を局の結果のネットワークが表せれば、種類は {@code OUTCOME} で、遷移を再現する結果はその 1 つだけ。表せなければ
   * 結果を持たない。そのときの種類は、包のある和了なら {@code PAO}、流し満貫なら {@code NAGASHI}、それ以外（荒牌で、立直した席が
   * 聴牌していない）なら {@code MISMATCH}。
   *
   * @param ends 半荘の列から作った局末の列か、そこから選んだ列（{@code HandEnds.select}）
   * @param counted 同じ半荘の局を数えたファイルの中身
   * @return 局末ごとの、次の局の結果
   * @throws IllegalArgumentException ファイルのルールが局末と違うとき、局末の半荘がファイルに無いとき、半荘の局の数か局の
   *     始まりの卓が局末と合わないとき
   */
  public static NextHands of(HandEnds ends, CountedHands counted) {
    if (counted.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + counted.variant() + " count file for " + ends.variant + " hand ends");
    }
    HandEnds root = ends.origin == null ? ends : ends.origin;
    int seats = ends.seats;
    long[] ids = new long[counted.games()];
    for (int game = 0; game < ids.length; game++) {
      ids[game] = counted.gameId(game);
    }
    LongIndex countedGames = LongIndex.of(ids);
    OutcomeCodes codes = new OutcomeCodes(ends.variant);
    RoundOutcome outcome = new RoundOutcome(ends.variant);
    int[] scores = new int[seats];
    int count = ends.handEnds();
    byte[] kinds = new byte[count];
    boolean[] over = new boolean[count];
    byte[] ranksAfter = new byte[count * seats];
    int[] firstFound = new int[count + 1];
    Found found = new Found();
    for (int index = 0; index < count; index++) {
      int h = ends.sources == null ? index : ends.sources[index];
      int game = root.game(h);
      long gameId = root.gameIds[game];
      int countedGame = countedGames.get(gameId);
      if (countedGame == LongIndex.ABSENT) {
        throw new IllegalArgumentException("Game " + gameId + " is not in the count file");
      }
      int handEnds = root.firstHand[game + 1] - root.firstHand[game];
      int first = counted.firstHand(countedGame);
      int hands = counted.firstHand(countedGame + 1) - first;
      if (hands != handEnds + 1) {
        throw new IllegalArgumentException(
            "Game " + gameId + " has " + hands + " counted hands for " + handEnds + " hand ends");
      }
      int hand = first + h - root.firstHand[game] + 1;
      if (!startsAt(counted, hand, root.states[h])) {
        throw new IllegalArgumentException("Counted hand " + hand + " of game " + gameId + " does not start at "
            + root.states[h]);
      }
      boolean last = hand == first + handEnds;
      over[index] = last;
      if (!last) {
        for (int seat = 0; seat < seats; seat++) {
          scores[seat] = counted.score(hand + 1, seat);
        }
      }
      for (int seat = 0; seat < seats; seat++) {
        ranksAfter[index * seats + seat] =
            (byte) (last ? root.finalRank[h * seats + seat] : Placement.rank(scores, seat));
      }
      long code = counted.code(hand);
      Kind kind;
      if (codes.decode(code)) {
        kind = Kind.OUTCOME;
        found.add(codes.main(), codes.base(), codes.other(), codes.riichi());
      } else {
        kind = unrepresented(code, outcome);
      }
      kinds[index] = (byte) kind.ordinal();
      firstFound[index + 1] = found.size;
    }
    return new NextHands(ends, kinds, over, ranksAfter, firstFound, found);
  }

  /** ファイルにある局の始まりの卓が、局末の卓と同じなら true。 */
  private static boolean startsAt(CountedHands counted, int hand, GrpState state) {
    if (counted.round(hand) != state.round()
        || counted.honba(hand) != state.honba()
        || counted.kyotaku(hand) != state.sticks()) {
      return false;
    }
    for (int seat = 0; seat < state.variant().seats(); seat++) {
      if (counted.score(hand, seat) != state.score(seat)) {
        return false;
      }
    }
    return true;
  }

  /**
   * 記録した結果のうち、局の結果のネットワークが表せないものについて、その種類を返す。
   *
   * @param code 局の結果の生の形のコード
   * @param scratch コードを展開する作業用の結果（中身は書き換わる）
   */
  private static Kind unrepresented(long code, RoundOutcome scratch) {
    HandResult.place(code, 0, scratch);
    if (scratch.kind() == EndKind.EXHAUSTIVE) {
      return scratch.nagashiMask() != 0 ? Kind.NAGASHI : Kind.MISMATCH;
    }
    for (int index = 0; index < scratch.winnerCount(); index++) {
      if (scratch.pao(index) != RoundOutcome.NO_SEAT) {
        return Kind.PAO;
      }
    }
    return Kind.MISMATCH;
  }

  public HandEnds ends() {
    return ends;
  }

  public Kind kind(int handEnd) {
    return KINDS[kinds[handEnd]];
  }

  /** 次の局で半荘が終わったなら true。記録の遷移で決まる。 */
  public boolean gameOver(int handEnd) {
    return over[handEnd];
  }

  /**
   * 遷移を再現する結果のうち、局の結果のネットワークが表せるものの数。種類が {@code ENDING} なら、半荘が終わるかを記録を作った
   * エンジンの判定で決めたときの数。ほかの種類（{@code PAO}・{@code NAGASHI}・{@code MISMATCH}）では 0。
   */
  public int found(int handEnd) {
    return firstFound[handEnd + 1] - firstFound[handEnd];
  }

  /** 再現する結果の、終わり方の番号（HandOutcomes の番号）。 */
  public int main(int handEnd, int index) {
    return mains[at(handEnd, index)];
  }

  /** 再現する結果の、和了者（ダブロンなら {@code HandOutcomes.winner} の席）の打点の番号。和了でなければ −1。 */
  public int base(int handEnd, int index) {
    return bases[at(handEnd, index)];
  }

  /** 再現する結果の、ダブロンの 2 人目の打点の番号。ほかは −1。 */
  public int other(int handEnd, int index) {
    return others[at(handEnd, index)];
  }

  /** 再現する結果の、立直が成立した席のビット（相対席）。 */
  public int riichi(int handEnd, int index) {
    return riichi[at(handEnd, index)];
  }

  /**
   * 局末ごとに log Σ_o q(o|s) を求める。o は遷移を再現する結果、q は渡したネットワーク。測らない局末（選ばなかった局末と、遷移を
   * 再現する結果を持たない局末）は NaN で、測った局末の値は必ず有限。局末を決まった数ずつのチャンクに分けて並列に求める。値は
   * 局末ごとに決まるので、スレッドの数にも求める順にも依らない。
   *
   * @param selected 測る局末を選ぶ関数（複数のスレッドから同時に呼ぶ）
   * @param network 測るネットワーク
   * @param workspaces 作業領域を作る関数。計算するスレッドごとに 1 度呼ぶ
   * @return 局末の順に並べた値
   * @throws IllegalArgumentException ネットワークのルールが局末と違うとき
   */
  double[] logLikelihoods(IntPredicate selected, OutcomeNetwork network, Supplier<OutcomeWorkspace> workspaces) {
    if (network.variant() != ends.variant) {
      throw new IllegalArgumentException("A " + network.variant() + " network for " + ends.variant + " hand ends");
    }
    HandOutcomes hands = HandOutcomes.of(ends.variant);
    int count = ends.handEnds();
    double[] out = new double[count];
    ThreadLocal<OutcomeWorkspace> locals = ThreadLocal.withInitial(workspaces);
    IntStream.range(0, Math.ceilDiv(count, CHUNK)).parallel().forEach(chunk -> {
      OutcomeWorkspace workspace = locals.get();
      short[] terms = new short[hands.maxTerms()];
      double[] logs = new double[1];
      int end = Math.min(count, (chunk + 1) * CHUNK);
      for (int h = chunk * CHUNK; h < end; h++) {
        int outcomes = firstFound[h + 1] - firstFound[h];
        if (outcomes == 0 || !selected.test(h)) {
          out[h] = Double.NaN;
          continue;
        }
        if (logs.length < outcomes) {
          logs = new double[2 * outcomes];
        }
        network.forward(ends.states[h], workspace);
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < outcomes; k++) {
          int at = firstFound[h] + k;
          int written = OutcomeRows.outcomeTerms(hands, mains[at], bases[at], others[at], riichi[at], terms, 0);
          double log = 0;
          for (int j = 0; j < written; j++) {
            log += workspace.term(terms[j]);
          }
          logs[k] = log;
          max = Math.max(max, log);
        }
        double total = 0;
        for (int k = 0; k < outcomes; k++) {
          total += StrictMath.exp(logs[k] - max);
        }
        out[h] = max + StrictMath.log(total);
      }
    });
    return out;
  }

  /**
   * 選んだ局末を種類ごとに数える。局末の無い種類も含めて、全ての種類を宣言の順に並べる。局末の無い種類では、順位が入れ替わった
   * 率は NaN。
   */
  public List<KindCount> kinds(IntPredicate selected) {
    int seats = ends.seats;
    Clusters nextChanged = new Clusters(KINDS.length);
    Clusters finalChanged = new Clusters(KINDS.length);
    Clusters seatChanged = new Clusters(KINDS.length);
    long[] counts = new long[KINDS.length];
    long total = 0;
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        int kind = kinds[h];
        counts[kind]++;
        total++;
        boolean nextDiffers = false;
        boolean finalDiffers = false;
        for (int seat = 0; seat < seats; seat++) {
          int j = h * seats + seat;
          boolean changed = ends.finalRank[j] != ends.rank[j];
          nextDiffers |= ranksAfter[j] != ends.rank[j];
          finalDiffers |= changed;
          seatChanged.add(kind, changed ? 1 : 0);
        }
        nextChanged.add(kind, nextDiffers ? 1 : 0);
        finalChanged.add(kind, finalDiffers ? 1 : 0);
      }
      nextChanged.endGame();
      finalChanged.endGame();
      seatChanged.endGame();
    }
    List<KindCount> rows = new ArrayList<>(KINDS.length);
    for (Kind kind : KINDS) {
      int cell = kind.ordinal();
      rows.add(new KindCount(
          kind,
          counts[cell],
          (double) counts[cell] / total,
          nextChanged.paired(cell),
          finalChanged.paired(cell),
          seatChanged.paired(cell)));
    }
    return rows;
  }

  /**
   * 今の順位のまま終わる率の誤差を、表せる局末と表せない局末に分ける。分けるのは、2 つの値それぞれの誤差と、同じ（席, 局末）の
   * 組での誤差の差。
   *
   * @param candidate 測る値（予測は周辺確率）。この局末の列について求めたもの
   * @param reference 比べる値。この局末の列について求めたもの
   * @param selected 測る局末
   * @param filter 測る席（区分・今の順位・点差で選ぶ）
   * @return 分けた結果
   */
  public KeepSplit keep(
      HandEndValues candidate, HandEndValues reference, IntPredicate selected, GrpMetrics.SeatFilter filter) {
    GrpMetrics.requireValues(ends, candidate);
    GrpMetrics.requireValues(ends, reference);
    int seats = ends.seats;
    double[] x = new double[QUANTITIES];
    // 量 v は、セル 3v に全ての局末、3v ＋ 1 に表せる局末、3v ＋ 2 に表せない局末の分を足す。最後の 2 つのセルには、
    // 表せる局末と表せない局末それぞれで、今の順位のまま終わったか（1 か 0）を足す。
    int keptCell = 3 * QUANTITIES;
    Clusters cells = new Clusters(keptCell + 2);
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        int part = kinds[h] == Kind.OUTCOME.ordinal() ? 1 : 2;
        for (int seat = 0; seat < seats; seat++) {
          int j = h * seats + seat;
          if (!filter.test(ends.group[h], ends.rank[j], ends.gap[j])) {
            continue;
          }
          double kept = keepErrors(candidate, reference, j, x);
          for (int v = 0; v < QUANTITIES; v++) {
            cells.add(3 * v, x[v]);
            cells.add(3 * v + part, x[v]);
          }
          cells.add(keptCell + part - 1, kept);
        }
      }
      cells.endGame();
    }
    double ratio = cells.count(0) / cells.count(1);
    // 「全体の平均 − 表せる局末の平均」の影響関数は (x − 全体の平均) − (全体の数 / 表せる局末の数)·[表せる局末]·(x − 表せる局末の平均)。
    // これを半荘ごとに足し、その和の分散を求める。
    Clusters influence = new Clusters(QUANTITIES);
    for (int game = 0; game < ends.games(); game++) {
      for (int h = ends.firstHand[game]; h < ends.firstHand[game + 1]; h++) {
        if (!selected.test(h)) {
          continue;
        }
        boolean represented = kinds[h] == Kind.OUTCOME.ordinal();
        for (int seat = 0; seat < seats; seat++) {
          int j = h * seats + seat;
          if (!filter.test(ends.group[h], ends.rank[j], ends.gap[j])) {
            continue;
          }
          keepErrors(candidate, reference, j, x);
          for (int v = 0; v < QUANTITIES; v++) {
            double error = x[v];
            influence.add(v, error - cells.mean(3 * v) - (represented ? ratio * (error - cells.mean(3 * v + 1)) : 0));
          }
        }
      }
      influence.endGame();
    }
    Split[] splits = new Split[QUANTITIES];
    for (int v = 0; v < QUANTITIES; v++) {
      splits[v] = new Split(
          cells.paired(3 * v),
          cells.paired(3 * v + 1),
          cells.paired(3 * v + 2),
          new Paired(cells.mean(3 * v) - cells.mean(3 * v + 1), Math.sqrt(influence.variance(v))));
    }
    return new KeepSplit(
        (long) cells.count(0),
        cells.count(2) / cells.count(0),
        cells.mean(keptCell),
        cells.mean(keptCell + 1),
        splits[0],
        splits[1],
        splits[2]);
  }

  /**
   * （席, 局末）の組 1 つについて、今の順位のまま終わる率の誤差を書く。書く順は、測る値の誤差、比べる値の誤差、その差（測る値の
   * 誤差 − 比べる値の誤差）。
   *
   * @param j （席, 局末）の組の番号（局末 × 席の数 ＋ 席）
   * @param out 書き込む先（長さ 3）
   * @return 今の順位のまま終わったか（1 か 0）
   */
  private double keepErrors(HandEndValues candidate, HandEndValues reference, int j, double[] out) {
    double kept = ends.finalRank[j] == ends.rank[j] ? 1 : 0;
    int at = j * ends.seats + ends.rank[j];
    out[0] = kept - candidate.marginals[at];
    out[1] = kept - reference.marginals[at];
    out[2] = reference.marginals[at] - candidate.marginals[at];
    return kept;
  }

  /**
   * 探索の結果から、遷移の種類を決める。遷移を再現する結果は探索の中に残っている。種類が {@code ENDING} なら、残っているのは
   * 記録を作ったエンジンの判定で探し直した結果。
   *
   * @param hits 局の結果のネットワークが表せる結果のうち、遷移を再現したものの数
   */
  private static Kind kind(TransitionSearch search, int hits) {
    if (hits > 0) {
      return Kind.OUTCOME;
    }
    if (search.liable()) {
      return Kind.PAO;
    }
    if (search.nagashi()) {
      return Kind.NAGASHI;
    }
    return search.ending() > 0 ? Kind.ENDING : Kind.MISMATCH;
  }

  private int at(int handEnd, int index) {
    Objects.checkIndex(index, found(handEnd));
    return firstFound[handEnd] + index;
  }

  /** 並列に調べたチャンク 1 つで見つけた、遷移を再現する結果。配列が足りなくなれば倍に伸ばす。 */
  private static final class Found {

    private byte[] mains = new byte[64];
    private byte[] bases = new byte[mains.length];
    private byte[] others = new byte[mains.length];
    private byte[] riichi = new byte[mains.length];
    private int size;

    void add(int main, int base, int other, int bits) {
      reserve(size + 1);
      mains[size] = (byte) main;
      bases[size] = (byte) base;
      others[size] = (byte) other;
      riichi[size] = (byte) bits;
      size++;
    }

    void append(Found chunk) {
      reserve(size + chunk.size);
      System.arraycopy(chunk.mains, 0, mains, size, chunk.size);
      System.arraycopy(chunk.bases, 0, bases, size, chunk.size);
      System.arraycopy(chunk.others, 0, others, size, chunk.size);
      System.arraycopy(chunk.riichi, 0, riichi, size, chunk.size);
      size += chunk.size;
    }

    private void reserve(int needed) {
      if (needed > mains.length) {
        int capacity = Math.max(needed, 2 * mains.length);
        mains = Arrays.copyOf(mains, capacity);
        bases = Arrays.copyOf(bases, capacity);
        others = Arrays.copyOf(others, capacity);
        riichi = Arrays.copyOf(riichi, capacity);
      }
    }
  }
}
