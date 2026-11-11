package com.epsilon.prism.input;

import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.DealerOutcome;
import com.epsilon.mahjong.rules.Placement;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.input.WaitScoring.Stance;
import java.util.Arrays;

/**
 * 聴牌になる遷移について、厳密な入力の値（{@link Field}）を作る。待ちの採点を元に、ロンとツモのそれぞれで、和了で半荘が終わる割合、
 * 終わるときの効用、得点の平均を求める。四人打ち・三人打ちの入力の形式は、この値を厳密な入力の列に写す。
 *
 * <p>和了は、待ち ×（赤でない和了牌・赤 5）× 和了牌の出どころ（ロンは他家の全ての席、ツモは自分）に分けて数える。重みはその和了牌の
 * 見えていない枚数で、ロンはどの席から出ても同じ重みにする。点の移動は、{@link Settlement} の払い（本場を含み、包はなし。三人打ちは
 * ツモ損）に、場の供託（立直の遷移なら立直棒を 1 本足す）を和了者の分として足したもの。終局かどうかは、精算の後の点で
 * {@link Progression#finishes} が決める。順位はその点で {@link Placement} が決め（同点は起家に近い席が上）、効用には作るときに渡した
 * ものを使う。
 *
 * <p>得点は、判断の時点から和了の後までの自分の点の変化（本場・供託を含む）。立直の遷移では立直棒の 1000 点を払って供託が 1 本増え、和了で
 * それを取り戻すので、立直棒は差し引きで消える。役のない和了と、振聴でロンできないと決まった遷移のロンは数えない。数える和了がなければ、
 * そのロンかツモの列は 0。何度も呼ばれるので、メモリを確保しない。
 */
public final class WaitProjection {

  /** 厳密な入力の値。 */
  public enum Field {
    /** 聴牌なら 1。 */
    TENPAI,
    /** 待ちの残り枚数（見えていない枚数）。 */
    LIVE_WAITS,
    /** ロン和了で半荘が終わる割合。 */
    RON_ENDS,
    /** ロン和了で半荘が終わるときの効用の平均。 */
    RON_UTILITY,
    /** ロン和了の得点の平均。 */
    RON_POINTS,
    /** ツモ和了で半荘が終わる割合。 */
    TSUMO_ENDS,
    /** ツモ和了で半荘が終わるときの効用の平均。 */
    TSUMO_UTILITY,
    /** ツモ和了の得点の平均。 */
    TSUMO_POINTS
  }

  private static final WinningTile[] WINNING_TILES = WinningTile.values();

  private final RankUtility utility;
  private final float[] values = new float[Field.values().length];
  private Variant variant;
  private int seats;
  private int[] scores = new int[0];
  private int[] after = new int[0];
  private int[] ranks = new int[0];
  private long[] endsByRank = new long[0];
  private int observer;
  private int dealerAbsolute;
  private boolean dealer;
  private int kyoku;
  private int honba;
  private int kyotaku;
  private long weight;
  private long points;

  /**
   * 終局の効用を渡して作る。
   *
   * @param utility 終局の効用
   */
  public WaitProjection(RankUtility utility) {
    this.utility = utility;
  }

  /**
   * 判断の記録から点数と局を読む。
   *
   * @param o 判断の記録
   */
  public void load(Observation o) {
    if (o.variant() != variant) {
      variant = o.variant();
      seats = variant.seats();
      scores = new int[seats];
      after = new int[seats];
      ranks = new int[seats];
      endsByRank = new long[seats];
    }
    observer = o.observer();
    dealer = o.dealer() == 0;
    dealerAbsolute = o.absoluteSeat(o.dealer());
    kyoku = o.kyoku();
    honba = o.honba();
    kyotaku = o.kyotaku();
    for (int seat = 0; seat < seats; seat++) {
      scores[seat] = o.score(seat);
    }
  }

  /**
   * 遷移の厳密な入力を作る。
   *
   * @param waits 遷移の後の手の待ちの採点
   * @param stance どちらの採点を使うか（ダマか、その遷移の立直か）
   * @param declaresRiichi その遷移で立直を宣言するか（宣言すれば立直棒を払う）
   * @param ronBlocked 振聴でロンできないか
   * @param tiles 種類ごとの枚数（見えていない枚数と見えていない赤を読む）
   * @param waitMask 手の形から見た待ち（牌のマスク）
   */
  public void project(
      WaitScoring waits,
      Stance stance,
      boolean declaresRiichi,
      boolean ronBlocked,
      TileCounts tiles,
      long waitMask) {
    Arrays.fill(values, 0f);
    values[Field.TENPAI.ordinal()] = 1f;
    values[Field.LIVE_WAITS.ordinal()] = tiles.live(waitMask);
    if (!ronBlocked) {
      accumulate(waits, stance, Method.RON, declaresRiichi, tiles);
      store(Field.RON_ENDS, Field.RON_UTILITY, Field.RON_POINTS);
    }
    accumulate(waits, stance, Method.TSUMO, declaresRiichi, tiles);
    store(Field.TSUMO_ENDS, Field.TSUMO_UTILITY, Field.TSUMO_POINTS);
  }

  /** 作った値（正規化の前）。 */
  public float value(Field field) {
    return values[field.ordinal()];
  }

  /** 和了を重み付きで足し合わせる。和は整数で持つので、待ちを足す順（物理の色の順）が変わっても同じ値になる。 */
  private void accumulate(
      WaitScoring waits, Stance stance, Method method, boolean declaresRiichi, TileCounts tiles) {
    weight = 0;
    points = 0;
    Arrays.fill(endsByRank, 0L);
    long unseenRed = tiles.unseenRed();
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      int tile = waits.waitTile(wait);
      int redCopies = (unseenRed >>> tile & 1L) != 0L ? 1 : 0;
      for (WinningTile winning : WINNING_TILES) {
        int copies = winning == WinningTile.RED ? redCopies : tiles.unseen(tile) - redCopies;
        int base = waits.basePoints(wait, stance, method, winning);
        if (copies == 0 || base == 0) {
          continue;
        }
        if (method == Method.TSUMO) {
          outcome(base, 0, copies, declaresRiichi);
        } else {
          for (int from = 1; from < seats; from++) {
            outcome(base, from, copies, declaresRiichi);
          }
        }
      }
    }
  }

  /** 自分が相対席 {@code from} から和了した場合を、重み {@code copies} で足す。基本点は {@code base}、{@code from} が 0 ならツモ。 */
  private void outcome(int base, int from, int copies, boolean declaresRiichi) {
    Arrays.fill(after, 0);
    if (from == 0) {
      Settlement.tsumo(variant, base, observer, dealerAbsolute, honba, RoundOutcome.NO_SEAT, after);
    } else {
      int loser = (observer + from) % seats;
      Settlement.ron(variant, base, observer, loser, dealerAbsolute, honba, RoundOutcome.NO_SEAT, after);
    }
    int sticks = kyotaku + (declaresRiichi ? 1 : 0);
    after[observer] += sticks * Variant.DEPOSIT - (declaresRiichi ? Variant.DEPOSIT : 0);
    for (int seat = 0; seat < seats; seat++) {
      int absolute = (observer + seat) % seats;
      after[absolute] += scores[seat];
    }
    boolean finished =
        Progression.finishes(
            variant, kyoku, after, dealer ? DealerOutcome.CONTINUES : DealerOutcome.PASSES);
    weight += copies;
    points += (long) copies * (after[observer] - scores[0]);
    if (finished) {
      Placement.ranks(after, ranks);
      endsByRank[ranks[observer]] += copies;
    }
  }

  private void store(Field endsField, Field utilityField, Field pointsField) {
    if (weight == 0) {
      return;
    }
    long ends = 0;
    double utility = 0;
    for (int rank = 0; rank < seats; rank++) {
      ends += endsByRank[rank];
      utility += endsByRank[rank] * this.utility.value(variant, rank);
    }
    values[endsField.ordinal()] = (float) ((double) ends / weight);
    values[utilityField.ordinal()] = (float) (utility / weight);
    values[pointsField.ordinal()] = (float) ((double) points / weight);
  }
}
