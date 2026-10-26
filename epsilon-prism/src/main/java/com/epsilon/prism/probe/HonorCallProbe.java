package com.epsilon.prism.probe;

import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.Tile;
import com.epsilon.prism.collect.CandidateProbe;
import com.epsilon.prism.core.DecisionClass;
import java.util.List;
import java.util.concurrent.atomic.LongAdder;

/**
 * 字牌のポンを測る測定器（probe-honor）。合法手に字牌のポンがある判断を、その牌の種類（{@link Kind}）と PRISM の手（ポン・スルー・ほか）で数える。
 * 種類が target で PRISM がスルーした判断だけ、ポンを測定器の手として返し、ポンとスルーを同じ山の分岐で局末まで測らせる（ΔY = Y[ポン] − Y[スルー]）。
 * それ以外は打った手を返す（一致として数え、行も分岐も作らない）。
 *
 * <p>{@link #choose} は複数の進行ワーカーから並行に呼ばれる（数は {@link LongAdder}）。
 */
public final class HonorCallProbe implements CandidateProbe {
  /** 字牌の種類（席の視点）。 */
  public enum Kind {
    /** 自風かつ場風。 */
    DOUBLE,
    /** 自風（場風でない）。 */
    SEAT,
    /** 場風（自風でない）。 */
    ROUND,
    /** 三元牌。 */
    DRAGON,
    /** オタ風。 */
    GUEST
  }

  /** PRISM の手。 */
  public enum Choice {
    PON,
    PASS,
    OTHER
  }

  private final Kind target;
  private final LongAdder[][] counts = new LongAdder[Kind.values().length][Choice.values().length];

  public HonorCallProbe(Kind target) {
    this.target = target;
    for (LongAdder[] row : counts)
      for (int c = 0; c < row.length; c++) row[c] = new LongAdder();
  }

  /** 字牌 tile の席 player から見た種類（字牌でなければ null）。 */
  static Kind kind(int tile, int seatWind, int roundWind) {
    if (tile >= Tile.HAKU && tile <= Tile.CHUN) return Kind.DRAGON;
    if (tile < Tile.TON || tile > Tile.PEI) return null;
    if (tile == seatWind) return tile == roundWind ? Kind.DOUBLE : Kind.SEAT;
    return tile == roundWind ? Kind.ROUND : Kind.GUEST;
  }

  @Override
  public int choose(PublicObservation observation, List<Action> legal, int played) {
    int pon = -1;
    Kind kind = null;
    for (int a = 0; a < legal.size(); a++) {
      Action action = legal.get(a);
      if (action.type() != Action.Type.PON) continue;
      kind =
          kind(
              action.tileType(),
              observation.seatWindTileType(observation.observer()),
              observation.roundWindTileType());
      if (kind != null) {
        pon = a;
        break;
      }
    }
    if (pon < 0) return played;
    Action.Type type = legal.get(played).type();
    Choice choice =
        played == pon ? Choice.PON : type == Action.Type.PASS ? Choice.PASS : Choice.OTHER;
    counts[kind.ordinal()][choice.ordinal()].increment();
    return kind == target && choice == Choice.PASS ? pon : played;
  }

  @Override
  public void counted(DecisionClass decisionClass, Outcome outcome) {}

  /** 種類 kind で PRISM が choice を打った判断の数。 */
  public long count(Kind kind, Choice choice) {
    return counts[kind.ordinal()][choice.ordinal()].sum();
  }
}
