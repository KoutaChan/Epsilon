package com.epsilon.replay;

import com.epsilon.calculate.scoring.HandScore;
import com.epsilon.calculate.scoring.ScoringYaku;
import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.WinClaim;
import com.epsilon.engine.WinPointDeltas;
import com.epsilon.engine.WinScorer;
import com.epsilon.replay.ReplayEvent.Hora;
import java.util.ArrayList;
import java.util.List;

/**
 * 牌譜に採点の内訳が無い和了を、実対局と同じ規則（{@link WinScorer}）で採点して内訳を補う。計算した点の移動が牌譜の点の移動と全ての席で
 * 一致した時だけ補い、一致しなければ（牌譜のルールが違う、裏ドラが分からないなど）牌譜のままにする。
 *
 * <p>役の符号は {@link ScoringYaku} の名前で、自風と場風の役だけ風を付ける（天鳳の牌譜と同じ YAKUHAI_SEAT_E など）。ドラは DORA・
 * AKADORA・URADORA の順に後ろへ並べ、役満には付けない。
 */
final class WinDetailsScorer {
  private static final ScoringYaku[] YAKU = ScoringYaku.values();
  private static final String[] WINDS = {"E", "S", "W", "N"};

  private final WinScorer scorer = new WinScorer();

  /**
   * 内訳を補った和了を返す。牌譜に採点の値がある、点の移動が無い、採点できない、または点の移動が合わないなら hora そのもの。牌譜が見せた
   * 裏ドラ表示牌は、採点の前に局面へ反映し、補った内訳にもそのまま残す。
   *
   * @param state 和了の点を動かす前の局面
   * @param honba この和了が受け取る本場
   * @param kyotaku この和了が受け取る供託の本数
   */
  Hora complete(Hora hora, GameState state, int honba, int kyotaku) {
    WinDetails recorded = hora.details();
    int[] deltas = hora.deltas();
    boolean scored = recorded != null && recorded.scored();
    if (scored || deltas == null || deltas.length != GameState.NUM_PLAYERS) return hora;
    List<String> ura = recorded == null ? null : recorded.uraIndicators();
    if (ura != null) revealUra(state, ura);
    WinClaim claim = scorer.claim(state, hora.actor(), honba);
    if (claim == null || !reproduces(claim, deltas, kyotaku)) return hora;
    return new Hora(
        hora.actor(), hora.target(), hora.winTileType(), deltas, details(claim, state, ura));
  }

  private static void revealUra(GameState state, List<String> ura) {
    int count = Math.min(ura.size(), state.doraState().indicatorCount());
    for (int index = 0; index < count; index++)
      state.revealUraIndicator(index, TileNotation.parseTileType(ura.get(index)));
  }

  /** 計算した点の移動に供託を足した物が、牌譜の点の移動と全ての席で等しいか。 */
  private static boolean reproduces(WinClaim claim, int[] deltas, int kyotaku) {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      int expected = claim.pointDelta().get(seat);
      if (seat == claim.winner()) expected += kyotaku * HanchanProgression.RIICHI_COST;
      if (deltas[seat] != expected) return false;
    }
    return true;
  }

  private static WinDetails details(WinClaim claim, GameState state, List<String> ura) {
    HandScore score = claim.score();
    List<WinDetails.Yaku> yaku = new ArrayList<>();
    for (ScoringYaku each : YAKU) {
      if (score.hasYaku(each))
        yaku.add(
            new WinDetails.Yaku(
                code(each, state, claim.winner()),
                each.han(score.isMenzen()),
                each.yakumanMultiplier()));
    }
    if (!score.isYakuman()) {
      addDora(yaku, "DORA", score.omoteDoraCount());
      addDora(yaku, "AKADORA", score.akaDoraCount());
      addDora(yaku, "URADORA", score.uraDoraCount());
    }
    int[] payments = new int[GameState.NUM_PLAYERS];
    WinPointDeltas.settleInto(
        score.basePoints(), claim.winner(), claim.from(), state.getOya(), 0, 0, payments);
    return new WinDetails(
        score.isYakuman() ? 0 : score.han(),
        score.isYakuman() ? null : score.fu(),
        payments[claim.winner()],
        score.yakumanMultiplier(),
        List.copyOf(yaku),
        ura);
  }

  private static String code(ScoringYaku yaku, GameState state, int winner) {
    return switch (yaku) {
      case YAKUHAI_SEAT -> "YAKUHAI_SEAT_" + WINDS[state.getJikaze(winner) - Tile.TON];
      case YAKUHAI_ROUND -> "YAKUHAI_ROUND_" + WINDS[state.getBakaze() - Tile.TON];
      default -> yaku.name();
    };
  }

  private static void addDora(List<WinDetails.Yaku> yaku, String code, int count) {
    if (count > 0) yaku.add(new WinDetails.Yaku(code, count, 0));
  }
}
