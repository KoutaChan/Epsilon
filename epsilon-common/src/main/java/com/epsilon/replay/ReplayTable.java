package com.epsilon.replay;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.MahjongTransition;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import java.util.Arrays;

/**
 * 再生の局面と、牌譜の出来事による書き換え。GameState に加えて、GameState が持たない牌譜の事実（宣言だけのリーチ、打牌を待つチー・ポン、局で和了を精算したか、手番の判断を
 * 起こした出来事、次に使う牌山）を持つ。局面を書き換えるのはここだけで、点の持ち主は GameState だけ。
 */
final class ReplayTable {
  private static final int STARTING_SCORE = 25000;

  /** 配牌で牌山から取った牌の数。 */
  private static final int DEALT_TILES = 13 * GameState.NUM_PLAYERS;

  private static final int[] NO_INDICATORS = {};

  private final GameState state = new GameState();
  private final MahjongTransition transition = new MahjongTransition(state);
  private final WinDetailsScorer winDetails = new WinDetailsScorer();
  private final int[][] walls;
  private final boolean[] riichiDeclared = new boolean[GameState.NUM_PLAYERS];
  private int nextWall;
  private Meld pendingCall;

  /** この局で和了を精算したか。同じ打牌への 2 人目以降のロンは本場と供託を受け取らない（上家取り）。 */
  private boolean winSettled;

  private int turnCause = -1;

  /** @param walls 局ごとに復号した牌山（無ければ null）。局の始めごとに先頭から順に使い、尽きれば牌山の無い道で組む */
  ReplayTable(int[][] walls) {
    this.walls = walls;
  }

  GameState state() {
    return state;
  }

  /** 手番の判断を起こした出来事の位置（ツモか鳴き）。 */
  int turnCause() {
    return turnCause;
  }

  /** 打牌を待つ直前のチー・ポン。無ければ null。 */
  Meld pendingCall() {
    return pendingCall;
  }

  /** 牌譜がリーチを宣言し、その打牌をまだ適用していないか。 */
  boolean riichiDeclared(int seat) {
    return riichiDeclared[seat];
  }

  /** seat が大明槓・加槓の嶺上牌をツモり、その新ドラが打牌の後まで待っているか。 */
  boolean kanDoraWaitsForDiscard(int seat) {
    return state.getTurnEvent() instanceof TurnEvent.Draw draw
        && draw.player() == seat
        && draw.doraRevealPending();
  }

  /** @throws IllegalStateException 手番の出来事がツモでない（ツモの無い手番の判断） */
  TurnEvent.Draw requireDraw(int index) {
    if (state.getTurnEvent() instanceof TurnEvent.Draw draw) return draw;
    throw new IllegalStateException(
        "Event "
            + index
            + " is a turn decision, but the last turn event is "
            + name(state.getTurnEvent())
            + ", not a draw.");
  }

  void startMatch() {
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) state.setScore(seat, STARTING_SCORE);
  }

  void startRound(int index, StartKyoku start) {
    if (start.scores() != null)
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
        state.setScore(seat, start.scores()[seat]);
    int round = (start.wind() - Tile.TON) * GameState.NUM_PLAYERS + start.kyoku() - 1;
    state.startRoundForReconstruction(round, start.oya(), start.honba(), start.kyotaku());
    if (walls != null && nextWall < walls.length) {
      state.initializeWall(walls[nextWall++]);
    } else {
      state.initializeWallForReconstruction(NO_INDICATORS, DEALT_TILES);
      state.revealDoraIndicator(start.doraIndicatorTileType(), start.doraIndicatorAka());
    }
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      for (int physicalTileId : start.initialHandPhysicalTileIds()[seat])
        state.hand(seat).addPhysicalTile(physicalTileId);
    Arrays.fill(riichiDeclared, false);
    pendingCall = null;
    winSettled = false;
    turnCause = index;
  }

  void draw(int index, Tsumo tsumo) {
    boolean rinshan =
        state.getTurnEvent() instanceof TurnEvent.Draw begun && begun.isRinshanDraw();
    transition.reconstructDraw(tsumo.actor(), tsumo.physicalTileId(), rinshan);
    turnCause = index;
  }

  /** 次の打牌をリーチの打牌にするだけで、局面は変えない。 */
  void declareRiichi(int seat) {
    riichiDeclared[seat] = true;
  }

  void payRiichi(int seat) {
    transition.commitRiichiPayment(seat);
  }

  /** @return 応答の元になる打牌（GameState が使い回す物） */
  TurnEvent.Discard discard(Dahai discard, Action action) {
    riichiDeclared[discard.actor()] = false;
    TurnEvent.Discard origin =
        transition.reconstructDiscard(discard.actor(), action, pendingCall == null);
    pendingCall = null;
    return origin;
  }

  void ankan(Ankan kan) {
    transition.completeAnkan(kan.actor(), Tile.typeOf(kan.consumedPhysicalTileIds()[0]));
    transition.beginReconstructionRinshan(kan.actor(), false);
    pendingCall = null;
  }

  /** @return 槍槓の応答の元（GameState が使い回す物）。窓を閉じるまで手番の出来事に残る */
  TurnEvent.KanAttempt declareKakan(Kakan kan) {
    int type = Tile.typeOf(kan.addedPhysicalTileId());
    transition.declareKakan(kan.actor(), type);
    return transition.recordKanAttempt(
        kan.actor(), type, Tile.isAka(kan.addedPhysicalTileId()), TurnEvent.KanKind.KAKAN);
  }

  /** 槍槓されなかった加槓を成立させ、嶺上ツモの境へ進める（新ドラは打牌の後に開く）。 */
  void completeKakan(int seat) {
    transition.beginReconstructionRinshan(seat, true);
    pendingCall = null;
  }

  /** 槍槓された加槓の宣言を手番の出来事から外す。 */
  void robKakan() {
    state.clearTurnEvent();
  }

  /** @throws IllegalStateException 手番の出来事が鳴かれる打牌でない */
  void chi(int index, Chi chi, Action action) {
    pendingCall = transition.applyChi(chi.actor(), requireDiscard(index), action);
    turnCause = index;
  }

  /** @throws IllegalStateException 手番の出来事が鳴かれる打牌でない */
  void pon(int index, Pon pon, Action action) {
    pendingCall =
        transition.applyPon(pon.actor(), requireDiscard(index), action.usesAkaTileFromHand());
    turnCause = index;
  }

  /** @throws IllegalStateException 手番の出来事が鳴かれる打牌でない */
  void daiminkan(int index, Daiminkan kan) {
    transition.completeDaiminkan(kan.actor(), requireDiscard(index));
    transition.beginReconstructionRinshan(kan.actor(), true);
    pendingCall = null;
    turnCause = index;
  }

  /** 表示牌の赤も局面へ開く（{@link MahjongTransition#revealReconstructedDora} は赤を捨てるので使わない）。 */
  void revealDora(Dora dora) {
    state.revealDoraIndicator(dora.indicatorTileType(), dora.indicatorAka());
  }

  /** @return 一時振聴に変わったか */
  boolean enterFuriten(int seat) {
    boolean changed = !state.isTemporaryFuriten(seat);
    state.enterTemporaryFuriten(seat);
    return changed;
  }

  /**
   * 和了の点を動かし、通知する和了を返す。牌譜に無い内訳は {@link WinDetailsScorer} が補い、牌譜の裏ドラ表示はその前に局面へ開く。局で最初に精算した和了だけが本場と
   * 供託を受け取り、和了の後の供託は 0。採点は手番の出来事を和了の元として読むので、元（ツモ・打牌・加槓の宣言）が手番の出来事のうちに呼ぶ（ここは手番の出来事を変えない）。
   */
  Hora settleWin(Hora win) {
    Hora settled =
        winSettled
            ? winDetails.complete(win, state, 0, 0)
            : winDetails.complete(win, state, state.getHonba(), state.getKyotakuCount());
    winSettled = true;
    addDeltas(win.deltas());
    state.setKyotakuCount(0);
    return settled;
  }

  void settleRyukyoku(Ryukyoku ryukyoku) {
    addDeltas(ryukyoku.deltas());
  }

  /** 牌譜の終局の点があれば、それを局面の点にする（配列は持たない）。 */
  void endMatch(EndGame end) {
    if (end.scores() == null) return;
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      state.setScore(seat, end.scores()[seat]);
  }

  /** 局面の点の新しい配列。 */
  int[] finalScores() {
    int[] scores = new int[GameState.NUM_PLAYERS];
    for (int seat = 0; seat < scores.length; seat++) scores[seat] = state.getScore(seat);
    return scores;
  }

  private TurnEvent.Discard requireDiscard(int index) {
    if (state.getTurnEvent() instanceof TurnEvent.Discard discard) return discard;
    throw new IllegalStateException(
        "Event "
            + index
            + " calls a tile, but the last turn event is "
            + name(state.getTurnEvent())
            + ", not a discard.");
  }

  private void addDeltas(int[] deltas) {
    if (deltas == null) return;
    for (int seat = 0; seat < Math.min(deltas.length, GameState.NUM_PLAYERS); seat++)
      state.addScore(seat, deltas[seat]);
  }

  private static String name(TurnEvent event) {
    return event.getClass().getSimpleName();
  }
}
