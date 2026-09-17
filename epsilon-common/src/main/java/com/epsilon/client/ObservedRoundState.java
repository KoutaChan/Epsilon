package com.epsilon.client;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Meld;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.MahjongTransition;
import com.epsilon.engine.PostCallDahaiRestriction;
import com.epsilon.engine.WinLegality.RonStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 外部サーバーから受信した公開イベントと自家の手牌を、共通の局面更新処理に順次反映する。 */
public final class ObservedRoundState {
  private static final Logger log = LoggerFactory.getLogger(ObservedRoundState.class);

  /** 副露を反映した直後の判断境界。次のイベントを反映する前に消費する。 */
  public sealed interface MeldResult permits OpenCall, Daiminkan, Kan {}

  /** チー・ポン後の打牌制約。 */
  public record OpenCall(PostCallDahaiRestriction restriction) implements MeldResult {}

  /** 大明槓後は槍槓応答を開かず、嶺上ツモを待つ。 */
  public enum Daiminkan implements MeldResult {
    INSTANCE
  }

  /** 暗槓・加槓の槍槓応答元。 */
  public record Kan(TurnEvent.KanAttempt attempt) implements MeldResult {}

  private enum Continuation {
    NORMAL,
    CALL_DISCARD,
    RINSHAN_DRAW
  }

  private final int selfSeat;
  private final GameState state = new GameState();
  private final MahjongTransition transition = new MahjongTransition(state);
  private Continuation continuation = Continuation.NORMAL;
  private int reachDiscardPlayer = -1;

  /** 対局中に変わらない自家の席を指定する。 */
  public ObservedRoundState(int selfSeat) {
    this.selfSeat = selfSeat;
  }

  /** この対局ループが所有する現在の観測状態を返す。 */
  public GameState state() {
    return state;
  }

  /** サーバーと同じ席番号系で自家の席を返す。 */
  public int selfSeat() {
    return selfSeat;
  }

  /** 観測状態が所有する自家手牌を返す。 */
  public Hand selfTiles() {
    return state.hand(selfSeat);
  }

  /** 局を初期化する。他家の非公開手牌や未知の山牌は作らない。 */
  public void startRound(
      int roundIndex,
      int dealer,
      int honba,
      int kyotaku,
      int doraIndicatorTileType,
      int[] selfPhysicalTileIds,
      int[] scores) {
    state.startRoundForReconstruction(roundIndex, dealer, honba, kyotaku);
    state.initializeWallForReconstruction(new int[] {doraIndicatorTileType}, 52);
    applyScores(scores);
    selfTiles().resetPhysicalTiles(selfPhysicalTileIds);
    continuation = Continuation.NORMAL;
    reachDiscardPlayer = -1;
    log.info(
        "Round start: {}{} honba={} kyotaku={} dealer=P{} self=P{} scores=[P0={}, P1={}, P2={},"
            + " P3={}]",
        "ESWN".charAt(roundIndex / GameState.NUM_PLAYERS),
        roundIndex % GameState.NUM_PLAYERS + 1,
        honba,
        kyotaku,
        dealer,
        selfSeat,
        state.getScore(0),
        state.getScore(1),
        state.getScore(2),
        state.getScore(3));
  }

  /** ツモを反映する。非公開の他家ツモは実牌 ID に {@code -1} を渡す。 */
  public TurnEvent.Draw draw(int player, int physicalTileId) {
    boolean rinshan = continuation == Continuation.RINSHAN_DRAW;
    if (rinshan) {
      state.clearAllIppatsu();
    }
    continuation = Continuation.NORMAL;
    return transition.reconstructDraw(player, physicalTileId, rinshan);
  }

  /** 打牌を反映する。ツモ切りの判定は入力プロトコル側で確定させる。 */
  public TurnEvent.Discard discard(int player, int tileType, boolean aka, boolean tsumogiri) {
    boolean riichi = reachDiscardPlayer == player;
    Action action =
        riichi
            ? Action.riichiDahai(tileType, aka, tsumogiri)
            : Action.dahai(tileType, aka, tsumogiri);
    TurnEvent.Discard discard =
        transition.reconstructDiscard(
            player, action, continuation != Continuation.CALL_DISCARD, player == selfSeat);
    continuation = Continuation.NORMAL;
    if (riichi) {
      reachDiscardPlayer = -1;
    }
    return discard;
  }

  /** 副露と手牌消費を一括反映し、次の打牌・槍槓応答・嶺上ツモを区別する。 */
  public MeldResult meld(int player, Meld meld, int[] consumedPhysicalTileIds) {
    transition.applyReconstructedMeld(player, meld, consumedPhysicalTileIds, player == selfSeat);
    return switch (meld.type()) {
      case CHI ->
          openCall(PostCallDahaiRestriction.afterChi(meld.baseTileType(), meld.calledTileType()));
      case PON -> openCall(PostCallDahaiRestriction.afterPon(meld.baseTileType()));
      case DAIMINKAN -> {
        continuation = Continuation.RINSHAN_DRAW;
        yield Daiminkan.INSTANCE;
      }
      case ANKAN -> {
        TurnEvent.KanAttempt attempt =
            transition.recordKanAttempt(
                player, meld.baseTileType(), false, TurnEvent.KanKind.ANKAN);
        state.clearAllIppatsu();
        continuation = Continuation.RINSHAN_DRAW;
        yield new Kan(attempt);
      }
      case KAKAN -> {
        TurnEvent.KanAttempt attempt =
            transition.recordKanAttempt(
                player, meld.baseTileType(), meld.addedTileIsAka(), TurnEvent.KanKind.KAKAN);
        continuation = Continuation.RINSHAN_DRAW;
        yield new Kan(attempt);
      }
    };
  }

  private OpenCall openCall(PostCallDahaiRestriction restriction) {
    continuation = Continuation.CALL_DISCARD;
    return new OpenCall(restriction);
  }

  /** 次の当該プレイヤーの打牌をリーチ宣言牌として扱う。支払いは成立通知で行う。 */
  public void reachDeclared(int player) {
    reachDiscardPlayer = player;
  }

  /** リーチ成立を反映する。得点スナップショットが省略された場合は千点の支払いを反映する。 */
  public void reachAccepted(int player, int[] scores) {
    state.setRiichi(player, true);
    transition.commitRiichiPayment(player);
    applyScores(scores);
  }

  /** サーバーの公開時点で新しいドラ表示牌を反映する。 */
  public void revealDora(int indicatorTileType) {
    transition.revealReconstructedDora(indicatorTileType);
  }

  /** 席順の得点スナップショットを反映する。省略されたスナップショットは更新しない。 */
  public void applyScores(int[] scores) {
    if (scores != null) {
      for (int player = 0; player < scores.length; player++) {
        state.setScore(player, scores[player]);
      }
    }
  }

  /** 役なし和了形の見逃しによる同巡フリテンを、応答要求の有無によらず反映する。 */
  public void observeResponse(RonStatus status) {
    if (status.createsImmediateFuriten()) {
      state.enterTemporaryFuriten(selfSeat);
    }
  }

  /** 合法なロンを選ばなかった自家のフリテンを反映する。 */
  public void completeResponse(RonStatus status, boolean canRon, Action selected) {
    if (status.createsFuritenWhenDeclined() && canRon && selected.type() != Action.Type.RON_AGARI) {
      state.enterTemporaryFuriten(selfSeat);
    }
  }
}
