package com.epsilon.engine;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.core.Action;
import com.epsilon.core.DoraState;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.RoundPublicStateIndex;
import java.util.Arrays;
import java.util.List;

/**
 * 判断を待つ局面の公開情報と、各合法手を適用した場合の解析結果を保持する再利用バッファ。
 *
 * <p>通常は{@link GameEngine#analyzeDecision(EngineDecisionPoint,
 * EngineDecisionBuffer)}で設定する。局面を複製せず、次の行動が反映されるまで元の局面を読み取り専用で参照する。
 */
public final class EngineDecisionBuffer {

  private static final int MAX_ACTIONS = 32;
  private static final int MAX_TRANSITIONS = 192;
  private static final ActionEffect.NextStep[] NEXT_STEPS = ActionEffect.NextStep.values();
  private static final Meld.AkaSource[] MELD_AKA_SOURCES = Meld.AkaSource.values();

  private final DecisionHandAnalysisBuffer handAnalysis = new DecisionHandAnalysisBuffer();
  private final DecisionHandAnalyzer analyzer = new DecisionHandAnalyzer();
  private final DecisionTileIndex tileIndex = new DecisionTileIndex();
  private final EngineActionBuffer immediateDiscards = new EngineActionBuffer();
  private final ActionEffect.ProjectedHand rootProjectionHand = new ActionEffect.ProjectedHand();
  private final ActionEffect.ProjectedHand borrowedAfterstate = new ActionEffect.ProjectedHand();
  private final ActionEffect.ProjectionResult projectionResult =
      new ActionEffect.ProjectionResult();

  private List<Action> legalActions = List.of();
  private GameState state;
  private PublicObservation observation;
  private int playerIndex;
  private RoundPublicStateIndex publicState;

  private final Action[] actions = new Action[MAX_ACTIONS];
  private final byte[] continuationByAction = new byte[MAX_ACTIONS];
  private final byte[] meldAkaSourceByAction = new byte[MAX_ACTIONS];
  private final int[] firstTransitionByAction = new int[MAX_ACTIONS];
  private final byte[] transitionCountByAction = new byte[MAX_ACTIONS];
  private final WinSettlementProjection[] immediateWinByAction =
      new WinSettlementProjection[MAX_ACTIONS];

  private final Action[] discardByTransition = new Action[MAX_TRANSITIONS];
  private final byte[] doraCountByTransition = new byte[MAX_TRANSITIONS];
  private int transitionCount;
  private int maximumTransitionsPerAction;
  private int materializedAction = -1;
  private int materializedTransition = -1;

  public EngineDecisionBuffer() {
    for (int index = 0; index < immediateWinByAction.length; index++) {
      immediateWinByAction[index] = new WinSettlementProjection();
    }
  }

  void bind(GameState state, EngineDecisionPoint decision, RoundPublicStateIndex publicState) {
    bind(state, decision.player(), decision.legalActions(), publicState);
  }

  void bind(
      GameState state, int player, List<Action> legalActions, RoundPublicStateIndex publicState) {
    if (legalActions.size() > MAX_ACTIONS) {
      throw new IllegalStateException("legal action capacity exceeded: " + legalActions.size());
    }
    clearReferences();
    this.state = state;
    this.observation = state.publicObservation(player);
    this.playerIndex = player;
    this.legalActions = legalActions;
    this.publicState = publicState;
    capturePublicHandContext();
    projectActions(state, player, legalActions);
  }

  /** 牌譜再生や外部復元局面を局面の実データのまま解析バッファへ結び付ける。 */
  public void analyze(
      GameState state, int player, List<Action> legalActions, RoundPublicStateIndex publicState) {
    bind(state, player, legalActions, publicState);
  }

  private void capturePublicHandContext() {
    tileIndex.load(state.hand(playerIndex()), publicState, state.doraState());
  }

  private void projectActions(GameState state, int player, List<Action> legalActions) {
    transitionCount = 0;
    maximumTransitionsPerAction = 1;
    materializedAction = -1;
    materializedTransition = -1;
    for (int actionIndex = 0; actionIndex < legalActions.size(); actionIndex++) {
      Action action = legalActions.get(actionIndex);
      ActionEffect.describeInto(
          state, player, action, rootProjectionHand, projectionResult, immediateDiscards);
      actions[actionIndex] = action;
      continuationByAction[actionIndex] = (byte) projectionResult.nextStep().ordinal();
      meldAkaSourceByAction[actionIndex] = (byte) projectionResult.meldAkaSource().ordinal();
      if (action.type() == Action.Type.RON_AGARI || action.type() == Action.Type.TSUMO_AGARI) {
        immediateWinByAction[actionIndex].load(state, player, action, analyzer.scoreEvaluator());
      }
      firstTransitionByAction[actionIndex] = transitionCount;
      int doraCountAfterAction = tileIndex.doraCountAfter(action);

      if (immediateDiscards.isEmpty()) {
        requireTransitionCapacity(transitionCount + 1);
        discardByTransition[transitionCount] =
            action.type().group() == Action.Group.DAHAI
                    || action.type().group() == Action.Group.RIICHI
                ? action
                : null;
        doraCountByTransition[transitionCount] = (byte) doraCountAfterAction;
        transitionCountByAction[actionIndex] = 1;
        transitionCount++;
        continue;
      }

      int followUpCount = immediateDiscards.size();
      requireTransitionCapacity(transitionCount + followUpCount);
      for (int index = 0; index < followUpCount; index++) {
        Action discard = immediateDiscards.get(index);
        discardByTransition[transitionCount] = discard;
        doraCountByTransition[transitionCount] =
            (byte) tileIndex.doraCountAfterDiscard(doraCountAfterAction, discard);
        transitionCount++;
      }
      transitionCountByAction[actionIndex] = (byte) followUpCount;
      maximumTransitionsPerAction = Math.max(maximumTransitionsPerAction, followUpCount);
    }
  }

  private void clearReferences() {
    if (legalActions.isEmpty()) {
      return;
    }
    Arrays.fill(actions, 0, legalActions.size(), null);
    Arrays.fill(discardByTransition, 0, transitionCount, null);
    immediateDiscards.clear();
  }

  private static void requireTransitionCapacity(int required) {
    if (required > MAX_TRANSITIONS) {
      throw new IllegalStateException("decision transition capacity exceeded: " + required);
    }
  }

  public PublicObservation state() {
    return observation;
  }

  public int playerIndex() {
    return playerIndex;
  }

  public int actionCount() {
    return legalActions.size();
  }

  public List<Action> legalActions() {
    return legalActions;
  }

  public Action action(int actionIndex) {
    return actions[actionIndex];
  }

  public ActionEffect.NextStep continuation(int actionIndex) {
    return NEXT_STEPS[continuationByAction[actionIndex]];
  }

  public Meld.AkaSource meldAkaSource(int actionIndex) {
    return MELD_AKA_SOURCES[meldAkaSourceByAction[actionIndex]];
  }

  public WinSettlementProjection immediateWin(int actionIndex) {
    Action.Type type = actions[actionIndex].type();
    return type == Action.Type.RON_AGARI || type == Action.Type.TSUMO_AGARI
        ? immediateWinByAction[actionIndex]
        : null;
  }

  public int transitionCount(int actionIndex) {
    return transitionCountByAction[actionIndex] & 0xff;
  }

  public int maximumTransitionsPerAction() {
    return maximumTransitionsPerAction;
  }

  public Action discardAction(int actionIndex, int transitionIndex) {
    return discardByTransition[firstTransitionByAction[actionIndex] + transitionIndex];
  }

  public HandView handAfterTransition(int actionIndex, int transitionIndex) {
    return materializeAfterstate(actionIndex, transitionIndex);
  }

  /** 形状のみを解析する。結果は次の解析まで有効。 */
  public DecisionHandAnalysisBuffer analyzeCurrentHand() {
    analyzer.analyzeShape(state.hand(playerIndex()), this, handAnalysis);
    return handAnalysis;
  }

  /** 現在待ちまでを解析し、一向聴からの探索は実行しない。 */
  public DecisionHandAnalysisBuffer analyzeCurrentWaits(RiichiState riichi) {
    analyzer.analyzeWaits(
        state.hand(playerIndex()), tileIndex.currentHandDoraCount(), riichi, this, handAnalysis);
    return handAnalysis;
  }

  /** 現在待ちと、一向聴からの到達待ちを解析する。 */
  public DecisionHandAnalysisBuffer analyzeCurrentHand(RiichiState riichi, long ownRiver) {
    analyzeCurrentWaits(riichi);
    analyzer.analyzeFrontier(state.hand(playerIndex()), ownRiver, this, handAnalysis);
    return handAnalysis;
  }

  /** 候補行動を適用した手牌の現在待ちまでを解析する。 */
  public DecisionHandAnalysisBuffer analyzeTransitionWaits(
      int action, int transition, RiichiState riichi) {
    analyzer.analyzeWaits(
        materializeAfterstate(action, transition),
        doraCountByTransition[firstTransitionByAction[action] + transition] & 0xff,
        riichi,
        this,
        handAnalysis);
    return handAnalysis;
  }

  /** 候補行動を適用した手牌の現在待ちと到達待ちを解析する。 */
  public DecisionHandAnalysisBuffer analyzeTransition(
      int action, int transition, RiichiState riichi, long ownRiver) {
    analyzeTransitionWaits(action, transition, riichi);
    analyzer.analyzeFrontier(
        materializeAfterstate(action, transition), ownRiver, this, handAnalysis);
    return handAnalysis;
  }

  private HandView materializeAfterstate(int actionIndex, int transitionIndex) {
    if (actionIndex == materializedAction && transitionIndex == materializedTransition) {
      return borrowedAfterstate;
    }
    Action action = actions[actionIndex];
    if (continuation(actionIndex) == ActionEffect.NextStep.IMMEDIATE_DISCARD) {
      if (actionIndex != materializedAction) {
        ActionEffect.projectInto(
            state, playerIndex, action, rootProjectionHand, projectionResult, immediateDiscards);
      }
      ActionEffect.projectImmediateDiscardInto(
          rootProjectionHand, discardAction(actionIndex, transitionIndex), borrowedAfterstate);
    } else {
      ActionEffect.projectInto(
          state, playerIndex, action, borrowedAfterstate, projectionResult, immediateDiscards);
    }
    materializedAction = actionIndex;
    materializedTransition = transitionIndex;
    return borrowedAfterstate;
  }

  public int visibleTileCount(int tileType) {
    return tileIndex.visibleCopies(tileType);
  }

  public int unseenCopies(int tileType) {
    return tileIndex.unseenCopies(tileType);
  }

  /** 所有配列を返す内部処理用API。呼び出し側は変更してはならない。 */
  public int[] unseenCopiesByTileType() {
    return tileIndex.unseenCopies();
  }

  public int totalUnseenCopies() {
    return tileIndex.totalUnseenCopies();
  }

  int countUnseenCopies(long tileTypeMask) {
    return tileIndex.countUnseenCopies(tileTypeMask);
  }

  int countLiveTileTypes(long tileTypeMask) {
    return tileIndex.countLiveTileTypes(tileTypeMask);
  }

  DoraState doraState() {
    return tileIndex.doraState();
  }

  public int visibleAkaTileTypeMask() {
    return tileIndex.visibleAkaTileTypeMask();
  }

  public int doraMultiplicity(int tileType) {
    return tileIndex.doraMultiplicity(tileType);
  }

  WinConditions winContext(RiichiState riichiStatus) {
    return WinConditions.publicDecision(
        state.seatWindTileType(playerIndex()), state.roundWindTileType(), riichiStatus);
  }

  public int doraIndicatorMultiplicity(int tileType) {
    return tileIndex.doraIndicatorMultiplicity(tileType);
  }

  public long ownRiverTileTypeMask() {
    return publicState.riverTileTypeMask(playerIndex());
  }

  public int riverDiscardCount(int relativeSeat, int tileType) {
    return publicState.riverDiscardCount(absoluteSeat(relativeSeat), tileType);
  }

  public int riichiDeclarationIndex(int relativeSeat) {
    return publicState.riichiDeclarationIndex(absoluteSeat(relativeSeat));
  }

  public int discardsAfterRiichi(int relativeSeat) {
    return publicState.discardsAfterRiichi(absoluteSeat(relativeSeat));
  }

  public int postRiichiTsumogiriCount(int relativeSeat) {
    return publicState.postRiichiTsumogiriCount(absoluteSeat(relativeSeat));
  }

  public long riichiGenbutsuTileTypeMask(int relativeSeat) {
    return publicState.riichiGenbutsuTileTypeMask(absoluteSeat(relativeSeat));
  }

  private int absoluteSeat(int relativeSeat) {
    return (playerIndex + relativeSeat) & (GameState.NUM_PLAYERS - 1);
  }
}
