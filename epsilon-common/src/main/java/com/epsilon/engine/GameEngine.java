package com.epsilon.engine;

import com.epsilon.core.Action;
import com.epsilon.core.DecisionLearningRole;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import java.util.List;

/**
 * 麻雀ゲームエンジン。
 *
 * <p>4 人打ちリーチ麻雀の対局進行を担当し、局開始から和了・流局までを処理する。
 */
public final class GameEngine {

  private final GameState state;

  /** 次の行動反映まで有効な、指定席の読み取り専用観測。 */
  public PublicObservation observation(int player) {
    return state.publicObservation(player);
  }

  private final GameRecorder recorder;
  private final ActionGenerator actionGeneration = new ActionGenerator();
  private final RoundWinResolver winResolver;
  private final RoundDrawResolver drawResolver;
  private final MahjongTransition actions;
  private final RoundScratch scratch = new RoundScratch();
  private final GameStepResult.AwaitingDecisions awaitingStep =
      new GameStepResult.AwaitingDecisions();

  private enum RoundRun {
    STANDALONE,
    HANCHAN
  }

  private enum Stage {
    IDLE,
    BETWEEN_ROUNDS,
    RUNNING,
    AWAIT_TURN,
    AWAIT_RESPONSES,
    AWAIT_CALL_DAHAI,
    AWAIT_CHANKAN,
    ROUND_COMPLETE,
    HANCHAN_COMPLETE;

    private boolean awaitingDecision() {
      return this == AWAIT_TURN
          || this == AWAIT_RESPONSES
          || this == AWAIT_CALL_DAHAI
          || this == AWAIT_CHANKAN;
    }
  }

  private static final long NO_RESPONSE_RESOLUTION = Long.MIN_VALUE;

  private static final class RoundScratch {

    private final EngineDecisionBatch decisions = new EngineDecisionBatch();
    private final EngineOutcomeBuffer decisionOutcomes = new EngineOutcomeBuffer();
    private final EngineSelectionBuffer singleSelection = new EngineSelectionBuffer();
    private final EngineCommitResult commitResult = new EngineCommitResult();
    private final Action[] selectedActions = new Action[GameState.NUM_PLAYERS];
    private final int[] ronWinners = new int[3];

    private void clearSelections() {
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        selectedActions[player] = null;
      }
    }
  }

  private Stage stage = Stage.IDLE;
  private RoundRun roundRun;
  private RoundTransition.NextRound pendingNextRound;
  private RoundResult completedRound;
  private GameStepResult.HanchanEnded completedHanchanStep;
  private TurnEvent.Discard pendingDiscard;
  private TurnEvent.KanAttempt pendingKan;
  private int turnCompletionPlayer;
  private boolean turnCompletionRiichiDahai;
  private boolean riichiPaymentPending;
  private int ronDeclineCandidateMask;
  private long nextDecisionId = 1L;

  /** 通常の乱数牌山と記録処理なしで新しいエンジンを作る。 */
  public GameEngine() {
    this(new GameState(), null);
  }

  /**
   * 決定的な牌山シードで新しいエンジンを作る。
   *
   * @param wallSeed 牌山シャッフルのシード
   */
  public GameEngine(long wallSeed) {
    this(new GameState(wallSeed), null);
  }

  /**
   * 通常の乱数牌山と指定記録処理で新しいエンジンを作る。
   *
   * @param recorder 局進行コールバック。不要なら {@code null}
   */
  public GameEngine(GameRecorder recorder) {
    this(new GameState(), recorder);
  }

  /**
   * 決定的な牌山シードと指定記録処理で新しいエンジンを作る。
   *
   * @param wallSeed 牌山シャッフルのシード
   * @param recorder 局進行コールバック。不要なら {@code null}
   */
  public GameEngine(long wallSeed, GameRecorder recorder) {
    this(new GameState(wallSeed), recorder);
  }

  private GameEngine(GameState state, GameRecorder recorder) {
    this.state = state;
    this.recorder = recorder;
    winResolver = new RoundWinResolver(state);
    drawResolver = new RoundDrawResolver(state);
    actions = new MahjongTransition(state, recorder);
  }

  private GameEngine(GameEngine source) {
    this(source.state.snapshotCopy(), null);
    if (!source.stage.awaitingDecision()) {
      throw new IllegalStateException("GameEngine can only fork at an action boundary");
    }
    scratch.decisions.copyFrom(source.scratch.decisions);
    stage = source.stage;
    roundRun = source.roundRun;
    pendingNextRound = source.pendingNextRound;
    completedRound = source.completedRound;
    pendingDiscard =
        stage == Stage.AWAIT_RESPONSES ? (TurnEvent.Discard) state.getTurnEvent() : null;
    pendingKan = stage == Stage.AWAIT_CHANKAN ? (TurnEvent.KanAttempt) state.getTurnEvent() : null;
    turnCompletionPlayer = source.turnCompletionPlayer;
    turnCompletionRiichiDahai = source.turnCompletionRiichiDahai;
    riichiPaymentPending = source.riichiPaymentPending;
    ronDeclineCandidateMask = source.ronDeclineCandidateMask;
    nextDecisionId = source.nextDecisionId;
  }

  /**
   * 現在の行動境界を独立したエンジンとして複製する。
   *
   * <p>牌山、手牌、河、点棒、局進行を内部の可変データも含めて複製し、返却済み判断と次の判断ID を保つ。分岐側には記録処理
   * を引き継がないため、別の行動を選んだ場合の試行が元対局のログへ混入しない。この操作は {@link GameStepResult.AwaitingDecisions}
   * を返した直後だけ許可する。
   *
   * @return 現在境界から独立して進行できるエンジン
   */
  public GameEngine forkAtDecisionBoundary() {
    if (!stage.awaitingDecision()) {
      throw new IllegalStateException("GameEngine can only fork at an action boundary");
    }
    return new GameEngine(this);
  }

  /**
   * 段階的に進む半荘を次の行動境界または終了境界まで進める。
   *
   * <p>判断要求の取得、バッチ処理、行動の反映という流れのうち、判断要求の取得に相当する。
   *
   * @return 次の判断要求、局の精算、または半荘終了境界
   */
  public GameStepResult stepHanchan() {
    while (true) {
      if (stage == Stage.HANCHAN_COMPLETE) {
        return completedHanchanStep;
      }
      if (stage == Stage.IDLE) {
        beginHanchanStepping();
        continue;
      }
      if (stage == Stage.BETWEEN_ROUNDS) {
        startSteppedHanchanRound();
        continue;
      }
      if (stage == Stage.ROUND_COMPLETE) {
        if (roundRun == RoundRun.HANCHAN) {
          return settleSteppedHanchanRound(completedRound);
        }
        throw new IllegalStateException("Standalone round is in progress");
      }
      if (stage == Stage.RUNNING && roundRun == RoundRun.HANCHAN) {
        GameStepResult step = stepRoundToBoundary();
        if (step instanceof GameStepResult.AwaitingDecisions) {
          return step;
        }
        if (stage == Stage.ROUND_COMPLETE) {
          return settleSteppedHanchanRound(completedRound);
        }
        throw new IllegalStateException("Round ended without a completed round");
      }
      if (stage.awaitingDecision() && roundRun == RoundRun.HANCHAN) {
        return awaitingStep.bind(scratch.decisions);
      }
      throw new IllegalStateException("Standalone round is in progress");
    }
  }

  private GameStepResult settleSteppedHanchanRound(RoundResult result) {
    RoundSettlement settlement = settleRound(result);
    recordRoundEnd(settlement);

    return switch (settlement.transition()) {
      case RoundTransition.HanchanFinished ignored -> {
        GameStepResult.HanchanEnded terminal = new GameStepResult.HanchanEnded(settlement);
        completedHanchanStep = terminal;
        stage = Stage.HANCHAN_COMPLETE;
        yield terminal;
      }
      case RoundTransition.NextRound next -> {
        pendingNextRound = next;
        stage = Stage.BETWEEN_ROUNDS;
        yield new GameStepResult.RoundSettled(settlement);
      }
    };
  }

  /**
   * 現在の局を外部行動が必要になるか局結果に到達するまで進める。
   *
   * <p>局の初期化は呼び出し元の責務。
   */
  GameStepResult stepRound() {
    if (stage == Stage.IDLE || stage == Stage.ROUND_COMPLETE && roundRun == RoundRun.STANDALONE) {
      beginRoundStepping(RoundRun.STANDALONE);
    }
    if (stage == Stage.RUNNING && roundRun == RoundRun.STANDALONE) {
      return stepRoundToBoundary();
    }
    if (stage.awaitingDecision() && roundRun == RoundRun.STANDALONE) {
      return awaitingStep.bind(scratch.decisions);
    }
    throw new IllegalStateException("Use stepHanchan() while a hanchan is in progress");
  }

  /**
   * 直近の{@link #stepRound()}または{@link #stepHanchan()}が要求した行動をすべて反映する。
   *
   * @param selections 未処理の判断 IDごとの提出行動
   * @return 行動を反映した後に到達した次のエンジン境界
   */
  public GameStepResult commitDecisions(List<EngineDecisionSelection> selections) {
    return commitDecisionsWithOutcomes(selections).step();
  }

  /**
   * 行動を反映し、応答優先順位を解決した後の各選択の成立結果も返す。
   *
   * <p>PASS と通常の手番行動は常に実行済みとして扱う。CHI/PON/DAIMINKAN は RON や上位の鳴きに
   * 競り負けた場合だけ未実行になる。Decision学習上の因果性は実行可否と分け、他家の選択を固定したまま 自席の合法手を選び替えて応答解決を変えられる判断だけをCAUSALにする。
   *
   * @param selections 未処理の判断 IDごとの提出行動
   * @return 次のエンジン境界と、各選択の実行可否・Decision学習役割の一覧
   */
  public EngineCommitResult commitDecisionsWithOutcomes(List<EngineDecisionSelection> selections) {
    if (!stage.awaitingDecision()) {
      throw new IllegalStateException("No pending engine decisions");
    }
    Stage committedStage = stage;
    RoundRun committedRun = roundRun;
    try {
      fillSelectedActions(scratch.decisions, selections);
      long responseResolution =
          switch (committedStage) {
            case AWAIT_RESPONSES -> classifySelectedResponses(pendingDiscard.player());
            case AWAIT_CHANKAN -> responseResolution(collectRonWinners(pendingKan.player()), -1);
            default -> NO_RESPONSE_RESOLUTION;
          };
      List<EngineDecisionOutcome> outcomes =
          decisionOutcomes(scratch.decisions, responseResolution);
      GameStepResult step =
          switch (committedStage) {
            case AWAIT_TURN -> commitTurnDecision();
            case AWAIT_RESPONSES -> resolveCommittedResponses(responseResolution);
            case AWAIT_CALL_DAHAI -> commitPostCallDahai();
            case AWAIT_CHANKAN -> resolveCommittedChankan();
            default -> throw new IllegalStateException("Not a decision stage: " + committedStage);
          };
      if (committedRun == RoundRun.HANCHAN && step instanceof GameStepResult.RoundEnded) {
        step = stepHanchan();
      }
      return scratch.commitResult.bind(step, outcomes);
    } finally {
      scratch.clearSelections();
    }
  }

  GameStepResult commitDecision(long decisionId, Action action) {
    EngineSelectionBuffer selection = scratch.singleSelection;
    selection.clear();
    selection.add(decisionId, action);
    return commitDecisions(selection);
  }

  private void beginHanchanStepping() {
    pendingNextRound = new RoundTransition.NextRound(0, 0, 0, 0);
    completedHanchanStep = null;
    stage = Stage.BETWEEN_ROUNDS;
  }

  private void beginRoundStepping(RoundRun run) {
    roundRun = run;
    stage = Stage.RUNNING;
  }

  private void startSteppedHanchanRound() {
    RoundTransition.NextRound next = pendingNextRound;
    state.startRound(next.kyokuIndex(), next.oya(), next.honba(), next.kyotakuCount());
    state.dealInitialHands();
    recordRoundStart(next.kyokuIndex(), next.honba(), next.kyotakuCount());
    beginRoundStepping(RoundRun.HANCHAN);
  }

  private GameStepResult stepRoundToBoundary() {
    if (stage == Stage.RUNNING) {
      return prepareTurnDecision();
    }
    if (stage.awaitingDecision()) {
      return awaitingStep.bind(scratch.decisions);
    }
    if (stage == Stage.ROUND_COMPLETE) {
      return new GameStepResult.RoundEnded(completedRound);
    }
    throw new IllegalStateException("No round is in progress");
  }

  private GameStepResult prepareTurnDecision() {
    int player = state.getCurrentPlayer();
    TurnEvent.Draw draw;
    if (state.getTurnEvent() instanceof TurnEvent.Draw currentDraw && currentDraw.isRinshanDraw()) {
      draw = currentDraw;
    } else {
      if (state.isWallExhausted()) {
        return completeRound(drawResolver.resolveExhaustiveDraw());
      }
      draw = actions.drawFromWall(player);
    }

    EngineDecisionBatch decisions = scratch.decisions;
    decisions.clear();
    EngineDecisionPoint decision = appendDecision(EngineDecisionKind.TURN, player);
    actionGeneration.generateTurnActionsInto(decision, state, player, draw);
    stage = Stage.AWAIT_TURN;
    return awaitingStep.bind(decisions);
  }

  private GameStepResult commitTurnDecision() {
    EngineDecisionPoint point = scratch.decisions.getFirst();
    Action action = scratch.selectedActions[point.player()];
    return applyTurnDecision(point, action);
  }

  private GameStepResult applyTurnDecision(EngineDecisionPoint decision, Action action) {
    int player = decision.player();
    switch (action.type()) {
      case TSUMO_AGARI -> {
        RoundResult result = winResolver.resolveTsumo(player, decision.requireImmediateWinResult());
        return completeRound(result);
      }
      case DAHAI -> actions.discard(player, action.tileType(), false, action.tileSelection(), true);
      case RIICHI_DAHAI -> actions.declareRiichi(player, action);
      case ANKAN -> {
        state.recordKanAttempt(player, action.tileType(), false, TurnEvent.KanKind.ANKAN);
        actions.completeAnkan(player, action.tileType());
        state.clearAllIppatsu();
        RoundResult result = kanDrawOrAbort(player, false);
        if (result != null) {
          return completeRound(result);
        }
      }
      case KAKAN -> {
        boolean kakanIsAka = actions.declareKakan(player, action.tileType()).addedTileIsAka();
        GameStepResult chankan = prepareChankan(player, action.tileType(), kakanIsAka);
        if (chankan != null) {
          return chankan;
        }
        RoundResult result = kanDrawOrAbort(player, true);
        if (result != null) {
          return completeRound(result);
        }
      }
      case KYUSHU_KYUHAI -> {
        return completeRound(
            drawResolver.resolveAbortiveDraw(
                RoundResult.AbortiveDrawReason.NINE_TERMINALS_AND_HONORS));
      }
      case CHI, PON, DAIMINKAN, RON_AGARI, PASS ->
          throw new IllegalStateException("Response action reached turn phase: " + action);
    }

    boolean riichiDahai = action.type() == Action.Type.RIICHI_DAHAI;
    if (state.getTurnEvent() instanceof TurnEvent.Discard discard) {
      return prepareResponseDecisions(player, riichiDahai, riichiDahai, discard);
    }
    return finishTurnAndContinue(player, riichiDahai);
  }

  private GameStepResult prepareResponseDecisions(
      int completionPlayer,
      boolean completionRiichiDahai,
      boolean paymentPending,
      TurnEvent.Discard discard) {
    turnCompletionPlayer = completionPlayer;
    turnCompletionRiichiDahai = completionRiichiDahai;
    riichiPaymentPending = paymentPending;
    pendingDiscard = discard;
    EngineDecisionBatch decisions = scratch.decisions;
    decisions.clear();
    int declinedMask = 0;
    for (int responsePlayer = 0; responsePlayer < GameState.NUM_PLAYERS; responsePlayer++) {
      if (responsePlayer == discard.player()) {
        continue;
      }
      EngineDecisionPoint decision =
          decisions.append(nextDecisionId, EngineDecisionKind.RESPONSE, responsePlayer);
      RonStatus ronStatus =
          actionGeneration.generateResponseActionsInto(decision, state, responsePlayer, discard);
      if (ronStatus.createsImmediateFuriten()) {
        state.enterTemporaryFuriten(responsePlayer);
      } else if (ronStatus.createsFuritenWhenDeclined()) {
        declinedMask |= 1 << responsePlayer;
      }
      if (decision.mutableLegalActions().isEmpty()) {
        decisions.discardLast();
      } else {
        nextDecisionId++;
      }
    }
    if (decisions.isEmpty()) {
      applyDeclinedRonFuriten(declinedMask);
      if (paymentPending) {
        actions.commitRiichiPayment(completionPlayer);
      }
      state.advancePlayer();
      return finishTurnAndContinue(completionPlayer, completionRiichiDahai);
    }
    ronDeclineCandidateMask = declinedMask;
    stage = Stage.AWAIT_RESPONSES;
    return awaitingStep.bind(decisions);
  }

  private GameStepResult resolveCommittedResponses(long resolution) {
    TurnEvent.Discard discard = pendingDiscard;
    int ronCount = responseRonCount(resolution);
    if (ronCount > 0) {
      if (riichiPaymentPending) {
        actions.cancelRiichiDeclaration(turnCompletionPlayer);
      }
      if (ronCount == 3) {
        return completeRound(
            drawResolver.resolveAbortiveDraw(RoundResult.AbortiveDrawReason.TRIPLE_RON));
      }
      return completeRound(
          winResolver.resolveRons(scratch.ronWinners, ronCount, discard, scratch.decisions));
    }
    if (riichiPaymentPending) {
      actions.commitRiichiPayment(turnCompletionPlayer);
    }
    applyDeclinedRonFuriten(ronDeclineCandidateMask);

    int caller = responseMeldCaller(resolution);
    if (caller >= 0) {
      Action action = scratch.selectedActions[caller];
      if (action.type() == Action.Type.DAIMINKAN) {
        actions.completeDaiminkan(caller, discard);
        RoundResult result = kanDrawOrAbort(caller, true);
        if (result != null) {
          return completeRound(result);
        }
        return finishTurnAndContinue(turnCompletionPlayer, turnCompletionRiichiDahai);
      }
      if (action.type() == Action.Type.PON) {
        actions.applyPon(caller, discard, action.usesAkaTileFromHand());
        return preparePostCallDahai(caller, PostCallDahaiRestriction.afterPon(discard.tileType()));
      }
      if (action.type() == Action.Type.CHI) {
        actions.applyChi(caller, discard, action);
        return preparePostCallDahai(
            caller,
            PostCallDahaiRestriction.afterChi(
                action.chiSequenceBaseTileType(), discard.tileType()));
      }
      throw new IllegalStateException("Unexpected selected meld action: " + action);
    }

    state.advancePlayer();
    return finishTurnAndContinue(turnCompletionPlayer, turnCompletionRiichiDahai);
  }

  private GameStepResult preparePostCallDahai(int player, PostCallDahaiRestriction restriction) {
    EngineDecisionBatch decisions = scratch.decisions;
    decisions.clear();
    EngineDecisionPoint decision = appendDecision(EngineDecisionKind.POST_CALL_DAHAI, player);
    ActionGenerator.generateCallDahaiActionsInto(
        decision.mutableLegalActions(), state.hand(player), restriction);
    stage = Stage.AWAIT_CALL_DAHAI;
    return awaitingStep.bind(decisions);
  }

  private GameStepResult commitPostCallDahai() {
    EngineDecisionPoint point = scratch.decisions.getFirst();
    Action discardAction = scratch.selectedActions[point.player()];
    TurnEvent.Discard discard =
        actions.discard(
            point.player(), discardAction.tileType(), false, discardAction.tileSelection(), false);
    return prepareResponseDecisions(
        turnCompletionPlayer, turnCompletionRiichiDahai, false, discard);
  }

  private GameStepResult prepareChankan(int kanPlayer, int kanTileType, boolean addedTileIsAka) {
    TurnEvent.KanAttempt kanAttempt =
        state.recordKanAttempt(kanPlayer, kanTileType, addedTileIsAka, TurnEvent.KanKind.KAKAN);

    EngineDecisionBatch decisions = scratch.decisions;
    decisions.clear();
    int declinedRonFuritenCandidateMask = 0;
    for (int playerOffset = 1; playerOffset <= 3; playerOffset++) {
      int respondingPlayer = (kanPlayer + playerOffset) % GameState.NUM_PLAYERS;
      EngineDecisionPoint decision =
          decisions.append(nextDecisionId, EngineDecisionKind.CHANKAN, respondingPlayer);
      RonStatus ronStatus =
          actionGeneration.generateResponseActionsInto(
              decision, state, respondingPlayer, kanAttempt);
      if (ronStatus.createsImmediateFuriten()) {
        state.enterTemporaryFuriten(respondingPlayer);
      } else if (ronStatus.createsFuritenWhenDeclined()) {
        declinedRonFuritenCandidateMask |= 1 << respondingPlayer;
      }
      if (decision.mutableLegalActions().isEmpty()) {
        decisions.discardLast();
      } else {
        nextDecisionId++;
      }
    }
    if (decisions.isEmpty()) {
      applyDeclinedRonFuriten(declinedRonFuritenCandidateMask);
      state.clearAllIppatsu();
      return null;
    }
    pendingKan = kanAttempt;
    ronDeclineCandidateMask = declinedRonFuritenCandidateMask;
    stage = Stage.AWAIT_CHANKAN;
    return awaitingStep.bind(decisions);
  }

  private GameStepResult resolveCommittedChankan() {
    TurnEvent.KanAttempt kan = pendingKan;
    int ronCount = collectRonWinners(kan.player());
    if (ronCount > 0) {
      if (ronCount == 3) {
        return completeRound(
            drawResolver.resolveAbortiveDraw(RoundResult.AbortiveDrawReason.TRIPLE_RON));
      }
      RoundResult result =
          winResolver.resolveRons(scratch.ronWinners, ronCount, kan, scratch.decisions);
      return completeRound(result);
    }
    applyDeclinedRonFuriten(ronDeclineCandidateMask);
    state.clearAllIppatsu();

    RoundResult result = kanDrawOrAbort(kan.player(), true);
    if (result != null) {
      return completeRound(result);
    }
    return finishTurnAndContinue(kan.player(), false);
  }

  private GameStepResult finishTurnAndContinue(int turnPlayer, boolean riichiDahai) {
    RoundResult abort = finishTurnAfterResponses(turnPlayer, riichiDahai);
    if (abort != null) {
      return completeRound(abort);
    }
    stage = Stage.RUNNING;
    return prepareTurnDecision();
  }

  private GameStepResult completeRound(RoundResult result) {
    if (recorder != null) {
      recorder.recordRoundResult(result);
    }
    completedRound = result;
    stage = Stage.ROUND_COMPLETE;
    return new GameStepResult.RoundEnded(result);
  }

  private EngineDecisionPoint appendDecision(EngineDecisionKind kind, int player) {
    return scratch.decisions.append(nextDecisionId++, kind, player);
  }

  private void fillSelectedActions(
      List<EngineDecisionPoint> decisions, List<EngineDecisionSelection> selections) {
    if (selections.size() != decisions.size()) {
      throw new IllegalArgumentException(
          "Expected " + decisions.size() + " selections, got " + selections.size());
    }
    for (int decisionIndex = 0; decisionIndex < decisions.size(); decisionIndex++) {
      EngineDecisionPoint decision = decisions.get(decisionIndex);
      EngineDecisionSelection selection = selectionFor(decision.id(), selections);
      if (selection == null) {
        throw new IllegalArgumentException("Missing selection for decision " + decision.id());
      }
      if (!decision.legalActions().contains(selection.action())) {
        throw new IllegalArgumentException(
            "Committed illegal action: decision=" + decision + " action=" + selection.action());
      }
      scratch.selectedActions[decision.player()] = selection.action();
    }
  }

  private static EngineDecisionSelection selectionFor(
      long decisionId, List<EngineDecisionSelection> selections) {
    for (int selectionIndex = 0; selectionIndex < selections.size(); selectionIndex++) {
      EngineDecisionSelection selection = selections.get(selectionIndex);
      if (selection.decisionId() == decisionId) {
        return selection;
      }
    }
    return null;
  }

  /** 槓後の途中流局または嶺上ツモを処理する。 */
  private RoundResult kanDrawOrAbort(int player, boolean doraRevealPending) {
    RoundResult suukan = drawResolver.afterKan(player);
    if (suukan != null) {
      return suukan;
    }
    actions.drawFromDeadWall(player, doraRevealPending);
    return null;
  }

  private int selectPonOrKanCaller(int discardingPlayer) {
    for (int playerOffset = 1; playerOffset <= 3; playerOffset++) {
      int player = (discardingPlayer + playerOffset) % GameState.NUM_PLAYERS;
      Action action = scratch.selectedActions[player];
      if (action == null) {
        continue;
      }
      Action.Type type = action.type();
      if (type == Action.Type.PON || type == Action.Type.DAIMINKAN) {
        return player;
      }
    }
    return -1;
  }

  private long classifySelectedResponses(int discardingPlayer) {
    int ronCount = collectRonWinners(discardingPlayer);
    if (ronCount > 0) {
      return responseResolution(ronCount, -1);
    }
    int caller = selectPonOrKanCaller(discardingPlayer);
    if (caller >= 0) {
      return responseResolution(0, caller);
    }
    int shimocha = (discardingPlayer + 1) % GameState.NUM_PLAYERS;
    Action shimochaAction = scratch.selectedActions[shimocha];
    int chiCaller =
        shimochaAction != null && shimochaAction.type() == Action.Type.CHI ? shimocha : -1;
    return responseResolution(0, chiCaller);
  }

  private List<EngineDecisionOutcome> decisionOutcomes(
      List<EngineDecisionPoint> decisions, long responseResolution) {
    EngineOutcomeBuffer outcomes = scratch.decisionOutcomes;
    outcomes.clear();
    for (int decisionIndex = 0; decisionIndex < decisions.size(); decisionIndex++) {
      EngineDecisionPoint decision = decisions.get(decisionIndex);
      Action selectedAction = scratch.selectedActions[decision.player()];
      boolean selectedActionWasExecuted =
          responseResolution == NO_RESPONSE_RESOLUTION
              || selectedAction.type() == Action.Type.PASS
              || (responseRonCount(responseResolution) > 0
                  ? selectedAction.type() == Action.Type.RON_AGARI
                  : responseMeldCaller(responseResolution) < 0
                      || responseMeldCaller(responseResolution) == decision.player());
      boolean decisionWasCausal = canDecisionAffectResponseResolution(decision, responseResolution);
      DecisionLearningRole role =
          !decisionWasCausal
              ? DecisionLearningRole.PREEMPTED
              : decision.legalActions().size() == 1
                  ? DecisionLearningRole.FORCED
                  : DecisionLearningRole.CAUSAL;
      outcomes.append(
          decision.id(), decision.player(), selectedAction, selectedActionWasExecuted, role);
    }
    return outcomes;
  }

  /**
   * 他家の選択を固定し、自席の合法手を選び替えた時に応答解決を変えられるかを判定する。
   *
   * <p>合法牌姿で同時に競合し得る非RONの鳴きは、一席のPON/DAIMINKANと下家のCHIだけである。同じ牌への
   * PON同士、PON対DAIMINKAN、CHI同士は牌数または席制約により発生しない。
   */
  private boolean canDecisionAffectResponseResolution(
      EngineDecisionPoint decision, long responseResolution) {
    if (responseResolution == NO_RESPONSE_RESOLUTION
        || hasLegalAction(decision, Action.Type.RON_AGARI)) {
      return true;
    }
    if (responseRonCount(responseResolution) > 0) {
      return false;
    }
    int meldCaller = responseMeldCaller(responseResolution);
    if (meldCaller < 0 || meldCaller == decision.player()) {
      return true;
    }
    Action.Type winningMeld = scratch.selectedActions[meldCaller].type();
    return winningMeld == Action.Type.CHI
        && (hasLegalAction(decision, Action.Type.PON)
            || hasLegalAction(decision, Action.Type.DAIMINKAN));
  }

  private static boolean hasLegalAction(EngineDecisionPoint decision, Action.Type type) {
    List<Action> legalActions = decision.legalActions();
    for (int actionIndex = 0; actionIndex < legalActions.size(); actionIndex++) {
      if (legalActions.get(actionIndex).type() == type) {
        return true;
      }
    }
    return false;
  }

  private static long responseResolution(int ronCount, int meldCaller) {
    return (long) ronCount << 32 | meldCaller & 0xffff_ffffL;
  }

  private static int responseRonCount(long resolution) {
    return (int) (resolution >>> 32);
  }

  private static int responseMeldCaller(long resolution) {
    return (int) resolution;
  }

  /**
   * 現在処理待ちになっている判断を、呼び出し側の再利用バッファへ解析する。
   *
   * <p>状態、合法手、公開牌はエンジンが所有する停止境界から取得する。行動を反映した後の古い判断や別エンジンの判断は受け付けない。
   */
  public void analyzeDecision(EngineDecisionPoint decision, EngineDecisionBuffer destination) {
    if (!stage.awaitingDecision() || !containsDecision(scratch.decisions, decision)) {
      throw new IllegalStateException("decision is not pending in this engine: " + decision.id());
    }
    destination.bind(state, decision, state.publicState());
  }

  private static boolean containsDecision(
      List<EngineDecisionPoint> pending, EngineDecisionPoint expected) {
    for (int index = 0; index < pending.size(); index++) {
      if (pending.get(index) == expected) {
        return true;
      }
    }
    return false;
  }

  private int collectRonWinners(int discardingPlayer) {
    int ronCount = 0;
    // 放銃者からツモ順に席をたどって集める。先頭の和了者を供託の受取人とする。
    for (int playerOffset = 1; playerOffset <= 3; playerOffset++) {
      int player = (discardingPlayer + playerOffset) % GameState.NUM_PLAYERS;
      if (scratch.selectedActions[player] != null
          && scratch.selectedActions[player].type() == Action.Type.RON_AGARI) {
        scratch.ronWinners[ronCount++] = player;
      }
    }
    return ronCount;
  }

  private void applyDeclinedRonFuriten(int candidateMask) {
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      if ((candidateMask & (1 << player)) == 0) {
        continue;
      }
      Action selectedAction = scratch.selectedActions[player];
      if (selectedAction == null || selectedAction.type() != Action.Type.RON_AGARI) {
        state.enterTemporaryFuriten(player);
      }
    }
  }

  private RoundResult finishTurnAfterResponses(int player, boolean riichiDahai) {
    RoundResult abort = drawResolver.afterDiscard(riichiDahai);
    if (abort != null) {
      return abort;
    }

    if (!riichiDahai) {
      state.setIppatsu(player, false);
    }
    return null;
  }

  RoundSettlement settleRound(RoundResult result) {
    return HanchanProgression.settle(state, result);
  }

  private void recordRoundStart(int roundIndex, int honba, int kyotakuCount) {
    if (recorder == null) {
      return;
    }
    int[] startingScores = snapshotScores();
    recorder.startRound(roundIndex, honba, kyotakuCount, startingScores);
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      Hand hand = state.hand(player);
      recorder.recordHaipai(player, hand.copyConcealedTileCounts(), hand.concealedAkaMask());
    }
    int indicatorCount = state.doraState().indicatorCount();
    int[] indicatorPhysicalTileIds = new int[indicatorCount];
    for (int index = 0; index < indicatorCount; index++) {
      indicatorPhysicalTileIds[index] = state.doraIndicatorPhysicalTileId(index);
    }
    recorder.recordDoraIndicators(indicatorPhysicalTileIds);
  }

  private void recordRoundEnd(RoundSettlement settlement) {
    if (recorder == null) {
      return;
    }
    RoundResult result = settlement.result();
    boolean showUra = result instanceof RoundResult.Winning winning && winning.hasRiichiWinner();
    int[] uraPhysicalTileIds;
    if (showUra) {
      int indicatorCount = state.doraState().indicatorCount();
      uraPhysicalTileIds = new int[indicatorCount];
      for (int index = 0; index < indicatorCount; index++) {
        uraPhysicalTileIds[index] = state.uraDoraIndicatorPhysicalTileId(index);
      }
    } else {
      uraPhysicalTileIds = new int[0];
    }
    recorder.endRound(
        new GameRecorder.RoundEnd(settlement.snapshotFinalScores(), uraPhysicalTileIds));
  }

  private int[] snapshotScores() {
    int[] scores = new int[GameState.NUM_PLAYERS];
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      scores[player] = state.getScore(player);
    }
    return scores;
  }

  /**
   * エンジンが所有する現在の変更可能対局状態を返す。
   *
   * @return 現在の対局状態
   */
  public GameState getState() {
    return state;
  }
}
