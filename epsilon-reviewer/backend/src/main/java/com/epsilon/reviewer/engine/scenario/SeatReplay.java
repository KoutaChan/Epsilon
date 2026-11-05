package com.epsilon.reviewer.engine.scenario;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineActionBuffer;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.engine.MahjongTransition;
import com.epsilon.engine.PostCallDiscards;
import com.epsilon.engine.WinLegality.RonStatus;
import com.epsilon.replay.ReplayEngine;
import com.epsilon.replay.ReplayEvent;
import com.epsilon.replay.ReplayEvent.Ankan;
import com.epsilon.replay.ReplayEvent.Chi;
import com.epsilon.replay.ReplayEvent.Dahai;
import com.epsilon.replay.ReplayEvent.Daiminkan;
import com.epsilon.replay.ReplayEvent.Dora;
import com.epsilon.replay.ReplayEvent.EndGame;
import com.epsilon.replay.ReplayEvent.EndKyoku;
import com.epsilon.replay.ReplayEvent.Hora;
import com.epsilon.replay.ReplayEvent.Kakan;
import com.epsilon.replay.ReplayEvent.None;
import com.epsilon.replay.ReplayEvent.Pon;
import com.epsilon.replay.ReplayEvent.Reach;
import com.epsilon.replay.ReplayEvent.ReachAccepted;
import com.epsilon.replay.ReplayEvent.Ryukyoku;
import com.epsilon.replay.ReplayEvent.StartGame;
import com.epsilon.replay.ReplayEvent.StartKyoku;
import com.epsilon.replay.ReplayEvent.Tsumo;
import com.epsilon.reviewer.engine.scenario.LocatedDecision.Kind;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.List;

/**
 * 1 局の出来事を判断の席の目で再生し、その席の 1 判断の局面と合法手を作る（もしも）。変更した出来事でも進められるよう、他家の伏せた牌は持たない
 * （ツモは牌を見ず、打牌は手を減らさない）。他家の副露は、鳴いた時に見せた牌だけを手へ置いてから自家と同じ遷移で作るので、面子と公開の記録は
 * 通常の再生と同じになる。判断の席の手は変更した配牌とツモから作るので、判断までの窓・振聴・リーチの可否も変更した手に従う。
 *
 * <p>判断の時点、応答の受付、見逃しの振聴、槓ドラを開く時点は {@link ReplayEngine} と同じ規則に従う。応答の窓は判断の席の合法手だけを持ち、他家の
 * 応答は牌譜のとおりに受け付ける（他家の合法手は伏せた牌がないと作れない）。他家の伏せた手の枚数は 0 のままで、どのモデルも読まない。
 */
final class SeatReplay {
  /** 配牌で牌山から取った牌の数。 */
  private static final int DEALT_TILES = 52;

  private final List<ReplayEvent> events;
  private final int seat;

  /** 判断の出来事の位置（局の始めからの位置）。 */
  private final int target;

  private final Kind kind;
  private final GameState state = new GameState();
  private final MahjongTransition transition = new MahjongTransition(state);
  private final ActionGenerator generator = new ActionGenerator();

  /** 手番の判断の合法手。 */
  private final EngineActionBuffer turnActions = new EngineActionBuffer();

  /** 持ち点が足りなくても宣言済みなら足すリーチの打牌（{@link ReplayEngine} の牌譜のリーチと同じ扱い）。 */
  private final EngineActionBuffer declaredRiichi = new EngineActionBuffer();

  /** 開いている窓での判断の席の応答の合法手（窓を開いた時に作る）。 */
  private final EngineActionBuffer responses = new EngineActionBuffer();

  private final boolean[] reachPending = new boolean[GameState.NUM_PLAYERS];

  /** 鳴いた後の打牌を待つ間の、直前のチー・ポン。ほかは null。 */
  private Meld call;

  /** すぐ後の打牌の後に開く大明槓・加槓の新ドラ（なければ null）。 */
  private Dora doraAfterKan;

  /** 打牌・加槓への応答の受付（なければ null）。 */
  private Window window;

  private int cursor;

  private SeatReplay(List<ReplayEvent> events, LocatedDecision decision) {
    this.events = events;
    seat = decision.seat();
    target = decision.target();
    kind = decision.kind();
  }

  /**
   * 局を判断まで再生する。
   *
   * @param round 局の出来事（局の始めから）
   */
  static Decision replay(List<ReplayEvent> round, LocatedDecision decision) {
    return new SeatReplay(round, decision).run();
  }

  /**
   * 応答の判断で判断の席がチー・ポンをした後の、打牌の局面と合法手。他家の応答は無かったとし、ロンを見送って鳴くなら振聴にする（牌譜の鳴きと
   * 同じ）。大明槓は嶺上のツモが分からないので扱わない。
   *
   * @param call 鳴く手の {@link Action#toIndex()}（判断の席の合法な応答）
   */
  static Decision afterCall(List<ReplayEvent> round, LocatedDecision decision, int call) {
    if (decision.kind() != Kind.RESPONSE)
      throw invalidEdit("call: the decision is not a response to a discard.");
    var replay = new SeatReplay(round, decision);
    for (Action action : replay.run().legal())
      if (action.toIndex() == call) return replay.call(action);
    throw invalidEdit("call: the call is not a legal response at the decision.");
  }

  /**
   * 判断の局面と合法手。局面は再生の持ち物で、評価の間だけ借りる。応答の合法手は窓を開いた時の手で、変更で応答できなくなった窓は見送りだけになる。
   */
  record Decision(GameState state, List<Action> legal) {
    /** 合法手の中の手の位置。合法でなければ -1。 */
    int slotOf(Action action) {
      if (action != null)
        for (int slot = 0; slot < legal.size(); slot++)
          if (legal.get(slot).toIndex() == action.toIndex()) return slot;
      return -1;
    }
  }

  private Decision run() {
    while (cursor < events.size()) {
      boolean reached = window == null ? play() : respond();
      if (reached) return decision();
    }
    // 牌譜が窓の途中で終わったときは、牌譜の終わりの位置で判断する（ReplayEngine と同じ）。
    if (window != null && kind == Kind.RESPONSE && cursor == target) return decision();
    throw new IllegalStateException("The replay did not reach the decision at event " + target);
  }

  private Decision decision() {
    if (kind == Kind.TURN) return new Decision(state, turnActions);
    return new Decision(state, responses.isEmpty() ? List.of(Action.pass()) : responses);
  }

  /** 窓のない位置の出来事を 1 つ適用する。手番の判断がこの出来事なら、適用せずに合法手を作って止める。 */
  private boolean play() {
    ReplayEvent event = events.get(cursor);
    if (kind == Kind.TURN && cursor == target) {
      prepareTurnActions(event);
      return true;
    }
    apply(event);
    cursor++;
    return false;
  }

  /**
   * 窓を開いた後の出来事を、{@link ReplayEngine} の応答の解決と同じ順に進める。応答の前の公開（ドラ・リーチの成立）は窓のまま適用し、続けて
   * 受け付ける応答の並びでは、判断の席の判断・見逃しの振聴・副露の順に処理して窓を閉じる。応答の判断なら、その判断の位置で止める。
   */
  private boolean respond() {
    ReplayEvent event = events.get(cursor);
    if (window.preserves(event)) {
      apply(event);
      cursor++;
      return false;
    }
    if (kind == Kind.RESPONSE && cursor == target) return true;
    int end = cursor;
    boolean selected = false;
    while (end < events.size() && accepts(end)) {
      selected |= events.get(end).responseActor() == seat;
      end++;
    }
    if (window.furitenWhenDeclined() && !selected) state.enterTemporaryFuriten(seat);
    boolean robbed = false;
    for (; cursor < end; cursor++) {
      if (kind == Kind.RESPONSE && cursor == target) return true;
      switch (events.get(cursor)) {
        case None ignored -> {}
        case Hora ignored -> robbed = true;
        case ReplayEvent response -> applyCall(response);
      }
    }
    close(robbed);
    return false;
  }

  /**
   * 位置 index の出来事を窓の応答として受け付けるか。形は {@link ReplayEngine} と同じで、他家の応答は牌譜のとおりに受け付ける。判断の席は合法な
   * 応答があるときだけ受け付けるが、変更で応答できなくなった判断の位置は判断のために受け付ける。
   */
  private boolean accepts(int index) {
    ReplayEvent event = events.get(index);
    int actor = event instanceof None none ? none.actor() : event.responseActor();
    if (actor < 0 || actor == window.player()) return false;
    if (actor == seat && !window.decides() && !(kind == Kind.RESPONSE && index == target))
      return false;
    return switch (event) {
      case None ignored -> true;
      case Hora win -> win.target() == window.player();
      case Chi chi -> window.source() == Source.DAHAI && chi.target() == window.player();
      case Pon pon -> window.source() == Source.DAHAI && pon.target() == window.player();
      case Daiminkan kan -> window.source() == Source.DAHAI && kan.target() == window.player();
      default -> false;
    };
  }

  private void apply(ReplayEvent event) {
    switch (event) {
      case StartKyoku start -> start(start);
      case Tsumo draw -> draw(draw);
      case Dahai discard -> discard(discard);
      case Ankan kan -> closedKan(kan);
      case Kakan kan -> addedKan(kan);
      case Reach reach -> reachPending[reach.actor()] = true;
      case ReachAccepted accepted -> transition.commitRiichiPayment(accepted.actor());
      case Dora dora -> {
        if (revealsAfterTheNextDiscard()) doraAfterKan = dora;
        else reveal(dora);
      }
      // 局の終わりの精算は判断の後、対局の始めと終わりは局の外、明示の見送りは局面を変えないので写さない。
      case Hora ignored -> {}
      case Ryukyoku ignored -> {}
      case EndKyoku ignored -> {}
      case EndGame ignored -> {}
      case StartGame ignored -> {}
      case None ignored -> {}
      case Chi ignored -> throw responseOutsideWindow(event);
      case Pon ignored -> throw responseOutsideWindow(event);
      case Daiminkan ignored -> throw responseOutsideWindow(event);
    }
  }

  /** 局を始める。判断の席の配牌だけを持ち、牌山は公開の情報だけで作る（記録の牌山は使わない）。 */
  private void start(StartKyoku start) {
    for (int player = 0; player < GameState.NUM_PLAYERS; player++)
      state.setScore(player, start.scores()[player]);
    state.startRoundForReconstruction(
        (start.wind() - Tile.TON) * GameState.NUM_PLAYERS + start.kyoku() - 1,
        start.oya(),
        start.honba(),
        start.kyotaku());
    state.initializeWallForReconstruction(new int[0], DEALT_TILES);
    state.revealDoraIndicator(start.doraIndicatorTileType(), start.doraIndicatorAka());
    for (int physicalTileId : start.initialHandPhysicalTileIds()[seat])
      state.hand(seat).addPhysicalTile(physicalTileId);
  }

  private void draw(Tsumo draw) {
    TurnEvent event = state.getTurnEvent();
    boolean rinshan =
        event instanceof TurnEvent.KanAttempt
            || event instanceof TurnEvent.Draw begun && begun.isRinshanDraw();
    transition.reconstructDraw(
        draw.actor(), draw.actor() == seat ? draw.physicalTileId() : -1, rinshan);
  }

  private void discard(Dahai discard) {
    int actor = discard.actor();
    Action action = discardAction(discard);
    reachPending[actor] = false;
    transition.reconstructDiscard(actor, action, call == null, actor == seat);
    call = null;
    if (doraAfterKan != null) {
      reveal(doraAfterKan);
      doraAfterKan = null;
    }
    open(Source.DAHAI, actor);
  }

  private Action discardAction(Dahai discard) {
    int type = Tile.typeOf(discard.physicalTileId());
    boolean red = Tile.isAka(discard.physicalTileId());
    return reachPending[discard.actor()]
        ? Action.riichiDahai(type, red, discard.tsumogiri())
        : Action.dahai(type, red, discard.tsumogiri());
  }

  private void closedKan(Ankan kan) {
    show(kan.actor(), kan.consumedPhysicalTileIds());
    transition.completeAnkan(kan.actor(), Tile.typeOf(kan.consumedPhysicalTileIds()[0]));
    transition.beginReconstructionRinshan(kan.actor(), false);
    call = null;
  }

  private void addedKan(Kakan kan) {
    int added = kan.addedPhysicalTileId();
    show(kan.actor(), added);
    transition.declareKakan(kan.actor(), Tile.typeOf(added));
    transition.recordKanAttempt(
        kan.actor(), Tile.typeOf(added), Tile.isAka(added), TurnEvent.KanKind.KAKAN);
    open(Source.KAKAN, kan.actor());
  }

  /** 打牌・加槓への応答を受け付ける。判断の席の合法手と振聴は、ここで窓を開いた時の局面で決める。 */
  private void open(Source source, int player) {
    responses.clear();
    boolean decides = false;
    boolean furitenWhenDeclined = false;
    if (player != seat) {
      RonStatus status =
          generator.generateResponseActionsInto(
              responses, state, seat, (TurnEvent.ResponseSource) state.getTurnEvent());
      if (status.createsImmediateFuriten()) state.enterTemporaryFuriten(seat);
      decides = responses.size() > 1;
      furitenWhenDeclined = status.createsFuritenWhenDeclined();
    }
    window = new Window(source, player, decides, furitenWhenDeclined);
  }

  /** 受け付けたチー・ポン・大明槓を適用する。判断の席がロンを見送って鳴いたなら、先に振聴にする。 */
  private void applyCall(ReplayEvent response) {
    int actor = response.responseActor();
    if (actor == seat && window.furitenWhenDeclined()) state.enterTemporaryFuriten(seat);
    var discard = (TurnEvent.Discard) state.getTurnEvent();
    switch (response) {
      case Chi chi -> {
        show(actor, chi.consumedPhysicalTileIds());
        afterCall(actor, transition.applyChi(actor, discard, chiAction(chi)));
      }
      case Pon pon -> {
        show(actor, pon.consumedPhysicalTileIds());
        afterCall(
            actor,
            transition.applyPon(actor, discard, containsAka(pon.consumedPhysicalTileIds())));
      }
      case Daiminkan kan -> {
        show(actor, kan.consumedPhysicalTileIds());
        transition.completeDaiminkan(actor, discard);
        transition.beginReconstructionRinshan(actor, true);
        call = null;
      }
      default -> throw new IllegalStateException("Unexpected response event: " + response);
    }
  }

  private static Action chiAction(Chi chi) {
    int called = Tile.typeOf(chi.calledPhysicalTileId());
    int[] consumed = chi.consumedPhysicalTileIds();
    int base = Math.min(called, Math.min(Tile.typeOf(consumed[0]), Tile.typeOf(consumed[1])));
    return Action.chiSequence(base, called, containsAka(consumed));
  }

  /** 応答の判断の窓で判断の席が鳴き、鳴いた後の打牌の合法手（喰い替えを除く）を作る。 */
  private Decision call(Action action) {
    if (action.type() != Action.Type.CHI && action.type() != Action.Type.PON)
      throw invalidEdit("call: only a chi or a pon is followed by a discard.");
    var discard = (TurnEvent.Discard) state.getTurnEvent();
    if (window.furitenWhenDeclined()) state.enterTemporaryFuriten(seat);
    Meld meld =
        action.type() == Action.Type.CHI
            ? transition.applyChi(seat, discard, action)
            : transition.applyPon(seat, discard, action.usesAkaTileFromHand());
    afterCall(seat, meld);
    window = null;
    turnActions.clear();
    PostCallDiscards.generateInto(state.hand(seat), meld, turnActions);
    return new Decision(state, turnActions);
  }

  private void afterCall(int actor, Meld meld) {
    state.clearAllIppatsu();
    state.setCurrentPlayer(actor);
    state.clearTurnEvent();
    call = meld;
  }

  /** 窓を閉じる。加槓は槍槓されなければ成立し、嶺上ツモの境へ進む。 */
  private void close(boolean robbed) {
    if (window.source() == Source.KAKAN) {
      if (state.getTurnEvent() instanceof TurnEvent.KanAttempt) state.clearTurnEvent();
      if (!robbed) {
        transition.beginReconstructionRinshan(window.player(), true);
        call = null;
      }
    }
    window = null;
  }

  /** 他家が副露で見せた牌を手へ置く（自家と同じ遷移がこの牌で面子を作り、手から外す）。 */
  private void show(int actor, int... physicalTileIds) {
    if (actor == seat) return;
    for (int physicalTileId : physicalTileIds) state.hand(actor).addPhysicalTile(physicalTileId);
  }

  /** 今のドラ表示が、すぐ後の打牌の後に開く大明槓・加槓の新ドラか。 */
  private boolean revealsAfterTheNextDiscard() {
    int next = cursor + 1;
    return next < events.size()
        && events.get(next) instanceof Dahai discard
        && state.getTurnEvent() instanceof TurnEvent.Draw draw
        && draw.player() == discard.actor()
        && draw.doraRevealPending();
  }

  private void reveal(Dora dora) {
    state.revealDoraIndicator(dora.indicatorTileType(), dora.indicatorAka());
  }

  /** 手番の判断の合法手を、{@link ReplayEngine} がその出来事に渡す合法手と同じに作る。 */
  private void prepareTurnActions(ReplayEvent event) {
    switch (event) {
      case Dahai discard when discard.actor() == seat -> prepareDiscardActions();
      case Ankan kan when kan.actor() == seat -> prepareDrawActions();
      case Kakan kan when kan.actor() == seat -> prepareDrawActions();
      case Hora win when win.actor() == seat && win.target() == seat -> prepareDrawActions();
      case Ryukyoku draw when draw.isNineTerminals() && state.getCurrentPlayer() == seat ->
          prepareDrawActions();
      default ->
          throw new IllegalStateException("Event " + target + " is not a turn of seat " + seat);
    }
  }

  private void prepareDrawActions() {
    generator.generateTurnActionsInto(
        turnActions, state, seat, (TurnEvent.Draw) state.getTurnEvent());
  }

  private void prepareDiscardActions() {
    if (call != null) {
      PostCallDiscards.generateInto(state.hand(seat), call, turnActions);
      return;
    }
    prepareDrawActions();
    if (reachPending[seat]
        && state.getScore(seat) < HanchanProgression.RIICHI_COST
        && !state.isRiichi(seat)
        && state.hand(seat).isMenzen()
        && state.remainingWallTiles() >= 4) {
      var draw = (TurnEvent.Draw) state.getTurnEvent();
      generator.generateRiichiDahaiActionsInto(declaredRiichi, state.hand(seat), draw);
      turnActions.addAll(declaredRiichi);
    }
  }

  private static boolean containsAka(int... physicalTileIds) {
    for (int physicalTileId : physicalTileIds) if (Tile.isAka(physicalTileId)) return true;
    return false;
  }

  private static IllegalStateException responseOutsideWindow(ReplayEvent event) {
    return new IllegalStateException("Response event outside response window: " + event);
  }

  private static ReviewerException invalidEdit(String message) {
    return new ReviewerException(ErrorCode.INVALID_EDIT, message);
  }

  /** 応答の元。 */
  private enum Source {
    DAHAI,
    /** 加槓（槍槓の窓）。 */
    KAKAN
  }

  /**
   * 1 回の打牌か加槓への応答の受付。
   *
   * @param player 打牌・加槓をした席
   * @param decides 判断の席に、見送りのほかの合法な応答がある
   * @param furitenWhenDeclined 判断の席がロンを見送ると振聴になる
   */
  private record Window(Source source, int player, boolean decides, boolean furitenWhenDeclined) {
    /** 応答の前の公開（打牌の後のドラ、打牌した席のリーチの成立）。窓を閉じずに適用する。 */
    boolean preserves(ReplayEvent event) {
      return switch (event) {
        case Dora ignored -> source == Source.DAHAI;
        case ReachAccepted accepted -> accepted.actor() == player;
        default -> false;
      };
    }
  }
}
