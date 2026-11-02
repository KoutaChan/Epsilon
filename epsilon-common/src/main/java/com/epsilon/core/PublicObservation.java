package com.epsilon.core;

import com.epsilon.calculate.scoring.HandScoreEvaluator;
import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.WinSettlementProjection;
import java.util.List;

/** 自家手牌と公開情報のみを読む、席に固定された借用参照。 エンジンが停止している間だけ有効。GameState、山、他家手牌への入口を公開しない。 */
public final class PublicObservation {
  private final GameState state;
  private final int observer;
  private final VisibleHand[] hands = new VisibleHand[4];
  private final VisibleDora dora = new VisibleDora();
  private final TurnEvent.Draw draw = new TurnEvent.Draw(-1, -1, TurnEvent.DrawSource.WALL, false);
  private final TurnEvent.Discard discard = new TurnEvent.Discard(-1, -1, false);
  private final TurnEvent.KanAttempt kan =
      new TurnEvent.KanAttempt(-1, -1, false, TurnEvent.KanKind.ANKAN, false);

  public PublicObservation(GameState state, int observer) {
    this.state = state;
    this.observer = observer;
    for (int seat = 0; seat < 4; seat++) hands[seat] = new VisibleHand(seat);
  }

  public int observer() {
    return observer;
  }

  public int currentPlayer() {
    return state.currentPlayer();
  }

  public int relativePosition(int from, int to) {
    return state.relativePosition(from, to);
  }

  public int roundIndex() {
    return state.roundIndex();
  }

  public int dealer() {
    return state.dealer();
  }

  public int honba() {
    return state.honba();
  }

  public int riichiSticks() {
    return state.riichiSticks();
  }

  public int roundWindTileType() {
    return state.roundWindTileType();
  }

  public int seatWindTileType(int player) {
    return state.seatWindTileType(player);
  }

  public int turnNumber() {
    return state.turnNumber();
  }

  public int score(int player) {
    return state.score(player);
  }

  public boolean isBeforeFirstDiscard(int player) {
    return state.isBeforeFirstDiscard(player);
  }

  public boolean firstTurnCallOccurred() {
    return state.firstTurnCallOccurred();
  }

  public boolean isRiichi(int player) {
    return state.isRiichi(player);
  }

  public boolean isDoubleRiichi(int player) {
    return state.isDoubleRiichi(player);
  }

  public boolean isIppatsu(int player) {
    return state.isIppatsu(player);
  }

  public boolean isFirstTurn() {
    return state.isFirstTurn();
  }

  public boolean isWallExhausted() {
    return state.isWallExhausted();
  }

  public int remainingWallTiles() {
    return state.remainingWallTiles();
  }

  public int totalRiverDahaiCount() {
    return state.totalRiverDahaiCount();
  }

  public boolean isTemporaryFuriten(int player) {
    requireObserver(player);
    return state.isTemporaryFuriten(player);
  }

  public VisibleHand hand(int player) {
    return hands[player];
  }

  public River river(int player) {
    return state.river(player);
  }

  public RoundPublicStateIndex publicState() {
    return state.publicState();
  }

  /** 局の公開の出来事の記録（打牌・副露・加槓。どの席にも公開の事実だけを持つ）。 */
  public PublicEventLog events() {
    return state.events();
  }

  public VisibleDora doraState() {
    return dora;
  }

  /** 組み込みルール解析へ借用局面を渡す。解析結果からも完全状態は取り出せない。 */
  public void analyze(List<Action> legalActions, EngineDecisionBuffer destination) {
    destination.analyze(state, observer, legalActions, state.publicState());
  }

  /**
   * 自家の合法なチー・ポンを仮の手へ適用し、鳴いた副露を返す（{@link ActionEffect#projectCallInto} へ借用局面を渡す）。局面は読むだけで
   * 変えず、書くのは呼ぶ側の {@code afterCall}（自家の手と公開の副露の写し）だけ。鳴いた後に打てる牌は
   * {@link com.epsilon.engine.PostCallDiscards} で求める。
   *
   * @param call 自家の合法なチーかポン（直前の出来事は他家の打牌）
   * @param afterCall 書く先（鳴いた後の自家の手）
   * @return 鳴いた副露
   */
  public Meld projectCall(Action call, ActionEffect.ProjectedHand afterCall) {
    return ActionEffect.projectCallInto(state, observer, call, afterCall);
  }

  /**
   * 自家の合法なロン・ツモの公開の射影（裏を除く点・支払いの下限・和了後の順位）を書く（{@link WinSettlementProjection#load} へ借用局面を
   * 渡す）。射影からも完全状態は取り出せない。
   *
   * @param win 自家の合法なロンかツモ
   * @param scoreEvaluator 呼ぶ側の作業域
   * @param into 書く先
   * @return {@code into}
   */
  public WinSettlementProjection projectWin(
      Action win, HandScoreEvaluator scoreEvaluator, WinSettlementProjection into) {
    return into.load(state, observer, win, scoreEvaluator);
  }

  /** 可変なエンジンイベント自体を外へ渡さず、参照が所有する小さいイベントを再利用する。 */
  public TurnEvent turnEvent() {
    return switch (state.turnEvent()) {
      case TurnEvent.Draw value ->
          draw.set(
              value.player(),
              value.player() == observer ? value.physicalTileId() : -1,
              value.drawSource(),
              value.doraRevealPending());
      case TurnEvent.Discard value ->
          discard.set(value.player(), value.tileType(), value.isAkaTile());
      case TurnEvent.KanAttempt value ->
          kan.set(
              value.player(),
              value.tileType(),
              value.isAkaTile(),
              value.kanKind(),
              value.doraRevealPending());
      case TurnEvent.None ignored -> TurnEvent.None.INSTANCE;
    };
  }

  private void requireObserver(int seat) {
    if (seat != observer)
      throw new IllegalArgumentException("Opponent concealed information is unavailable");
  }

  /** 公開面子・枚数は全席、牌種は自家だけを読む。内部のHand型へキャストすることはできない。 */
  public final class VisibleHand implements HandView {
    private final int seat;

    private VisibleHand(int seat) {
      this.seat = seat;
    }

    private Hand own() {
      requireObserver(seat);
      return state.hand(seat);
    }

    @Override
    public void copyShapeInto(HandShapeState target) {
      own().copyShapeInto(target);
    }

    @Override
    public void copyTileCountsInto(HandShapeState.TileCountBuffer target) {
      own().copyTileCountsInto(target);
    }

    @Override
    public int count(int tile) {
      return own().count(tile);
    }

    @Override
    public int concealedTileCount() {
      return state.hand(seat).concealedTileCount();
    }

    @Override
    public int meldCount() {
      return state.hand(seat).meldCount();
    }

    @Override
    public long concealedTileTypeMask() {
      return own().concealedTileTypeMask();
    }

    @Override
    public long concealedPairTileTypeMask() {
      return own().concealedPairTileTypeMask();
    }

    @Override
    public boolean hasAkaTile(int tile) {
      return own().hasAkaTile(tile);
    }

    @Override
    public int concealedAkaMask() {
      return own().concealedAkaMask();
    }

    @Override
    public int ownedAkaMask() {
      return own().ownedAkaMask();
    }

    @Override
    public Meld meld(int index) {
      return state.hand(seat).meld(index);
    }

    @Override
    public int meldCallAfterRiverIndex(int index) {
      return state.hand(seat).meldCallAfterRiverIndex(index);
    }

    @Override
    public int meldKanAfterRiverIndex(int index) {
      return state.hand(seat).meldKanAfterRiverIndex(index);
    }

    public int meldCreationEvent(int index) {
      return state.hand(seat).meldCreationEvent(index);
    }

    public int meldKakanEvent(int index) {
      return state.hand(seat).meldKakanEvent(index);
    }

    @Override
    public boolean isMenzen() {
      return state.hand(seat).isMenzen();
    }

    @Override
    public int kanCount() {
      return state.hand(seat).kanCount();
    }

    @Override
    public long canonicalMeldSignature() {
      return state.hand(seat).canonicalMeldSignature();
    }
  }

  /** 裏ドラを含まない表示済みドラ情報。 */
  public final class VisibleDora {
    private VisibleDora() {}

    public int indicatorCount() {
      return state.doraState().indicatorCount();
    }

    public int indicatorTileType(int index) {
      return state.doraState().indicatorTileType(index);
    }

    public int doraTileType(int index) {
      return state.doraState().doraTileType(index);
    }

    public int doraMultiplicity(int tile) {
      return state.doraState().doraMultiplicity(tile);
    }

    public int indicatorMultiplicity(int tile) {
      return state.doraState().indicatorMultiplicity(tile);
    }

    public int visibleIndicatorAkaMask() {
      return state.doraState().visibleIndicatorAkaMask();
    }

    public long doraTileTypesPacked() {
      return state.doraState().doraTileTypesPacked();
    }
  }
}
