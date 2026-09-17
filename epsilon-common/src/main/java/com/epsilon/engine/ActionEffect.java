package com.epsilon.engine;

import com.epsilon.calculate.shape.HandShapeState;
import com.epsilon.core.Action;
import com.epsilon.core.AkaTileMask;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import java.util.List;

/**
 * 合法手を自家手牌へ適用する処理と、局面を変更せずに適用後の手牌を調べる処理を共有する。
 *
 * <p>候補の比較では{@link Hand}や34要素の配列を複製せず、牌の枚数をビット列に詰めた値と手牌形の符号を使う。未知の嶺上牌は補わず、次に必要な処理を{@link
 * NextStep}で表す。
 */
public final class ActionEffect {

  static final class ProjectionResult {
    private NextStep nextStep;
    private Meld.AkaSource meldAkaSource;

    NextStep nextStep() {
      return nextStep;
    }

    Meld.AkaSource meldAkaSource() {
      return meldAkaSource;
    }

    private void set(NextStep nextStep, Meld.AkaSource meldAkaSource) {
      this.nextStep = nextStep;
      this.meldAkaSource = meldAkaSource;
    }
  }

  /** 行動適用直後に必要な次のエンジン処理。 */
  public enum NextStep {
    /** 現在の判断が完了する。 */
    DECISION_COMPLETE,
    /** チー・ポン成立後に同じ手番で打牌を選ぶ。 */
    IMMEDIATE_DISCARD,
    /** 槓成立後に未知の嶺上牌を引く。 */
    RINSHAN_DRAW,
    /** 和了または途中流局で局が終了する。 */
    ROUND_COMPLETE
  }

  private ActionEffect() {}

  /** 次の処理と鳴き後の合法打牌を準備する。手牌の参照が必要な副露・槓では、適用後の手牌も作る。 */
  static void describeInto(
      GameState state,
      int player,
      Action action,
      ProjectedHand scratch,
      ProjectionResult result,
      List<Action> immediateDiscards) {
    switch (action.type()) {
      case DAHAI, RIICHI_DAHAI, PASS -> {
        immediateDiscards.clear();
        result.set(NextStep.DECISION_COMPLETE, Meld.AkaSource.NONE);
      }
      case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI -> {
        immediateDiscards.clear();
        result.set(NextStep.ROUND_COMPLETE, Meld.AkaSource.NONE);
      }
      case CHI, PON, DAIMINKAN, ANKAN, KAKAN ->
          projectInto(state, player, action, scratch, result, immediateDiscards);
    }
  }

  static void projectInto(
      GameState state,
      int player,
      Action action,
      ProjectedHand resultingHand,
      ProjectionResult result,
      List<Action> immediateDiscards) {
    resultingHand.copyFrom(state.hand(player));
    resultingHand.setOccurrenceAfterRiverIndex(state.river(player).size());
    immediateDiscards.clear();
    switch (action.type()) {
      case DAHAI, RIICHI_DAHAI -> {
        removeDiscardedTile(resultingHand, action.tileType(), action.usesAkaTileFromHand());
        result.set(NextStep.DECISION_COMPLETE, Meld.AkaSource.NONE);
      }
      case CHI ->
          projectChiInto(state, action, resultingHand, resultingHand, result, immediateDiscards);
      case PON ->
          projectPonInto(
              state, player, action, resultingHand, resultingHand, result, immediateDiscards);
      case DAIMINKAN -> {
        Meld meld = createDaiminkan(resultingHand, state, player, currentDiscard(state));
        result.set(NextStep.RINSHAN_DRAW, meld.akaSource());
      }
      case ANKAN -> {
        Meld meld = createAnkan(resultingHand, action.tileType());
        result.set(NextStep.RINSHAN_DRAW, meld.akaSource());
      }
      case KAKAN -> {
        Meld meld = createKakan(resultingHand, action.tileType());
        result.set(NextStep.RINSHAN_DRAW, meld.akaSource());
      }
      case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI ->
          result.set(NextStep.ROUND_COMPLETE, Meld.AkaSource.NONE);
      case PASS -> result.set(NextStep.DECISION_COMPLETE, Meld.AkaSource.NONE);
    }
  }

  static void projectImmediateDiscardInto(
      HandView handAfterCall, Action discardAction, ProjectedHand resultingHand) {
    resultingHand.copyFrom(handAfterCall);
    resultingHand.setOccurrenceAfterRiverIndex(-1);
    removeDiscardedTile(
        resultingHand, discardAction.tileType(), discardAction.usesAkaTileFromHand());
  }

  static boolean removeDiscardedTile(Hand hand, int tileType, boolean usesAkaTileFromHand) {
    if (hand.hasAkaTile(tileType) && (usesAkaTileFromHand || !hand.hasNonAkaTile(tileType))) {
      hand.removeAka(tileType);
      return true;
    }
    hand.removeNonAka(tileType);
    return false;
  }

  static Meld applyAnkan(GameState state, int player, int tileType) {
    return createAnkan(new GameHandMutationTarget().bind(state, player), tileType);
  }

  static Meld applyAnkan(
      GameState state, int player, int tileType, GameHandMutationTarget mutationTarget) {
    return createAnkan(mutationTarget.bind(state, player), tileType);
  }

  static Meld applyKakan(GameState state, int player, int tileType) {
    return createKakan(new GameHandMutationTarget().bind(state, player), tileType);
  }

  static Meld applyKakan(
      GameState state, int player, int tileType, GameHandMutationTarget mutationTarget) {
    return createKakan(mutationTarget.bind(state, player), tileType);
  }

  static Meld applyDaiminkan(GameState state, int player, TurnEvent.Discard discard) {
    return createDaiminkan(
        new GameHandMutationTarget().bind(state, player), state, player, discard);
  }

  static Meld applyDaiminkan(
      GameState state,
      int player,
      TurnEvent.Discard discard,
      GameHandMutationTarget mutationTarget) {
    return createDaiminkan(mutationTarget.bind(state, player), state, player, discard);
  }

  static Meld applyPon(
      GameState state, int player, TurnEvent.Discard discard, boolean consumeAkaFromHand) {
    return createPon(
        new GameHandMutationTarget().bind(state, player),
        state,
        player,
        discard,
        consumeAkaFromHand);
  }

  static Meld applyPon(
      GameState state,
      int player,
      TurnEvent.Discard discard,
      boolean consumeAkaFromHand,
      GameHandMutationTarget mutationTarget) {
    return createPon(
        mutationTarget.bind(state, player), state, player, discard, consumeAkaFromHand);
  }

  static Meld applyChi(GameState state, int player, TurnEvent.Discard discard, Action chiAction) {
    return createChi(new GameHandMutationTarget().bind(state, player), discard, chiAction);
  }

  static Meld applyChi(
      GameState state,
      int player,
      TurnEvent.Discard discard,
      Action chiAction,
      GameHandMutationTarget mutationTarget) {
    return createChi(mutationTarget.bind(state, player), discard, chiAction);
  }

  private static void projectChiInto(
      GameState state,
      Action action,
      ProjectedHand resultingHand,
      MutationTarget target,
      ProjectionResult result,
      List<Action> immediateDiscards) {
    TurnEvent.Discard discard = currentDiscard(state);
    Meld meld = createChi(target, discard, action);
    PostCallDahaiRestriction restriction =
        PostCallDahaiRestriction.afterChi(action.chiSequenceBaseTileType(), discard.tileType());
    ActionGenerator.generateCallDahaiActionsInto(immediateDiscards, resultingHand, restriction);
    result.set(NextStep.IMMEDIATE_DISCARD, meld.akaSource());
  }

  private static void projectPonInto(
      GameState state,
      int player,
      Action action,
      ProjectedHand resultingHand,
      MutationTarget target,
      ProjectionResult result,
      List<Action> immediateDiscards) {
    TurnEvent.Discard discard = currentDiscard(state);
    Meld meld = createPon(target, state, player, discard, action.usesAkaTileFromHand());
    ActionGenerator.generateCallDahaiActionsInto(
        immediateDiscards, resultingHand, PostCallDahaiRestriction.afterPon(discard.tileType()));
    result.set(NextStep.IMMEDIATE_DISCARD, meld.akaSource());
  }

  private static boolean removeDiscardedTile(
      MutationTarget hand, int tileType, boolean usesAkaTileFromHand) {
    if (hand.hasAkaTile(tileType) && (usesAkaTileFromHand || !hand.hasNonAkaTile(tileType))) {
      hand.removeAka(tileType);
      return true;
    }
    hand.removeNonAka(tileType);
    return false;
  }

  private static Meld createAnkan(MutationTarget hand, int tileType) {
    boolean consumedAka = removeCopies(hand, tileType, 4, true);
    Meld meld = Meld.ankan(tileType, consumedAka);
    hand.addMeld(meld);
    return meld;
  }

  private static Meld createKakan(MutationTarget hand, int tileType) {
    boolean addedTileIsAka = false;
    if (hand.hasAkaTile(tileType)) {
      hand.removeAka(tileType);
      addedTileIsAka = true;
    } else {
      hand.removeNonAka(tileType);
    }
    Meld pon = requirePon(hand, tileType);
    Meld kakan = Meld.kakan(pon, addedTileIsAka);
    hand.replacePonWithKakan(pon, kakan);
    return kakan;
  }

  private static Meld createDaiminkan(
      MutationTarget hand, GameState state, int player, TurnEvent.Discard discard) {
    int tileType = discard.tileType();
    boolean consumedAka = removeCopies(hand, tileType, 3, true);
    Meld meld =
        Meld.daiminkan(
            tileType,
            relativeSource(state, player, discard.player()),
            Meld.AkaSource.forCall(discard.isAkaTile(), consumedAka));
    hand.addMeld(meld);
    return meld;
  }

  private static Meld createPon(
      MutationTarget hand,
      GameState state,
      int player,
      TurnEvent.Discard discard,
      boolean consumeAkaFromHand) {
    int tileType = discard.tileType();
    boolean consumedAka;
    if (hand.hasAkaTile(tileType)
        && (consumeAkaFromHand || !hand.hasMultipleNonAkaTiles(tileType))) {
      hand.removeAka(tileType);
      hand.removeNonAka(tileType);
      consumedAka = true;
    } else {
      hand.removeNonAka(tileType);
      hand.removeNonAka(tileType);
      consumedAka = false;
    }
    Meld meld =
        Meld.pon(
            tileType,
            relativeSource(state, player, discard.player()),
            Meld.AkaSource.forCall(discard.isAkaTile(), consumedAka));
    hand.addMeld(meld);
    return meld;
  }

  private static Meld createChi(MutationTarget hand, TurnEvent.Discard discard, Action chiAction) {
    boolean consumedAka = false;
    int sequenceBaseTileType = chiAction.chiSequenceBaseTileType();
    for (int tileType = sequenceBaseTileType; tileType <= sequenceBaseTileType + 2; tileType++) {
      if (tileType == discard.tileType()) {
        continue;
      }
      if (hand.hasAkaTile(tileType)
          && (chiAction.usesAkaTileFromHand() || !hand.hasNonAkaTile(tileType))) {
        hand.removeAka(tileType);
        consumedAka = true;
      } else {
        hand.removeNonAka(tileType);
      }
    }
    Meld meld =
        Meld.chi(
            sequenceBaseTileType,
            discard.tileType(),
            Meld.AkaSource.forCall(discard.isAkaTile(), consumedAka));
    hand.addMeld(meld);
    return meld;
  }

  private static boolean removeCopies(
      MutationTarget hand, int tileType, int copies, boolean consumeAkaFirst) {
    boolean removedAka = false;
    if (consumeAkaFirst && hand.hasAkaTile(tileType)) {
      hand.removeAka(tileType);
      copies--;
      removedAka = true;
    }
    for (int removed = 0; removed < copies; removed++) {
      hand.removeNonAka(tileType);
    }
    return removedAka;
  }

  private static Meld requirePon(MutationTarget hand, int tileType) {
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      if (meld.type() == Meld.Type.PON && meld.baseTileType() == tileType) {
        return meld;
      }
    }
    throw new IllegalStateException("PON meld not found for " + Tile.name(tileType));
  }

  private static Meld.RelativeSource relativeSource(GameState state, int player, int sourcePlayer) {
    return Meld.RelativeSource.fromPlayerOffset(state.getRelativePosition(player, sourcePlayer));
  }

  private static TurnEvent.Discard currentDiscard(GameState state) {
    if (state.getTurnEvent() instanceof TurnEvent.Discard discard) {
      return discard;
    }
    throw new IllegalStateException("call action requires a discard event");
  }

  private interface MutationTarget {
    boolean hasAkaTile(int tileType);

    boolean hasNonAkaTile(int tileType);

    boolean hasMultipleNonAkaTiles(int tileType);

    void removeAka(int tileType);

    void removeNonAka(int tileType);

    int meldCount();

    Meld meld(int meldIndex);

    void addMeld(Meld meld);

    void replacePonWithKakan(Meld pon, Meld kakan);
  }

  /** MahjongTransitionが所有する手牌更新用の補助オブジェクト。副露・槓を適用するたびに、更新先の局面と席を設定する。 */
  static final class GameHandMutationTarget implements MutationTarget {
    private GameState state;
    private int player;
    private Hand hand;

    GameHandMutationTarget() {}

    private GameHandMutationTarget bind(GameState state, int player) {
      this.state = state;
      this.player = player;
      hand = state.hand(player);
      return this;
    }

    @Override
    public boolean hasAkaTile(int tileType) {
      return hand.hasAkaTile(tileType);
    }

    @Override
    public boolean hasNonAkaTile(int tileType) {
      return hand.hasNonAkaTile(tileType);
    }

    @Override
    public boolean hasMultipleNonAkaTiles(int tileType) {
      return hand.hasMultipleNonAkaTiles(tileType);
    }

    @Override
    public void removeAka(int tileType) {
      hand.removeAka(tileType);
    }

    @Override
    public void removeNonAka(int tileType) {
      hand.removeNonAka(tileType);
    }

    @Override
    public int meldCount() {
      return hand.meldCount();
    }

    @Override
    public Meld meld(int meldIndex) {
      return hand.meld(meldIndex);
    }

    @Override
    public void addMeld(Meld meld) {
      state.addMeld(player, meld, state.river(player).size());
    }

    @Override
    public void replacePonWithKakan(Meld pon, Meld kakan) {
      state.replacePonWithKakan(player, pon, kakan, state.river(player).size());
    }
  }

  /** 34要素配列を持たない行動候補専用手牌。生成後は外部から変更できない。 */
  static final class ProjectedHand implements HandView, MutationTarget {
    private static final int MELD_ID_BITS = 14;
    private static final int TIMELINE_BITS = 8;
    private static final int TIMELINE_MASK = (1 << TIMELINE_BITS) - 1;
    private final HandShapeState shape = new HandShapeState();

    @Override
    public void copyShapeInto(HandShapeState destination) {
      shape.copyShapeInto(destination);
    }

    @Override
    public void copyTileCountsInto(HandShapeState.TileCountBuffer destination) {
      shape.copyTileCountsInto(destination);
    }

    private int concealedAkaMask;
    private int ownedAkaMask;
    private long meldSignature;
    private int callTimeline;
    private int kanTimeline;
    private boolean menzen;
    private int kanCount;
    private int occurrenceAfterRiverIndex;

    ProjectedHand() {}

    void copyFrom(HandView source) {
      shape.load(source);
      concealedAkaMask = source.concealedAkaMask();
      ownedAkaMask = source.ownedAkaMask();
      meldSignature = source.canonicalMeldSignature();
      menzen = source.isMenzen();
      kanCount = source.kanCount();
      callTimeline = 0;
      kanTimeline = 0;
      for (int meldIndex = 0; meldIndex < meldCount(); meldIndex++) {
        callTimeline =
            setTimeline(callTimeline, meldIndex, source.meldCallAfterRiverIndex(meldIndex));
        kanTimeline = setTimeline(kanTimeline, meldIndex, source.meldKanAfterRiverIndex(meldIndex));
      }
    }

    private void setOccurrenceAfterRiverIndex(int occurrenceAfterRiverIndex) {
      this.occurrenceAfterRiverIndex = occurrenceAfterRiverIndex;
    }

    @Override
    public int count(int tileType) {
      return shape.count(tileType);
    }

    @Override
    public int concealedTileCount() {
      return shape.concealedTileCount();
    }

    @Override
    public boolean hasAkaTile(int tileType) {
      return AkaTileMask.containsTile(concealedAkaMask, tileType);
    }

    @Override
    public boolean hasNonAkaTile(int tileType) {
      return HandView.super.hasNonAkaTile(tileType);
    }

    @Override
    public boolean hasMultipleNonAkaTiles(int tileType) {
      return HandView.super.hasMultipleNonAkaTiles(tileType);
    }

    @Override
    public int concealedAkaMask() {
      return concealedAkaMask;
    }

    @Override
    public int ownedAkaMask() {
      return ownedAkaMask;
    }

    @Override
    public long concealedTileTypeMask() {
      return shape.concealedTileTypeMask();
    }

    @Override
    public int distinctConcealedTileTypeCount() {
      return shape.distinctConcealedTileTypeCount();
    }

    @Override
    public int concealedPairTileTypeCount() {
      return shape.concealedPairTileTypeCount();
    }

    @Override
    public long concealedPairTileTypeMask() {
      return shape.concealedPairTileTypeMask();
    }

    @Override
    public int kokushiTileTypeCount() {
      return shape.kokushiTileTypeCount();
    }

    @Override
    public int kokushiPairTileTypeCount() {
      return shape.kokushiPairTileTypeCount();
    }

    @Override
    public int meldCount() {
      return shape.meldCount();
    }

    @Override
    public Meld meld(int meldIndex) {
      int canonicalId =
          (int) ((meldSignature >>> (meldIndex * MELD_ID_BITS)) & ((1L << MELD_ID_BITS) - 1L));
      return Meld.fromCanonicalId(canonicalId);
    }

    @Override
    public int meldCallAfterRiverIndex(int meldIndex) {
      return timelineValue(callTimeline, meldIndex);
    }

    @Override
    public int meldKanAfterRiverIndex(int meldIndex) {
      return timelineValue(kanTimeline, meldIndex);
    }

    @Override
    public boolean isMenzen() {
      return menzen;
    }

    @Override
    public int kanCount() {
      return kanCount;
    }

    @Override
    public long canonicalMeldSignature() {
      return meldSignature;
    }

    @Override
    public void removeAka(int tileType) {
      decrementTile(tileType);
      concealedAkaMask = AkaTileMask.excludeTile(concealedAkaMask, tileType);
      ownedAkaMask = AkaTileMask.excludeTile(ownedAkaMask, tileType);
    }

    @Override
    public void removeNonAka(int tileType) {
      decrementTile(tileType);
    }

    private void decrementTile(int tileType) {
      shape.remove(tileType);
    }

    @Override
    public void addMeld(Meld meld) {
      meldSignature |= (long) meld.canonicalId() << (meldCount() * MELD_ID_BITS);
      boolean call = meld.type() == Meld.Type.CHI || meld.type() == Meld.Type.PON;
      callTimeline = setTimeline(callTimeline, meldCount(), call ? occurrenceAfterRiverIndex : -1);
      kanTimeline = setTimeline(kanTimeline, meldCount(), call ? -1 : occurrenceAfterRiverIndex);
      shape.setMeldCount(meldCount() + 1);
      menzen &= meld.preservesMenzen();
      if (meld.isKan()) {
        kanCount++;
      }
      if (meld.containsAkaTile()) {
        int akaTileType = meld.akaTileType();
        ownedAkaMask = AkaTileMask.includeTile(ownedAkaMask, akaTileType);
      }
    }

    @Override
    public void replacePonWithKakan(Meld pon, Meld kakan) {
      int meldIndex = indexOfMeldIdentity(pon);
      long segmentMask = ((1L << MELD_ID_BITS) - 1L) << (meldIndex * MELD_ID_BITS);
      meldSignature =
          (meldSignature & ~segmentMask) | (long) kakan.canonicalId() << (meldIndex * MELD_ID_BITS);
      kanTimeline = setTimeline(kanTimeline, meldIndex, occurrenceAfterRiverIndex);
      kanCount++;
      if (kakan.addedTileIsAka()) {
        ownedAkaMask = AkaTileMask.includeTile(ownedAkaMask, kakan.baseTileType());
      }
    }

    private int indexOfMeldIdentity(Meld expected) {
      for (int meldIndex = 0; meldIndex < meldCount(); meldIndex++) {
        if (meld(meldIndex) == expected) {
          return meldIndex;
        }
      }
      return -1;
    }

    private static int setTimeline(int packed, int meldIndex, int riverIndex) {
      int shift = meldIndex * TIMELINE_BITS;
      return (packed & ~(TIMELINE_MASK << shift)) | (riverIndex + 1) << shift;
    }

    private static int timelineValue(int packed, int meldIndex) {
      return ((packed >>> (meldIndex * TIMELINE_BITS)) & TIMELINE_MASK) - 1;
    }
  }
}
