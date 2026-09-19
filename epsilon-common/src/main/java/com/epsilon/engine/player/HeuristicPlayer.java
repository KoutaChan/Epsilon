package com.epsilon.engine.player;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.shape.HandShapeAnalyzer;
import com.epsilon.calculate.shape.HandShapeCursor;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.HandView;
import com.epsilon.core.Meld;
import com.epsilon.core.River;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.core.ScoreRanking;
import com.epsilon.core.Tile;
import com.epsilon.engine.ActionEffect;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.HandAnalysisBuffer;
import com.epsilon.engine.Player;
import com.epsilon.engine.ScorePayments;
import com.epsilon.engine.YakuRouteAnalyzer;
import java.util.List;

/**
 * 公開情報だけで牌効率・打点・鳴き・押し引きを評価する決定的プレイヤー。
 *
 * <p>現在の局面を変更せずに各候補を適用した手牌を作り、打牌時点の未確認牌の枚数を使って受け入れ枚数を数える。テンパイでは、役の成立を確認したロン・ツモの待ち牌と、裏ドラを含まない点数を使う。
 * 守備では、現物・スジ・壁を警戒する他家ごとに評価する。一人の現物を別のリーチ者にも安全とはみなさない。
 */
public final class HeuristicPlayer implements Player {

  private final DecisionContext decisionContext = new DecisionContext();

  private static final int DANGER_GENBUTSU = 0;
  private static final int DANGER_FOURTH_HONOR = 5;
  private static final int DANGER_SEEN_HONOR = 20;
  private static final int DANGER_SUJI_OR_NO_CHANCE = 30;
  private static final int DANGER_LIVE_HONOR = 40;
  private static final int DANGER_EARLY_ONE_CHANCE = 40;
  private static final int DANGER_NON_SUJI_TERMINAL = 50;
  private static final int DANGER_LATE_ONE_CHANCE = 50;
  private static final int DANGER_NON_SUJI_TWO_EIGHT = 70;
  private static final int DANGER_NON_SUJI_THREE_SEVEN = 80;
  private static final int DANGER_NON_SUJI_MIDDLE = 90;
  private static final int ONE_CHANCE_RELIABLE_MIN_WALL_TILES = 40;

  private static final int GOOD_WAIT_MIN_LIVE_TILES = 5;
  private static final int GOOD_WAIT_MIN_TILE_TYPES = 2;
  private static final int VALUABLE_TENPAI_MIN_RON_POINTS = 5_200;
  private static final int DAMATEN_HIGH_VALUE_MIN_RON_POINTS = 7_700;
  private static final int ONE_SHANTEN_MIN_UKEIRE = 8;
  private static final int ONE_SHANTEN_MIN_UKEIRE_TYPES = 3;
  private static final int ONE_SHANTEN_MIN_SELF_DRAWS = 5;
  private static final int KYUSHU_CONTINUE_MAX_SHANTEN = 2;
  private static final int KAN_UKEIRE_RETENTION_PERCENT = 80;

  private static final long UKEIRE_WEIGHT = 1_000L;
  private static final long SECOND_ORDER_WEIGHT = 160L;
  private static final long UKEIRE_TYPE_WEIGHT = 80L;
  private static final long VALUE_HAN_WEIGHT = 1_600L;
  private static final long CONFIRMED_YAKU_WEIGHT = 400L;
  private static final long LIVE_WAIT_WEIGHT = 10_000L;
  private static final long RON_WAIT_WEIGHT = 2_500L;
  private static final long RON_POINT_WEIGHT_DIVISOR = 4L;
  private static final long PRESS_DANGER_WEIGHT = 300L;

  /** 判断作業領域を所有し、選択間で再利用するプレイヤーを作る。 */
  public HeuristicPlayer() {}

  @Override
  public Action selectAction(GameState state, int playerIndex, List<Action> legalActions) {
    if (legalActions.isEmpty()) {
      throw new IllegalArgumentException("legalActions must not be empty");
    }

    Action win = findWin(legalActions);
    if (win != null) {
      return win;
    }

    DecisionContext context = decisionContext.bind(state, playerIndex, legalActions);
    Action pass = find(legalActions, Action.Type.PASS);
    if (pass != null) {
      return chooseResponse(context, legalActions, pass);
    }
    return chooseTurn(context, legalActions);
  }

  private static Action chooseResponse(
      DecisionContext context, List<Action> legalActions, Action pass) {
    if (context.threatCount != 0) {
      return pass;
    }

    HandQuality current = context.analyzeCurrent(RiichiState.NONE);
    CallChoice best = null;
    for (Action action : legalActions) {
      if (!action.type().isCall()) {
        continue;
      }
      CallChoice choice = evaluateCall(context, action);
      if (choice != null
          && acceptCall(context, current, choice)
          && (best == null || betterCall(choice, best))) {
        best = choice;
      }
    }
    return best == null ? pass : best.action;
  }

  private static CallChoice evaluateCall(DecisionContext context, Action action) {
    int actionIndex = context.actionIndex(action);
    if (context.decision.continuation(actionIndex) == ActionEffect.NextStep.IMMEDIATE_DISCARD) {
      DiscardChoice best = null;
      for (int transition = 0;
          transition < context.decision.transitionCount(actionIndex);
          transition++) {
        Action discard = context.decision.discardAction(actionIndex, transition);
        HandQuality quality = context.analyzeTransition(actionIndex, transition, RiichiState.NONE);
        int danger = context.danger(discard.tileType());
        DiscardChoice candidate =
            context.discardChoice(discard, quality, danger, offensiveScore(quality, false));
        if (best == null || betterDiscard(candidate, best, AttackMode.NORMAL)) {
          best = candidate;
        }
      }
      return best == null ? null : context.callChoice(action, best.quality, best.score);
    }
    if (context.decision.continuation(actionIndex) != ActionEffect.NextStep.RINSHAN_DRAW) {
      return null;
    }
    HandQuality quality = context.analyzeTransition(actionIndex, 0, RiichiState.NONE);
    return context.callChoice(action, quality, offensiveScore(quality, false));
  }

  private static boolean acceptCall(
      DecisionContext context, HandQuality current, CallChoice choice) {
    HandQuality after = choice.quality;
    if (!hasReliableOpenYaku(after)) {
      return false;
    }

    if (choice.action.type() == Action.Type.DAIMINKAN) {
      return after.minimumShanten == 0
          && after.liveWaitTiles >= GOOD_WAIT_MIN_LIVE_TILES
          && after.minimumRonPoints >= 2_000
          && after.minimumRonPoints <= 5_200;
    }

    boolean yakuhaiPon =
        choice.action.type() == Action.Type.PON && context.isYakuhai(choice.action.tileType());
    if (after.minimumShanten > current.minimumShanten) {
      return false;
    }
    if (after.minimumShanten == current.minimumShanten) {
      return yakuhaiPon
          && after.minimumShanten <= 1
          && choice.score > offensiveScore(current, false);
    }
    return after.minimumShanten <= 2 || after.valueHan() >= 2 || yakuhaiPon;
  }

  private static boolean betterCall(CallChoice candidate, CallChoice incumbent) {
    if (candidate.quality.minimumShanten != incumbent.quality.minimumShanten) {
      return candidate.quality.minimumShanten < incumbent.quality.minimumShanten;
    }
    if (candidate.score != incumbent.score) {
      return candidate.score > incumbent.score;
    }
    int candidateAkaHan = Integer.bitCount(candidate.quality.ownedAkaMask);
    int incumbentAkaHan = Integer.bitCount(incumbent.quality.ownedAkaMask);
    if (candidateAkaHan != incumbentAkaHan) {
      return candidateAkaHan > incumbentAkaHan;
    }
    return callTieBreak(candidate.action, incumbent.action) < 0;
  }

  private static Action chooseTurn(DecisionContext context, List<Action> legalActions) {
    Action abort = find(legalActions, Action.Type.KYUSHU_KYUHAI);
    if (abort != null && shouldAbort(context)) {
      return abort;
    }

    DiscardChoice offensiveDama =
        chooseBestDiscard(context, legalActions, Action.Type.DAHAI, AttackMode.NORMAL);
    DiscardChoice bestRiichi =
        chooseBestDiscard(context, legalActions, Action.Type.RIICHI_DAHAI, AttackMode.NORMAL);

    if (offensiveDama == null && bestRiichi == null) {
      return legalActions.getFirst();
    }

    AttackMode mode = chooseAttackMode(context, offensiveDama != null ? offensiveDama : bestRiichi);
    DiscardChoice selectedDama = offensiveDama;
    if (offensiveDama != null && mode != AttackMode.NORMAL) {
      selectedDama = chooseBestDiscard(context, legalActions, Action.Type.DAHAI, mode);
    }

    DiscardChoice selected = selectedDama != null ? selectedDama : bestRiichi;
    if (bestRiichi != null && shouldDeclareRiichi(context, bestRiichi, offensiveDama, mode)) {
      selected = bestRiichi;
    }

    Action kan = chooseKan(context, legalActions, selected, mode);
    return kan != null ? kan : selected.action;
  }

  private static AttackMode chooseAttackMode(DecisionContext context, DiscardChoice attack) {
    if (context.threatCount == 0) {
      return AttackMode.NORMAL;
    }
    if (context.lateLastPlace) {
      return AttackMode.PRESS;
    }
    if (context.lateFirstPlace && attack.quality.minimumShanten > 0) {
      return AttackMode.FOLD;
    }

    HandQuality quality = attack.quality;
    boolean valuable =
        quality.valueHan() >= 2 || quality.minimumRonPoints >= VALUABLE_TENPAI_MIN_RON_POINTS;
    int positiveConditions = 0;
    positiveConditions += quality.goodWait() ? 1 : 0;
    positiveConditions += valuable ? 1 : 0;
    positiveConditions += context.dealer ? 1 : 0;
    positiveConditions += context.selfDrawsLeft >= ONE_SHANTEN_MIN_SELF_DRAWS ? 1 : 0;

    if (quality.minimumShanten == 0) {
      int required = context.riichiThreatCount >= 2 ? 3 : 2;
      return positiveConditions >= required ? AttackMode.PRESS : AttackMode.FOLD;
    }
    if (quality.minimumShanten == 1) {
      boolean wide =
          quality.ukeireCount >= ONE_SHANTEN_MIN_UKEIRE
              && quality.ukeireTileTypes >= ONE_SHANTEN_MIN_UKEIRE_TYPES;
      boolean earlyEnough = context.selfDrawsLeft >= ONE_SHANTEN_MIN_SELF_DRAWS;
      return context.riichiThreatCount <= 1 && wide && earlyEnough && (valuable || context.dealer)
          ? AttackMode.PRESS
          : AttackMode.FOLD;
    }
    return AttackMode.FOLD;
  }

  private static boolean shouldDeclareRiichi(
      DecisionContext context, DiscardChoice riichi, DiscardChoice dama, AttackMode mode) {
    if (context.state.isRiichi(context.player) || riichi.quality.liveWaitTiles == 0) {
      return false;
    }
    if (mode == AttackMode.FOLD) {
      return false;
    }
    if (mode == AttackMode.PRESS && riichi.danger > 50 && !context.lateLastPlace) {
      return false;
    }
    if (dama == null || dama.quality.liveRonTiles == 0) {
      return true;
    }
    if (context.lateFirstPlace && dama.quality.liveRonTiles > 0) {
      return false;
    }
    if (dama.quality.minimumRonPoints >= DAMATEN_HIGH_VALUE_MIN_RON_POINTS) {
      return false;
    }
    return riichi.quality.goodWait()
        || riichi.quality.liveWaitTiles >= 3
        || context.dealer
        || context.state.getTurnNumber() >= 8;
  }

  private static Action chooseKan(
      DecisionContext context,
      List<Action> legalActions,
      DiscardChoice selectedDiscard,
      AttackMode mode) {
    if (mode != AttackMode.NORMAL || context.threatCount != 0) {
      return null;
    }
    if (selectedDiscard.quality.sevenPairsShanten <= 1
        && selectedDiscard.quality.specialHandsAvailable) {
      return null;
    }

    Action bestAction = null;
    HandQuality bestQuality = null;
    for (Action action : legalActions) {
      if (!action.type().isTurnKan()) {
        continue;
      }
      HandQuality after =
          context.analyzeTransition(context.actionIndex(action), 0, context.currentRiichiStatus());
      if (after.minimumShanten > 1
          || after.minimumShanten > selectedDiscard.quality.minimumShanten) {
        continue;
      }
      if (after.minimumShanten == selectedDiscard.quality.minimumShanten
          && after.ukeireCount * 100
              < selectedDiscard.quality.ukeireCount * KAN_UKEIRE_RETENTION_PERCENT) {
        continue;
      }
      if (action.type() == Action.Type.KAKAN
          && (after.valueHan() < 2 || after.minimumShanten > 0 || !after.goodWait())) {
        continue;
      }
      if (bestQuality == null || betterQuality(after, bestQuality)) {
        bestAction = action;
        bestQuality = after;
      }
    }
    return bestAction;
  }

  private static boolean shouldAbort(DecisionContext context) {
    return !context.lateLastPlace
        && context.shapeAnalyzer.calculateStandard(context.hand) > KYUSHU_CONTINUE_MAX_SHANTEN
        && context.shapeAnalyzer.calculateChiitoitsu(context.hand) > KYUSHU_CONTINUE_MAX_SHANTEN
        && context.shapeAnalyzer.calculateKokushi(context.hand) > KYUSHU_CONTINUE_MAX_SHANTEN;
  }

  private static DiscardChoice chooseBestDiscard(
      DecisionContext context, List<Action> actions, Action.Type targetType, AttackMode mode) {
    DiscardChoice best = null;
    for (Action action : actions) {
      if (action.type() != targetType) {
        continue;
      }
      RiichiState riichiStatus =
          targetType == Action.Type.RIICHI_DAHAI
              ? context.declarationRiichiStatus()
              : context.currentRiichiStatus();
      HandQuality quality = context.analyzeTransition(context.actionIndex(action), 0, riichiStatus);
      boolean furiten = context.isFuriten(quality.shapeWaitMask, action.tileType());
      int danger = context.danger(action.tileType());
      long attackScore = offensiveScore(quality, furiten);
      long score =
          mode == AttackMode.PRESS ? attackScore - danger * PRESS_DANGER_WEIGHT : attackScore;
      DiscardChoice candidate = context.discardChoice(action, quality, danger, score);
      if (best == null || betterDiscard(candidate, best, mode)) {
        best = candidate;
      }
    }
    return best;
  }

  private static boolean betterDiscard(
      DiscardChoice candidate, DiscardChoice incumbent, AttackMode mode) {
    if (mode == AttackMode.FOLD && candidate.danger != incumbent.danger) {
      return candidate.danger < incumbent.danger;
    }
    if (candidate.quality.minimumShanten != incumbent.quality.minimumShanten) {
      return candidate.quality.minimumShanten < incumbent.quality.minimumShanten;
    }
    if (candidate.score != incumbent.score) {
      return candidate.score > incumbent.score;
    }
    int candidateAkaHan = Integer.bitCount(candidate.quality.ownedAkaMask);
    int incumbentAkaHan = Integer.bitCount(incumbent.quality.ownedAkaMask);
    if (candidateAkaHan != incumbentAkaHan) {
      return candidateAkaHan > incumbentAkaHan;
    }
    if (candidate.danger != incumbent.danger) {
      return candidate.danger < incumbent.danger;
    }
    if (candidate.action.usesAkaTileFromHand() != incumbent.action.usesAkaTileFromHand()) {
      return !candidate.action.usesAkaTileFromHand();
    }
    if (candidate.action.isTsumogiri() != incumbent.action.isTsumogiri()) {
      return candidate.action.isTsumogiri();
    }
    return candidate.action.discardIdentityIndex() < incumbent.action.discardIdentityIndex();
  }

  private static boolean betterQuality(HandQuality candidate, HandQuality incumbent) {
    if (candidate.minimumShanten != incumbent.minimumShanten) {
      return candidate.minimumShanten < incumbent.minimumShanten;
    }
    return offensiveScore(candidate, false) > offensiveScore(incumbent, false);
  }

  private static long offensiveScore(HandQuality quality, boolean furiten) {
    if (quality.minimumShanten == 0) {
      int liveRon = furiten ? 0 : quality.liveRonTiles;
      return quality.liveWaitTiles * LIVE_WAIT_WEIGHT
          + liveRon * RON_WAIT_WEIGHT
          + quality.averageRonPoints * (long) liveRon / RON_POINT_WEIGHT_DIVISOR
          + quality.valueHan() * VALUE_HAN_WEIGHT;
    }
    return quality.ukeireCount * UKEIRE_WEIGHT
        + quality.secondOrderUkeire * SECOND_ORDER_WEIGHT
        + quality.ukeireTileTypes * UKEIRE_TYPE_WEIGHT
        + quality.valueHan() * VALUE_HAN_WEIGHT
        + Long.bitCount(quality.confirmedYakuRouteBits) * CONFIRMED_YAKU_WEIGHT;
  }

  private static boolean hasReliableOpenYaku(HandQuality quality) {
    if (quality.minimumShanten == 0) {
      return quality.liveWaitTiles > 0;
    }
    if (quality.confirmedYakuRouteBits != 0L) {
      return true;
    }
    return quality.allTilesSimple || quality.oneSuitAndHonors;
  }

  private static boolean allTilesAreSimple(HandView hand) {
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      if (hand.count(tile) > 0 && Tile.isTerminalOrHonor(tile)) {
        return false;
      }
    }
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
        if (Tile.isTerminalOrHonor(meld.tileAt(tileIndex))) {
          return false;
        }
      }
    }
    return true;
  }

  private static boolean allTilesUseOneSuitAndHonors(HandView hand) {
    int suitMask = 0;
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      if (hand.count(tile) > 0 && Tile.isNumberTile(tile)) {
        suitMask |= 1 << Tile.suitOf(tile);
      }
    }
    for (int meldIndex = 0; meldIndex < hand.meldCount(); meldIndex++) {
      Meld meld = hand.meld(meldIndex);
      for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
        int tile = meld.tileAt(tileIndex);
        if (Tile.isNumberTile(tile)) {
          suitMask |= 1 << Tile.suitOf(tile);
        }
      }
    }
    return Integer.bitCount(suitMask) <= 1;
  }

  private static Action findWin(List<Action> actions) {
    for (Action action : actions) {
      if (action.type() == Action.Type.TSUMO_AGARI || action.type() == Action.Type.RON_AGARI) {
        return action;
      }
    }
    return null;
  }

  private static Action find(List<Action> actions, Action.Type type) {
    for (Action action : actions) {
      if (action.type() == type) {
        return action;
      }
    }
    return null;
  }

  private static int callTieBreak(Action left, Action right) {
    int type = Integer.compare(left.type().ordinal(), right.type().ordinal());
    if (type != 0) {
      return type;
    }
    int tile = Integer.compare(left.tileType(), right.tileType());
    return tile != 0
        ? tile
        : Boolean.compare(left.usesAkaTileFromHand(), right.usesAkaTileFromHand());
  }

  private enum AttackMode {
    NORMAL,
    PRESS,
    FOLD
  }

  private static final class DiscardChoice {
    private Action action;
    private HandQuality quality;
    private int danger;
    private long score;

    private DiscardChoice bind(Action action, HandQuality quality, int danger, long score) {
      this.action = action;
      this.quality = quality;
      this.danger = danger;
      this.score = score;
      return this;
    }
  }

  private static final class CallChoice {
    private Action action;
    private HandQuality quality;
    private long score;

    private CallChoice bind(Action action, HandQuality quality, long score) {
      this.action = action;
      this.quality = quality;
      this.score = score;
      return this;
    }
  }

  private static final class HandQuality {
    private boolean allTilesSimple;
    private boolean oneSuitAndHonors;
    private int minimumShanten;
    private int sevenPairsShanten;
    private boolean specialHandsAvailable;
    private int ukeireCount;
    private int ukeireTileTypes;
    private int secondOrderUkeire;
    private int doraCount;
    private int ownedAkaMask;
    private long shapeWaitMask;
    private int liveWaitTiles;
    private int liveWaitTileTypes;
    private int liveRonTiles;
    private int minimumRonPoints;
    private int averageRonPoints;
    private long confirmedYakuRouteBits;

    private HandQuality bind(
        HandView hand,
        HandAnalysisBuffer analysis,
        int secondOrderUkeire,
        long confirmedYakuRouteBits) {
      allTilesSimple = allTilesAreSimple(hand);
      oneSuitAndHonors = allTilesUseOneSuitAndHonors(hand);
      minimumShanten = analysis.minimumShanten();
      sevenPairsShanten = analysis.chiitoitsuShanten();
      specialHandsAvailable = analysis.specialHandShantenAvailable();
      ukeireCount = analysis.liveImprovingCopies();
      ukeireTileTypes = analysis.liveImprovingTileTypes();
      this.secondOrderUkeire = secondOrderUkeire;
      doraCount = analysis.doraCount();
      ownedAkaMask = analysis.ownedAkaMask();
      shapeWaitMask = analysis.shapeWaitTileTypeMask();
      this.confirmedYakuRouteBits = confirmedYakuRouteBits;
      return this;
    }

    private int valueHan() {
      return doraCount + Integer.bitCount(ownedAkaMask);
    }

    private boolean goodWait() {
      return liveWaitTiles >= GOOD_WAIT_MIN_LIVE_TILES
          && liveWaitTileTypes >= GOOD_WAIT_MIN_TILE_TYPES;
    }
  }

  private static final class DecisionContext {
    private static final int MAX_ANALYSES = 512;
    private static final int MAX_CHOICES = 256;

    private GameState state;
    private int player;
    private HandView hand;
    private final EngineDecisionBuffer decision = new EngineDecisionBuffer();
    private final HandShapeAnalyzer shapeAnalyzer = new HandShapeAnalyzer();
    private RoundPublicStateIndex publicState;
    private final int[] visibleTileCounts = new int[Tile.NUM_TILE_TYPES];
    private int[] unseenTileCounts;
    private final HandShapeCursor secondOrderShape = new HandShapeCursor();
    private final int[] secondOrderRemaining = new int[Tile.NUM_TILE_TYPES];
    private final boolean[] threatRiichi = new boolean[GameState.NUM_PLAYERS - 1];
    private final int[] threatWeightPercent = new int[GameState.NUM_PLAYERS - 1];
    private final long[] threatSafeTileMask = new long[GameState.NUM_PLAYERS - 1];
    private final long[] threatSujiAnchorMask = new long[GameState.NUM_PLAYERS - 1];
    private final HandQuality[] qualities = new HandQuality[MAX_ANALYSES];
    private final DiscardChoice[] discardChoices = new DiscardChoice[MAX_CHOICES];
    private final CallChoice[] callChoices = new CallChoice[GameState.NUM_PLAYERS * 8];
    private long ownRiverTileMask;
    private int threatCount;
    private int riichiThreatCount;
    private int selfDrawsLeft;
    private boolean dealer;
    private boolean lateFirstPlace;
    private boolean lateLastPlace;
    private int qualityCursor;
    private int discardChoiceCursor;
    private int callChoiceCursor;

    private DecisionContext() {
      for (int index = 0; index < qualities.length; index++) {
        qualities[index] = new HandQuality();
      }
      for (int index = 0; index < discardChoices.length; index++) {
        discardChoices[index] = new DiscardChoice();
      }
      for (int index = 0; index < callChoices.length; index++) {
        callChoices[index] = new CallChoice();
      }
    }

    private DecisionContext bind(GameState state, int player, List<Action> legalActions) {
      this.state = state;
      this.player = player;
      hand = state.hand(player);
      publicState = state.publicState();
      decision.analyze(state, player, legalActions, publicState);
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
        visibleTileCounts[tile] = publicState.visibleTileCount(tile);
      }
      unseenTileCounts = decision.unseenCopiesByTileType();
      ownRiverTileMask = publicState.riverTileTypeMask(player);
      qualityCursor = 0;
      discardChoiceCursor = 0;
      callChoiceCursor = 0;
      collectThreats();
      selfDrawsLeft =
          (state.remainingWallTiles() + GameState.NUM_PLAYERS - 1) / GameState.NUM_PLAYERS;
      dealer = state.getOya() == player;
      int rank = ScoreRanking.rankOf(state, player);
      boolean lateRound = state.getKyokuIndex() >= 6;
      lateFirstPlace = lateRound && rank == 0;
      lateLastPlace = lateRound && rank == GameState.NUM_PLAYERS - 1;
      return this;
    }

    private void collectThreats() {
      threatCount = 0;
      riichiThreatCount = 0;
      int turn = state.getTurnNumber();
      for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
        if (seat == player) {
          continue;
        }
        if (state.isRiichi(seat)) {
          int weight = state.getOya() == seat ? 150 : 100;
          addThreat(
              true,
              weight,
              publicState.riichiGenbutsuTileTypeMask(seat),
              preRiichiDiscardMask(seat));
          riichiThreatCount++;
          continue;
        }

        HandView opponent = state.hand(seat);
        int melds = opponent.meldCount();
        if (melds < 3 && (melds < 2 || turn < 12)) {
          continue;
        }
        int weight = 35 + Math.max(0, melds - 2) * 20 + visibleMeldValue(opponent, seat) * 10;
        if (state.getOya() == seat) {
          weight = weight * 3 / 2;
        }
        addThreat(false, Math.min(weight, 100), publicState.riverTileTypeMask(seat), 0L);
      }
    }

    private void addThreat(boolean riichi, int weight, long safeMask, long sujiMask) {
      threatRiichi[threatCount] = riichi;
      threatWeightPercent[threatCount] = weight;
      threatSafeTileMask[threatCount] = safeMask;
      threatSujiAnchorMask[threatCount] = sujiMask;
      threatCount++;
    }

    private int visibleMeldValue(HandView opponent, int seat) {
      int value = 0;
      int seatWind = state.getJikaze(seat);
      int roundWind = state.getBakaze();
      for (int meldIndex = 0; meldIndex < opponent.meldCount(); meldIndex++) {
        Meld meld = opponent.meld(meldIndex);
        if (meld.type() != Meld.Type.CHI) {
          int tile = meld.baseTileType();
          value += Tile.isDragon(tile) ? 1 : 0;
          value += tile == seatWind ? 1 : 0;
          value += tile == roundWind ? 1 : 0;
        }
        if (meld.containsAkaTile()) {
          value++;
        }
        for (int tileIndex = 0; tileIndex < meld.size(); tileIndex++) {
          value += state.doraState().doraMultiplicity(meld.tileAt(tileIndex));
        }
      }
      return value;
    }

    private long preRiichiDiscardMask(int seat) {
      River river = state.river(seat);
      int declarationIndex = river.riichiDeclarationIndex();
      if (declarationIndex <= 0) {
        return 0L;
      }
      long mask = 0L;
      for (int index = 0; index < declarationIndex; index++) {
        mask |= 1L << river.discard(index).tileType();
      }
      return mask;
    }

    private int actionIndex(Action action) {
      for (int index = 0; index < decision.actionCount(); index++) {
        if (decision.action(index) == action || decision.action(index).equals(action)) {
          return index;
        }
      }
      throw new AssertionError("legal action is not projected: " + action);
    }

    private HandQuality analyzeCurrent(RiichiState riichiStatus) {
      return quality(hand, decision.analyzeCurrentWaits(riichiStatus));
    }

    private HandQuality analyzeTransition(
        int actionIndex, int transitionIndex, RiichiState riichiStatus) {
      HandView projectedHand = decision.handAfterTransition(actionIndex, transitionIndex);
      return quality(
          projectedHand,
          decision.analyzeTransitionWaits(actionIndex, transitionIndex, riichiStatus));
    }

    private HandQuality quality(HandView projectedHand, HandAnalysisBuffer analysis) {
      int secondOrder =
          analysis.minimumShanten() >= 1 && analysis.minimumShanten() <= 2
              ? secondOrderUkeire(projectedHand, analysis.minimumShanten())
              : 0;
      HandQuality quality = nextQuality();
      quality.bind(
          projectedHand,
          analysis,
          secondOrder,
          YakuRouteAnalyzer.confirmedRouteBits(
              state.getJikaze(player), state.getBakaze(), projectedHand));
      fillWaitValue(quality, analysis);
      return quality;
    }

    private HandQuality nextQuality() {
      if (qualityCursor >= qualities.length) {
        throw new IllegalStateException("heuristic analysis capacity exceeded");
      }
      return qualities[qualityCursor++];
    }

    private DiscardChoice discardChoice(
        Action action, HandQuality quality, int danger, long score) {
      if (discardChoiceCursor >= discardChoices.length) {
        throw new IllegalStateException("heuristic discard choice capacity exceeded");
      }
      return discardChoices[discardChoiceCursor++].bind(action, quality, danger, score);
    }

    private CallChoice callChoice(Action action, HandQuality quality, long score) {
      if (callChoiceCursor >= callChoices.length) {
        throw new IllegalStateException("heuristic call choice capacity exceeded");
      }
      return callChoices[callChoiceCursor++].bind(action, quality, score);
    }

    private void fillWaitValue(HandQuality quality, HandAnalysisBuffer waits) {
      int liveTiles = 0;
      int liveTypes = 0;
      int liveRonTiles = 0;
      int minimumRonPoints = Integer.MAX_VALUE;
      long ronPointTotal = 0L;
      for (int index = 0; index < waits.waitCount(); index++) {
        int tileType = waits.waitTileType(index);
        int remaining = unseenTileCounts[tileType];
        if (remaining <= 0) {
          continue;
        }
        if (waits.ronYakuBits(index) != 0L || waits.tsumoYakuBits(index) != 0L) {
          liveTiles += remaining;
          liveTypes++;
        }
        if (waits.ronYakuBits(index) != 0L) {
          int points = ronPoints(waits.ronBasePoints(index), dealer);
          liveRonTiles += remaining;
          minimumRonPoints = Math.min(minimumRonPoints, points);
          ronPointTotal += (long) remaining * points;
        }
      }
      if (liveRonTiles == 0) {
        minimumRonPoints = 0;
      }
      int averageRonPoints =
          liveRonTiles == 0 ? 0 : (int) Math.round(ronPointTotal / (double) liveRonTiles);
      quality.liveWaitTiles = liveTiles;
      quality.liveWaitTileTypes = liveTypes;
      quality.liveRonTiles = liveRonTiles;
      quality.minimumRonPoints = minimumRonPoints;
      quality.averageRonPoints = averageRonPoints;
    }

    private int secondOrderUkeire(HandView projectedHand, int currentShanten) {
      secondOrderShape.load(projectedHand);
      System.arraycopy(unseenTileCounts, 0, secondOrderRemaining, 0, Tile.NUM_TILE_TYPES);
      long weighted = 0L;
      int weights = 0;
      long improving = shapeAnalyzer.improvingTileTypeMask(projectedHand, currentShanten);
      for (long draws = improving; draws != 0L; draws &= draws - 1) {
        int draw = Long.numberOfTrailingZeros(draws);
        int weight = secondOrderRemaining[draw];
        if (weight <= 0) continue;
        secondOrderShape.add(draw);
        secondOrderRemaining[draw]--;
        int bestShanten = Integer.MAX_VALUE, bestUkeire = 0;
        try {
          for (long discards = secondOrderShape.concealedTileTypeMask();
              discards != 0L;
              discards &= discards - 1) {
            int discard = Long.numberOfTrailingZeros(discards);
            secondOrderShape.remove(discard);
            try {
              int shanten = shapeAnalyzer.calculateMinimum(secondOrderShape);
              int ukeire =
                  shapeAnalyzer.remainingImprovementTileCount(
                      secondOrderShape, secondOrderRemaining, shanten);
              if (shanten < bestShanten || shanten == bestShanten && ukeire > bestUkeire) {
                bestShanten = shanten;
                bestUkeire = ukeire;
              }
            } finally {
              secondOrderShape.add(discard);
            }
          }
        } finally {
          secondOrderRemaining[draw]++;
          secondOrderShape.remove(draw);
        }
        weighted += (long) weight * bestUkeire;
        weights += weight;
      }
      return weights == 0 ? 0 : (int) Math.round(weighted / (double) weights);
    }

    private boolean isFuriten(long shapeWaitMask, int newlyDiscardedTile) {
      long resultingRiverMask = ownRiverTileMask | 1L << newlyDiscardedTile;
      return state.isTemporaryFuriten(player) || (shapeWaitMask & resultingRiverMask) != 0L;
    }

    private int danger(int tile) {
      int total = 0;
      for (int threat = 0; threat < threatCount; threat++) {
        total += dangerAgainst(tile, threat) * threatWeightPercent[threat] / 100;
      }
      return total;
    }

    private int dangerAgainst(int tile, int threat) {
      long tileBit = 1L << tile;
      if ((threatSafeTileMask[threat] & tileBit) != 0L) {
        return DANGER_GENBUTSU;
      }

      int knownCopies = visibleTileCounts[tile] + hand.count(tile);
      if (Tile.isHonor(tile)) {
        int danger =
            knownCopies >= 4
                ? DANGER_FOURTH_HONOR
                : knownCopies >= 2 ? DANGER_SEEN_HONOR : DANGER_LIVE_HONOR;
        return danger + doraDanger(tile);
      }

      int number = Tile.numberOf(tile);
      int danger;
      if (number == 0 || number == 8) {
        danger = DANGER_NON_SUJI_TERMINAL;
      } else if (number == 1 || number == 7) {
        danger = DANGER_NON_SUJI_TWO_EIGHT;
      } else if (number == 2 || number == 6) {
        danger = DANGER_NON_SUJI_THREE_SEVEN;
      } else {
        danger = DANGER_NON_SUJI_MIDDLE;
      }

      int sujiAnchors = sujiAnchorCount(tile, threatSujiAnchorMask[threat]);
      int neededAnchors = number >= 3 && number <= 5 ? 2 : 1;
      if (sujiAnchors >= neededAnchors || isNoChance(tile)) {
        danger = Math.min(danger, DANGER_SUJI_OR_NO_CHANCE);
      } else if (sujiAnchors > 0) {
        danger = Math.min(danger, DANGER_LATE_ONE_CHANCE);
      } else if (isOneChance(tile)) {
        int oneChanceDanger =
            state.remainingWallTiles() >= ONE_CHANCE_RELIABLE_MIN_WALL_TILES
                ? DANGER_EARLY_ONE_CHANCE
                : DANGER_LATE_ONE_CHANCE;
        danger = Math.min(danger, oneChanceDanger);
      }
      return danger + doraDanger(tile);
    }

    private int doraDanger(int tile) {
      int danger = state.doraState().doraMultiplicity(tile) * 15;
      if (!Tile.isNumberTile(tile)) {
        return danger;
      }
      int suit = Tile.suitOf(tile);
      int number = Tile.numberOf(tile);
      int lower = number - 1;
      int upper = number + 1;
      if (lower >= 0) {
        danger += state.doraState().doraMultiplicity(suit * 9 + lower) * 5;
      }
      if (upper < 9) {
        danger += state.doraState().doraMultiplicity(suit * 9 + upper) * 5;
      }
      return danger;
    }

    private boolean isNoChance(int tile) {
      return sequenceConstraint(tile, Tile.TILES_PER_TYPE);
    }

    private boolean isOneChance(int tile) {
      return sequenceConstraint(tile, Tile.TILES_PER_TYPE - 1);
    }

    private boolean sequenceConstraint(int tile, int knownThreshold) {
      int number = Tile.numberOf(tile);
      int suitBase = Tile.suitOf(tile) * 9;
      boolean hasSequence = false;
      for (int start = Math.max(0, number - 2); start <= Math.min(number, 6); start++) {
        hasSequence = true;
        boolean blocked = false;
        for (int offset = 0; offset < 3; offset++) {
          int otherNumber = start + offset;
          if (otherNumber == number) {
            continue;
          }
          int other = suitBase + otherNumber;
          if (visibleTileCounts[other] + hand.count(other) >= knownThreshold) {
            blocked = true;
            break;
          }
        }
        if (!blocked) {
          return false;
        }
      }
      return hasSequence;
    }

    private int sujiAnchorCount(int tile, long anchorMask) {
      int number = Tile.numberOf(tile);
      int suitBase = Tile.suitOf(tile) * 9;
      int count = 0;
      if (number <= 5 && (anchorMask & 1L << (suitBase + number + 3)) != 0L) {
        count++;
      }
      if (number >= 3 && (anchorMask & 1L << (suitBase + number - 3)) != 0L) {
        count++;
      }
      return count;
    }

    private RiichiState currentRiichiStatus() {
      if (state.isDoubleRiichi(player)) {
        return RiichiState.DOUBLE_RIICHI;
      }
      return state.isRiichi(player) ? RiichiState.RIICHI : RiichiState.NONE;
    }

    private RiichiState declarationRiichiStatus() {
      return state.isFirstDraw(player) && !state.isFirstTurnCallOccurred()
          ? RiichiState.DOUBLE_RIICHI
          : RiichiState.RIICHI;
    }

    private boolean isYakuhai(int tile) {
      return HeuristicPlayer.isYakuhai(tile, state.getJikaze(player), state.getBakaze());
    }
  }

  private static boolean isYakuhai(int tile, int seatWind, int roundWind) {
    return Tile.isDragon(tile) || tile == seatWind || tile == roundWind;
  }

  private static int ronPoints(int basePoints, boolean dealer) {
    return ScorePayments.ronPoints(basePoints, dealer);
  }
}
