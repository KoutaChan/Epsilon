package com.epsilon.reviewer.engine;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.PublicEventLog;
import com.epsilon.core.PublicObservation;
import com.epsilon.core.River;
import com.epsilon.core.RoundPublicStateIndex;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.EngineDecisionBuffer;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.WinSettlementProjection;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * 方策が受け取る入力（公開観測と合法手）を、比べられる値へ写す試験の道具。公開観測が読める値を全て写し、借用局面を読む解析
 * （{@link PublicObservation#analyze}）の結果も写す。他家の伏せた手の枚数は、席の目の再生が持たない値なので写さない（どのモデルも読まない）。
 */
record PolicyInput(
    EngineDecisionKind kind,
    String round,
    int honba,
    int kyotaku,
    List<Integer> scores,
    String seats,
    boolean furiten,
    String hand,
    String melds,
    String rivers,
    List<String> events,
    String publicState,
    List<Integer> indicators,
    String dora,
    String turn,
    List<Integer> legal,
    String analysis) {

  static PolicyInput of(PublicObservation o, List<Action> legal) {
    int observer = o.observer();
    List<Integer> scores = new ArrayList<>();
    List<String> seats = new ArrayList<>();
    List<String> melds = new ArrayList<>();
    List<String> rivers = new ArrayList<>();
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      scores.add(o.score(seat));
      seats.add(
          join(
              o.seatWindTileType(seat),
              o.isBeforeFirstDiscard(seat),
              o.isRiichi(seat),
              o.isDoubleRiichi(seat),
              o.isIppatsu(seat)));
      melds.add(melds(o.hand(seat)));
      rivers.add(river(o.river(seat)));
    }
    List<Integer> indicators = new ArrayList<>();
    for (int i = 0; i < o.doraState().indicatorCount(); i++)
      indicators.add(o.doraState().indicatorTileType(i));
    return new PolicyInput(
        EngineDecisionKind.of(o, legal),
        join(
            observer,
            o.currentPlayer(),
            o.roundIndex(),
            o.dealer(),
            o.roundWindTileType(),
            o.turnNumber(),
            o.firstTurnCallOccurred(),
            o.isFirstTurn(),
            o.isWallExhausted(),
            o.remainingWallTiles(),
            o.totalRiverDahaiCount(),
            o.events().anyMeld()),
        o.honba(),
        o.riichiSticks(),
        scores,
        String.join("|", seats),
        o.isTemporaryFuriten(observer),
        hand(o.hand(observer)),
        String.join("|", melds),
        String.join("|", rivers),
        events(o.events()),
        publicState(o.publicState()),
        indicators,
        dora(o.doraState()),
        turn(o.turnEvent()),
        legal.stream().map(Action::toIndex).toList(),
        analysis(o, legal));
  }

  /** 最初に違う成分の名前と両方の値。同じなら null。 */
  static String difference(PolicyInput expected, PolicyInput actual) {
    try {
      for (RecordComponent component : PolicyInput.class.getRecordComponents()) {
        Object left = component.getAccessor().invoke(expected);
        Object right = component.getAccessor().invoke(actual);
        if (!Objects.equals(left, right))
          return component.getName() + ": expected <" + left + "> but was <" + right + ">";
      }
      return null;
    } catch (ReflectiveOperationException failure) {
      throw new AssertionError(failure);
    }
  }

  private static String hand(PublicObservation.VisibleHand hand) {
    List<Integer> counts = new ArrayList<>();
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) counts.add(hand.count(tile));
    return join(
        counts,
        hand.concealedTileCount(),
        hand.concealedAkaMask(),
        hand.ownedAkaMask(),
        hand.concealedTileTypeMask(),
        hand.concealedPairTileTypeMask());
  }

  private static String melds(PublicObservation.VisibleHand hand) {
    var text =
        new StringBuilder(
            join(
                hand.meldCount(),
                hand.isMenzen(),
                hand.kanCount(),
                hand.canonicalMeldSignature()));
    for (int i = 0; i < hand.meldCount(); i++)
      text.append(' ')
          .append(
              join(
                  hand.meld(i).canonicalId(),
                  hand.meldCallAfterRiverIndex(i),
                  hand.meldKanAfterRiverIndex(i),
                  hand.meldCreationEvent(i),
                  hand.meldKakanEvent(i)));
    return text.toString();
  }

  private static String river(River river) {
    List<Integer> events = new ArrayList<>();
    for (int i = 0; i < river.size(); i++) events.add(river.event(i));
    List<Integer> counts = new ArrayList<>();
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) counts.add(river.discardCount(tile));
    return join(
        events,
        river.riichiDeclarationIndex(),
        river.riichiDeclarationSequence(),
        river.postRiichiDiscardCount(),
        river.postRiichiTsumogiriCount(),
        river.discardedTileTypeMask(),
        counts);
  }

  private static List<String> events(PublicEventLog log) {
    List<String> events = new ArrayList<>();
    for (int event = 0; event < log.size(); event++)
      events.add(
          log.isDiscard(event)
              ? join(
                  "discard",
                  log.seat(event),
                  log.tile(event),
                  log.aka(event),
                  log.tsumogiri(event),
                  log.riichi(event),
                  log.calledBy(event),
                  log.turn(event),
                  log.sequence(event))
              : join(
                  "meld",
                  log.seat(event),
                  log.meld(event).canonicalId(),
                  log.afterRiver(event),
                  log.ponEvent(event)));
    return events;
  }

  private static String publicState(RoundPublicStateIndex index) {
    var text =
        new StringBuilder(
            join(
                index.copyVisibleTileCounts(),
                index.visibleDiscardOrMeldAkaMask(),
                index.totalDiscardCount(),
                index.totalKanCount()));
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
      List<Integer> counts = new ArrayList<>();
      for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++)
        counts.add(index.riverDiscardCount(seat, tile));
      text.append('|')
          .append(
              join(
                  index.kanCount(seat),
                  index.riverTileTypeMask(seat),
                  index.riichiGenbutsuTileTypeMask(seat),
                  index.riichiDeclarationIndex(seat),
                  index.discardsAfterRiichi(seat),
                  index.postRiichiTsumogiriCount(seat),
                  index.riichiDeclarationSequence(seat),
                  counts));
    }
    return text.toString();
  }

  private static String dora(PublicObservation.VisibleDora dora) {
    List<Integer> doraTiles = new ArrayList<>();
    for (int i = 0; i < dora.indicatorCount(); i++) doraTiles.add(dora.doraTileType(i));
    List<Integer> multiplicities = new ArrayList<>();
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++) {
      multiplicities.add(dora.doraMultiplicity(tile));
      multiplicities.add(dora.indicatorMultiplicity(tile));
    }
    return join(
        doraTiles, dora.visibleIndicatorAkaMask(), dora.doraTileTypesPacked(), multiplicities);
  }

  private static String turn(TurnEvent event) {
    return switch (event) {
      case TurnEvent.Draw draw ->
          join(
              "draw",
              draw.player(),
              draw.physicalTileId(),
              draw.drawSource(),
              draw.doraRevealPending());
      case TurnEvent.Discard discard ->
          join("discard", discard.player(), discard.tileType(), discard.isAkaTile());
      case TurnEvent.KanAttempt kan ->
          join(
              "kan",
              kan.player(),
              kan.tileType(),
              kan.isAkaTile(),
              kan.kanKind(),
              kan.doraRevealPending());
      case TurnEvent.None ignored -> "none";
    };
  }

  /** 借用局面を読む解析（Nano・Major・Pico の入力の元、Prism の和了と鳴いた後の打牌と同じ関数）。 */
  private static String analysis(PublicObservation o, List<Action> legal) {
    var buffer = new EngineDecisionBuffer();
    o.analyze(legal, buffer);
    var text = new StringBuilder();
    for (int action = 0; action < buffer.actionCount(); action++) {
      text.append(
          join(
              buffer.continuation(action),
              buffer.meldAkaSource(action),
              buffer.transitionCount(action)));
      for (int transition = 0; transition < buffer.transitionCount(action); transition++) {
        Action discard = buffer.discardAction(action, transition);
        text.append(' ').append(discard == null ? -1 : discard.toIndex());
      }
      WinSettlementProjection win = buffer.immediateWin(action);
      if (win != null) {
        text.append(
            join(
                "",
                win.visibleScore().yakuBits(),
                win.visibleScore().visibleHan(),
                win.visibleScore().fu(),
                win.visibleScore().basePoints(),
                win.paoApplies(),
                win.uraEligible(),
                win.uraIndicatorCount(),
                win.settlementAssumption()));
        for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
          text.append(
              join("", win.paymentFloor(seat), win.projectedRank(seat), win.scoreGap(seat)));
      }
      text.append(';');
    }
    for (int tile = 0; tile < Tile.NUM_TILE_TYPES; tile++)
      text.append(
          join(
              "",
              buffer.visibleTileCount(tile),
              buffer.unseenCopies(tile),
              buffer.doraMultiplicity(tile),
              buffer.doraIndicatorMultiplicity(tile)));
    text.append(
        join(
            "",
            buffer.visibleAkaTileTypeMask(),
            buffer.totalUnseenCopies(),
            buffer.ownRiverTileTypeMask()));
    for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++)
      text.append(
          join(
              "",
              buffer.riichiDeclarationIndex(seat),
              buffer.discardsAfterRiichi(seat),
              buffer.postRiichiTsumogiriCount(seat),
              buffer.riichiGenbutsuTileTypeMask(seat)));
    return text.toString();
  }

  private static String join(Object... values) {
    return Arrays.stream(values)
        .map(value -> value instanceof int[] array ? Arrays.toString(array) : String.valueOf(value))
        .collect(Collectors.joining(" "));
  }
}
