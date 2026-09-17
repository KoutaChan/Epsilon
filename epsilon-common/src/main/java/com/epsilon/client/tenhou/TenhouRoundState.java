package com.epsilon.client.tenhou;

import com.epsilon.client.ObservedRoundState;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Hand;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.WinLegality.RonStatus;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** 正規化済みの天鳳イベントを、観測可能な局面に反映する。 */
public final class TenhouRoundState {

  private static final Logger log = LoggerFactory.getLogger(TenhouRoundState.class);
  private static final int SELF = 0;

  private final ObservedRoundState round = new ObservedRoundState(SELF);

  public GameState state() {
    return round.state();
  }

  public Hand selfTiles() {
    return round.selfTiles();
  }

  public TenhouRoundOutcome apply(TenhouEvent.GameEvent event) {
    return switch (event) {
      case TenhouEvent.Init init -> {
        applyInit(init);
        yield TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.Draw draw -> {
        TurnEvent.Draw recorded = round.draw(draw.player(), draw.physicalTileId());
        yield draw.prompt().present()
            ? new TenhouRoundOutcome.DrawDecision(recorded, draw.prompt())
            : TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.Dahai dahai -> responseOutcome(applyDahai(dahai), dahai.prompt());
      case TenhouEvent.Naki naki -> applyNaki(naki);
      case TenhouEvent.ReachDeclared reach -> {
        round.reachDeclared(reach.player());
        yield TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.ReachAccepted reach -> {
        round.reachAccepted(reach.player(), reach.scores());
        yield TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.NewDora dora -> {
        applyDora(dora);
        yield TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.Agari agari -> {
        applyAgari(agari);
        yield TenhouRoundOutcome.None.INSTANCE;
      }
      case TenhouEvent.Ryukyoku ryukyoku -> {
        applyRyukyoku(ryukyoku);
        yield TenhouRoundOutcome.None.INSTANCE;
      }
    };
  }

  public void observeResponse(RonStatus status) {
    round.observeResponse(status);
  }

  public void completeResponse(
      TenhouRoundOutcome.ResponseDecision response, RonStatus status, Action selected) {
    round.completeResponse(status, response.prompt().canRon(), selected);
  }

  private void applyInit(TenhouEvent.Init init) {
    round.startRound(
        init.roundIndex(),
        init.dealer(),
        init.honba(),
        init.kyotaku(),
        init.doraIndicatorTileType(),
        init.initialHandPhysicalTileIds(),
        init.scores());
  }

  private TurnEvent.Discard applyDahai(TenhouEvent.Dahai dahai) {
    int player = dahai.player();
    int physicalTileId = dahai.physicalTileId();
    boolean tsumogiri =
        dahai.tsumogiri()
            || (player == SELF
                && state().getTurnEvent() instanceof TurnEvent.Draw draw
                && physicalTileId == draw.physicalTileId());
    return round.discard(
        player, Tile.typeOf(physicalTileId), Tile.isAka(physicalTileId), tsumogiri);
  }

  private TenhouRoundOutcome applyNaki(TenhouEvent.Naki naki) {
    TenhouMeldDecoder.DecodedMeld decoded = naki.meld();
    ObservedRoundState.MeldResult result =
        round.meld(naki.player(), decoded.meld(), decoded.consumedPhysicalTileIds());
    TenhouRoundOutcome outcome =
        switch (result) {
          case ObservedRoundState.OpenCall call ->
              naki.player() == SELF
                  ? new TenhouRoundOutcome.PostCallDahai(call.restriction())
                  : TenhouRoundOutcome.None.INSTANCE;
          case ObservedRoundState.Daiminkan ignored -> TenhouRoundOutcome.None.INSTANCE;
          case ObservedRoundState.Kan kan -> responseOutcome(kan.attempt(), naki.prompt());
        };
    log.debug("Call: P{} {}", naki.player(), naki.meld().meld().type());
    return outcome;
  }

  private void applyDora(TenhouEvent.NewDora dora) {
    round.revealDora(dora.indicatorTileType());
    log.debug("New dora indicator: {}", Tile.name(dora.indicatorTileType()));
  }

  private void applyAgari(TenhouEvent.Agari agari) {
    round.applyScores(agari.scores());
    String type = agari.winner() == agari.fromPlayer() ? "tsumo" : "ron";
    log.info("Win: P{} {} (from P{})", agari.winner(), type, agari.fromPlayer());
  }

  private void applyRyukyoku(TenhouEvent.Ryukyoku ryukyoku) {
    round.applyScores(ryukyoku.scores());
    log.info("Ryukyoku");
  }

  private static TenhouRoundOutcome responseOutcome(
      TurnEvent.ResponseSource source, TenhouEvent.ActionPrompt prompt) {
    return source.player() == SELF
        ? TenhouRoundOutcome.None.INSTANCE
        : new TenhouRoundOutcome.ResponseDecision(source, prompt);
  }
}
