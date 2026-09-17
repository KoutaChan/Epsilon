package com.epsilon.client.riichi;

import com.epsilon.client.ObservedRoundState;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/** 自席向け MJAI 差分を、天鳳と共有する観測状態へ一度だけ反映する。 */
public final class MjaiRoundState {
  private static final int[] NO_TILES = new int[0];
  private final ObservedRoundState round;
  private final ActionGenerator actionGenerator = new ActionGenerator();
  private int roundNumber;
  private boolean selfReachDeclared;

  public MjaiRoundState(int selfSeat) {
    round = new ObservedRoundState(selfSeat);
  }

  public ObservedRoundState round() {
    return round;
  }

  public int roundNumber() {
    return roundNumber;
  }

  public boolean selfReachDeclared() {
    return selfReachDeclared;
  }

  public void apply(JsonArray events) {
    for (var encoded : events) {
      apply(JsonParser.parseString(encoded.getAsString()).getAsJsonObject());
    }
  }

  private void apply(JsonObject event) {
    String type = event.get("type").getAsString();
    switch (type) {
      case "start_kyoku" -> startRound(event);
      case "tsumo" -> {
        int actor = event.get("actor").getAsInt();
        int physical =
            actor == round.selfSeat()
                ? MjaiTiles.nextPhysicalTile(round.selfTiles(), event.get("pai").getAsString())
                : -1;
        round.draw(actor, physical);
      }
      case "dahai" -> {
        int tile = MjaiTiles.parse(event.get("pai").getAsString());
        observe(
            round.discard(
                event.get("actor").getAsInt(),
                Tile.typeOf(tile),
                Tile.isAka(tile),
                event.get("tsumogiri").getAsBoolean()));
      }
      case "chi", "pon", "daiminkan", "ankan", "kakan" -> applyMeld(type, event);
      case "reach" -> {
        int actor = event.get("actor").getAsInt();
        round.reachDeclared(actor);
        if (actor == round.selfSeat()) selfReachDeclared = true;
      }
      case "reach_accepted" ->
          round.reachAccepted(
              event.get("actor").getAsInt(),
              event.has("scores") ? integers(event.getAsJsonArray("scores")) : null);
      case "dora" ->
          round.revealDora(Tile.typeOf(MjaiTiles.parse(event.get("dora_marker").getAsString())));
      case "hora", "ryukyoku", "end_kyoku" -> {
        if (event.has("scores")) round.applyScores(integers(event.getAsJsonArray("scores")));
        else if (event.has("deltas")) {
          JsonArray deltas = event.getAsJsonArray("deltas");
          for (int seat = 0; seat < GameState.NUM_PLAYERS; seat++) {
            round.state().addScore(seat, deltas.get(seat).getAsInt());
          }
        }
      }
      default -> {
        /* start_game と未知の補助イベントは局状態を変更しない。 */
      }
    }
  }

  private void startRound(JsonObject event) {
    if (event.getAsJsonArray("tehais").size() != GameState.NUM_PLAYERS
        || event.getAsJsonArray("scores").size() != GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("Riichi play requires a four-player game");
    }
    int wind = Tile.typeOf(MjaiTiles.parse(event.get("bakaze").getAsString())) - Tile.TON;
    int roundIndex = wind * 4 + event.get("kyoku").getAsInt() - 1;
    round.startRound(
        roundIndex,
        event.get("oya").getAsInt(),
        event.get("honba").getAsInt(),
        event.get("kyotaku").getAsInt(),
        Tile.typeOf(MjaiTiles.parse(event.get("dora_marker").getAsString())),
        MjaiTiles.initialHand(
            event.getAsJsonArray("tehais").get(round.selfSeat()).getAsJsonArray()),
        integers(event.getAsJsonArray("scores")));
    roundNumber++;
    selfReachDeclared = false;
  }

  private void applyMeld(String type, JsonObject event) {
    int actor = event.get("actor").getAsInt();
    JsonArray consumed = event.getAsJsonArray("consumed");
    int tile =
        MjaiTiles.parse((type.equals("ankan") ? consumed.get(0) : event.get("pai")).getAsString());
    int tileType = Tile.typeOf(tile);
    boolean consumedAka = MjaiTiles.containsAka(consumed, false);
    Meld meld =
        switch (type) {
          case "chi" ->
              Meld.chi(
                  Math.min(
                      tileType,
                      Math.min(
                          Tile.typeOf(MjaiTiles.parse(consumed.get(0).getAsString())),
                          Tile.typeOf(MjaiTiles.parse(consumed.get(1).getAsString())))),
                  tileType,
                  Meld.AkaSource.forCall(Tile.isAka(tile), consumedAka));
          case "pon" ->
              Meld.pon(
                  tileType,
                  source(actor, event),
                  Meld.AkaSource.forCall(Tile.isAka(tile), consumedAka));
          case "daiminkan" ->
              Meld.daiminkan(
                  tileType,
                  source(actor, event),
                  Meld.AkaSource.forCall(Tile.isAka(tile), consumedAka));
          case "ankan" -> Meld.ankan(tileType, consumedAka);
          case "kakan" ->
              Meld.kakan(round.state().hand(actor).requirePon(tileType), Tile.isAka(tile));
          default -> throw new IllegalArgumentException("Unknown MJAI meld: " + type);
        };
    int[] physical;
    if (type.equals("kakan")) {
      physical =
          new int[] {
            actor == round.selfSeat()
                ? round.selfTiles().physicalTileId(tileType, Tile.isAka(tile), 0)
                : tile
          };
    } else
      physical =
          actor == round.selfSeat() ? MjaiTiles.consumed(round.selfTiles(), consumed) : NO_TILES;
    var result = round.meld(actor, meld, physical);
    if (result instanceof ObservedRoundState.Kan kan) observe(kan.attempt());
  }

  private void observe(TurnEvent.ResponseSource source) {
    if (source.player() != round.selfSeat()) {
      round.observeResponse(actionGenerator.ronStatus(round.state(), round.selfSeat(), source));
    }
  }

  private static Meld.RelativeSource source(int actor, JsonObject event) {
    return Meld.RelativeSource.fromPlayerOffset((event.get("target").getAsInt() - actor + 4) % 4);
  }

  public static int[] integers(JsonArray values) {
    int[] result = new int[values.size()];
    for (int index = 0; index < result.length; index++)
      result[index] = values.get(index).getAsInt();
    return result;
  }
}
