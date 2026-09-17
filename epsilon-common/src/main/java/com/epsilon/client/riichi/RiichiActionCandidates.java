package com.epsilon.client.riichi;

import com.epsilon.client.ObservedRoundState;
import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.core.TurnEvent;
import com.epsilon.engine.ActionGenerator;
import com.epsilon.engine.EngineActionBuffer;
import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import java.util.ArrayList;
import java.util.List;

/** サーバーが提示した合法手の候補と共通の行動表現を、行動要求を受けた時点で対応付ける。 */
public final class RiichiActionCandidates {
  private final EngineActionBuffer actions = new EngineActionBuffer();
  private final EngineActionBuffer riichiDiscards = new EngineActionBuffer();
  private final List<JsonObject> wireActions = new ArrayList<>();
  private final ActionGenerator actionGenerator = new ActionGenerator();

  public List<Action> actions() {
    return actions;
  }

  public void bind(JsonObject observation, JsonArray possibleActions, ObservedRoundState round) {
    actions.clear();
    wireActions.clear();
    int drawnTile =
        observation.has("drawn_tile") && !observation.get("drawn_tile").isJsonNull()
            ? observation.get("drawn_tile").getAsInt()
            : -1;
    for (var value : observation.getAsJsonArray("_legal_actions")) {
      JsonObject raw = value.getAsJsonObject();
      String type = raw.get("action_type").getAsString();
      JsonObject wire = findWire(type, raw, possibleActions, drawnTile);
      if (type.equals("Riichi")) {
        if (!(round.state().getTurnEvent() instanceof TurnEvent.Draw draw)) {
          throw new IllegalArgumentException("Riichi declaration has no draw boundary");
        }
        actionGenerator.generateRiichiDahaiActionsInto(riichiDiscards, round.selfTiles(), draw);
        for (Action action : riichiDiscards) add(action, wire);
      } else add(decode(type, raw, drawnTile), wire);
    }
    if (actions.isEmpty())
      throw new IllegalArgumentException("Riichi request has no supported legal actions");
  }

  private void add(Action action, JsonObject wire) {
    if (!actions.contains(action)) {
      actions.add(action);
      wireActions.add(wire);
    }
  }

  public String encode(Action action, long requestId, ObservedRoundState round) {
    int index = actions.indexOf(action);
    if (index < 0) throw new IllegalArgumentException("Selected action is not a Riichi candidate");
    JsonObject reply = new JsonObject();
    for (var entry : wireActions.get(index).entrySet()) reply.add(entry.getKey(), entry.getValue());
    reply.addProperty("request_id", requestId);
    reply.addProperty("actor", round.selfSeat());
    if (action.type() == Action.Type.DAHAI) {
      reply.addProperty("tsumogiri", action.tileSelection().isTsumogiri());
    }
    if (action.type().isCall() || action.type() == Action.Type.RON_AGARI) {
      TurnEvent.ResponseSource source = (TurnEvent.ResponseSource) round.state().getTurnEvent();
      reply.addProperty("target", source.player());
    } else if (action.type() == Action.Type.TSUMO_AGARI)
      reply.addProperty("target", round.selfSeat());
    return reply.toString();
  }

  private static Action decode(String type, JsonObject raw, int drawnTile) {
    int tile = raw.get("tile").isJsonNull() ? -1 : raw.get("tile").getAsInt();
    JsonArray consumed = raw.getAsJsonArray("consume_tiles");
    return switch (type) {
      case "Discard" -> Action.dahai(Tile.typeOf(tile), Tile.isAka(tile), tile == drawnTile);
      case "Chi" ->
          Action.chiSequence(
              Math.min(
                  Tile.typeOf(tile),
                  Math.min(
                      Tile.typeOf(consumed.get(0).getAsInt()),
                      Tile.typeOf(consumed.get(1).getAsInt()))),
              Tile.typeOf(tile),
              MjaiTiles.containsAka(consumed, true));
      case "Pon" -> Action.pon(Tile.typeOf(tile), MjaiTiles.containsAka(consumed, true));
      case "Daiminkan" -> Action.daiminkan(Tile.typeOf(tile));
      case "Ankan" -> Action.ankan(Tile.typeOf(consumed.get(0).getAsInt()));
      case "Kakan" -> Action.kakan(Tile.typeOf(tile));
      case "Ron" -> Action.ronAgari();
      case "Tsumo" -> Action.tsumoAgari();
      case "Pass" -> Action.pass();
      case "KyushuKyuhai" -> Action.kyushuKyuhai();
      default -> throw new IllegalArgumentException("Unsupported Riichi action: " + type);
    };
  }

  private static JsonObject findWire(
      String type, JsonObject raw, JsonArray possibleActions, int drawnTile) {
    String mjaiType =
        switch (type) {
          case "Discard" -> "dahai";
          case "Chi" -> "chi";
          case "Pon" -> "pon";
          case "Daiminkan" -> "daiminkan";
          case "Ankan" -> "ankan";
          case "Kakan" -> "kakan";
          case "Riichi" -> "reach";
          case "Ron", "Tsumo" -> "hora";
          case "Pass" -> "none";
          case "KyushuKyuhai" -> "ryukyoku";
          default -> throw new IllegalArgumentException("Unsupported Riichi action: " + type);
        };
    for (var value : possibleActions) {
      JsonObject wire = value.getAsJsonObject();
      if (!mjaiType.equals(wire.get("type").getAsString())) continue;
      if (wire.has("pai")
          && !MjaiTiles.sameTile(
              raw.get("tile").getAsInt(), MjaiTiles.parse(wire.get("pai").getAsString()))) continue;
      if (wire.has("consumed")
          && !MjaiTiles.sameConsumed(
              raw.getAsJsonArray("consume_tiles"), wire.getAsJsonArray("consumed"))) continue;
      if (type.equals("Discard")
          && wire.has("tsumogiri")
          && wire.get("tsumogiri").getAsBoolean() != (raw.get("tile").getAsInt() == drawnTile))
        continue;
      return wire;
    }
    throw new IllegalArgumentException(
        "Riichi legal action is absent from possible_actions: " + type);
  }
}
