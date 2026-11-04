package com.epsilon.reviewer.engine.format;

import com.epsilon.reviewer.engine.view.Meld;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RiverTile;
import com.epsilon.reviewer.engine.view.TableState;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.List;

/**
 * 局面の JSON（protocol/result.ts の TableState）。全欄を書き、draw は無ければ null、河の印は真のときだけ書く。
 * 書き手は null を書く設定にする。
 */
public final class TableJson {
  private TableJson() {}

  public static void write(JsonWriter json, TableState table) throws IOException {
    json.beginObject().name("players").beginArray();
    for (PlayerState player : table.players()) player(json, player);
    json.endArray().name("dora");
    strings(json, table.dora());
    json.name("honba").value(table.honba());
    json.name("kyotaku").value(table.kyotaku());
    json.name("remaining").value(table.remaining());
    json.name("activeSeat").value(table.activeSeat());
    json.endObject();
  }

  static void player(JsonWriter json, PlayerState player) throws IOException {
    json.beginObject();
    json.name("score").value(player.score());
    json.name("riichi").value(player.riichi());
    json.name("hand");
    strings(json, player.hand());
    json.name("draw").value(player.draw());
    json.name("river").beginArray();
    for (RiverTile tile : player.river()) riverTile(json, tile);
    json.endArray().name("melds").beginArray();
    for (Meld meld : player.melds()) meld(json, meld);
    json.endArray().endObject();
  }

  static void riverTile(JsonWriter json, RiverTile tile) throws IOException {
    json.beginObject().name("tile").value(tile.tile());
    if (tile.tsumogiri()) json.name("tsumogiri").value(true);
    if (tile.riichi()) json.name("riichi").value(true);
    if (tile.called()) json.name("called").value(true);
    json.endObject();
  }

  static void meld(JsonWriter json, Meld meld) throws IOException {
    json.beginObject().name("type").value(meld.type()).name("tiles");
    strings(json, meld.tiles());
    json.name("from").value(meld.from());
    json.name("calledIndex").value(meld.calledIndex());
    json.name("addedIndex").value(meld.addedIndex());
    json.endObject();
  }

  static void string(JsonWriter json, String value) throws IOException {
    json.value(value);
  }

  static void strings(JsonWriter json, List<String> values) throws IOException {
    json.beginArray();
    for (String value : values) json.value(value);
    json.endArray();
  }
}
