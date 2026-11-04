package com.epsilon.reviewer.engine.format;

import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.TableState;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;
import java.util.List;
import java.util.Objects;

/**
 * 前の局面からの差（protocol/result.ts の StateDelta）。変わった席だけを席の順に書き、配列は前後の同じ並びを除いた 1 か所の置き換え
 * [位置, 消す数, ...値] にする。draw の null はツモ牌が無くなったこと。
 */
public final class TableDiff {
  private TableDiff() {}

  /** 差を delta の欄として書く。同じ局面なら何も書かない。 */
  public static void write(JsonWriter json, TableState before, TableState after)
      throws IOException {
    if (before.equals(after)) return;
    json.name("delta").beginObject();
    boolean players = false;
    for (int seat = 0; seat < after.players().size(); seat++) {
      PlayerState old = before.players().get(seat);
      PlayerState next = after.players().get(seat);
      if (old.equals(next)) continue;
      if (!players) json.name("players").beginArray();
      players = true;
      json.beginObject().name("seat").value(seat);
      if (old.score() != next.score()) json.name("score").value(next.score());
      if (old.riichi() != next.riichi()) json.name("riichi").value(next.riichi());
      splice(json, "hand", old.hand(), next.hand(), TableJson::string);
      if (!Objects.equals(old.draw(), next.draw())) json.name("draw").value(next.draw());
      splice(json, "river", old.river(), next.river(), TableJson::riverTile);
      splice(json, "melds", old.melds(), next.melds(), TableJson::meld);
      json.endObject();
    }
    if (players) json.endArray();
    splice(json, "dora", before.dora(), after.dora(), TableJson::string);
    if (before.honba() != after.honba()) json.name("honba").value(after.honba());
    if (before.kyotaku() != after.kyotaku()) json.name("kyotaku").value(after.kyotaku());
    if (before.remaining() != after.remaining()) json.name("remaining").value(after.remaining());
    if (before.activeSeat() != after.activeSeat())
      json.name("activeSeat").value(after.activeSeat());
    json.endObject();
  }

  private static <T> void splice(
      JsonWriter json, String name, List<T> before, List<T> after, ValueWriter<T> values)
      throws IOException {
    int prefix = 0;
    int oldEnd = before.size();
    int newEnd = after.size();
    while (prefix < oldEnd
        && prefix < newEnd
        && Objects.equals(before.get(prefix), after.get(prefix))) prefix++;
    if (prefix == oldEnd && prefix == newEnd) return;
    while (oldEnd > prefix
        && newEnd > prefix
        && Objects.equals(before.get(oldEnd - 1), after.get(newEnd - 1))) {
      oldEnd--;
      newEnd--;
    }
    json.name(name).beginArray().value(prefix).value(oldEnd - prefix);
    for (int i = prefix; i < newEnd; i++) values.write(json, after.get(i));
    json.endArray();
  }

  @FunctionalInterface
  private interface ValueWriter<T> {
    void write(JsonWriter json, T value) throws IOException;
  }
}
