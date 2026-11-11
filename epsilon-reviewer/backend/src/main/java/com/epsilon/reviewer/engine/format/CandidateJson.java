package com.epsilon.reviewer.engine.format;

import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;

/**
 * 判断の候補（protocol/result.ts の CandidateSet）。各行は [行動 ID, 値] で、鳴きと槓の行だけは [行動 ID, 値, 牌] になる。値は float32 を
 * 最短の十進表記（{@link Float#toString}）で書く。損失も丸めない。
 */
public final class CandidateJson {
  private CandidateJson() {}

  /** candidates と chosen のフィールドを書く。chosen は、牌譜に無い判断では書かない。 */
  public static void write(JsonWriter json, EvaluatedDecision decision) throws IOException {
    json.name("candidates").beginArray();
    for (int i = 0; i < decision.size(); i++) {
      json.beginArray().value(decision.actionIds()[i]).value(decision.values()[i]);
      String[] tiles = decision.tiles()[i];
      if (tiles != null) {
        json.beginArray();
        for (String tile : tiles) json.value(tile);
        json.endArray();
      }
      json.endArray();
    }
    json.endArray();
    if (decision.chosen() >= 0) json.name("chosen").value(decision.chosen());
  }
}
