package com.epsilon.reviewer.engine.format;

import com.epsilon.reviewer.engine.review.EvaluatedDecision;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;

/**
 * 判断の候補（protocol/result.ts の CandidateSet）。行は [行動 ID, 値] か、鳴き・槓だけ [行動 ID, 値, 牌]。値は float32 の最短の十進
 * （{@link Float#toString}）で、損失は丸めない。
 */
public final class CandidateJson {
  private CandidateJson() {}

  /** candidates・chosen・probabilities の欄を書く（chosen と probabilities は無ければ書かない）。 */
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
    if (decision.probabilities() != null) {
      json.name("probabilities").beginArray();
      for (float probability : decision.probabilities()) json.value(probability);
      json.endArray();
    }
  }
}
