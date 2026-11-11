package com.epsilon.reviewer.json;

import com.epsilon.reviewer.engine.format.CandidateJson;
import com.epsilon.reviewer.engine.format.TableJson;
import com.epsilon.reviewer.engine.scenario.ScenarioAnswer;
import com.google.gson.TypeAdapter;
import com.google.gson.stream.JsonReader;
import com.google.gson.stream.JsonWriter;
import java.io.IOException;

/** もしもの答えを、解析の判断と同じ CandidateSet の形で書く（局面の draw が null でも書く）。読み込みはしない。 */
final class ScenarioAnswerJson extends TypeAdapter<ScenarioAnswer> {
  @Override
  public void write(JsonWriter out, ScenarioAnswer answer) throws IOException {
    boolean nulls = out.getSerializeNulls();
    out.setSerializeNulls(true);
    try {
      out.beginObject();
      CandidateJson.write(out, answer.decision());
      if (answer.table() != null) {
        out.name("table");
        TableJson.write(out, answer.table());
      }
      out.endObject();
    } finally {
      out.setSerializeNulls(nulls);
    }
  }

  @Override
  public ScenarioAnswer read(JsonReader in) {
    throw new UnsupportedOperationException("Scenario answers are only written.");
  }
}
