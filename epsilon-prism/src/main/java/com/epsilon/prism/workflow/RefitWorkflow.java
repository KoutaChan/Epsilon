package com.epsilon.prism.workflow;

import com.epsilon.prism.run.Pipeline;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.Gson;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * run の反復 t の学習を、収集せずにやり直す（{@link Pipeline#refit}）。学習器の作り方（learner）・並べ替えの種（replicate）・学習の行の範囲と
 * 間引き（{@link Pipeline.RowWindow}）・設定の上書きだけを変えて、同じ行・同じ版 t から学習の網を作り、out/snapshot に版の形で書く。設定は run の
 * config.json に --learner と --overrides を重ねたもの（run は読むだけ）。
 *
 * <ul>
 *   <li>--cohorts A-B: 学習の行のコホートの範囲（既定は記憶の窓 max(1, t−memoryCohorts+1)..t。保留の指標はいつもコホート t）
 *   <li>--fraction F: 学習の行を残す割合、--max-rows N: 学習の行の数の上限の目安（どちらも行ごとの決まった硬貨で間引く）
 *   <li>--overrides k=v,k=v: config.json の欄の上書き（例 memoryDecay=1.0,fitSteps=10000）。型は元の欄に合わせ、知らない欄は拒む
 * </ul>
 */
public final class RefitWorkflow implements CommandWorkflow<Path> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "refit",
        WorkflowKind.TRAINING,
        "refit --run DIR --iteration T --out DIR [--learner crossfit|single] [--replicate R]"
            + " [--cohorts A-B] [--fraction F] [--max-rows N] [--overrides k=v,...]",
        6,
        18);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    int t = Integer.parseInt(options.optionalText("iteration", "0"));
    Path out = options.path("out");
    String learner = options.optionalText("learner", null);
    int replicate = Integer.parseInt(options.optionalText("replicate", "0"));
    String cohorts = options.optionalText("cohorts", null);
    double fraction = Double.parseDouble(options.optionalText("fraction", "1"));
    long maxRows = Long.parseLong(options.optionalText("max-rows", "0"));
    String overrides = options.optionalText("overrides", null);
    options.requireEmpty();
    if (t < 1) throw new IllegalArgumentException("refit requires --iteration T (T >= 1)");
    if (replicate < 0) throw new IllegalArgumentException("--replicate " + replicate);
    int from = 0, to = 0;
    if (cohorts != null) {
      String[] range = cohorts.split("-");
      if (range.length != 2) throw new IllegalArgumentException("--cohorts A-B: " + cohorts);
      from = Integer.parseInt(range[0].strip());
      to = Integer.parseInt(range[1].strip());
      if (from < 1 || to > t) throw new IllegalArgumentException("--cohorts outside 1.." + t);
    }
    JsonObject tree =
        JsonParser.parseString(Files.readString(run.resolve("config.json"))).getAsJsonObject();
    if (learner != null) tree.addProperty("learner", learner);
    else if (!tree.has("learner")) tree.addProperty("learner", PrismConfig.LEARNER_CROSSFIT);
    if (overrides != null) override(tree, overrides);
    PrismConfig config = new Gson().fromJson(tree, PrismConfig.class);
    return Pipeline.refit(
        run, config, t, replicate, new Pipeline.RowWindow(from, to, fraction, maxRows), out);
  }

  /** config.json の木の欄を k=v（読点区切り）で上書きする。値は元の欄の型（数・真偽・文字列）で読む。 */
  static void override(JsonObject tree, String overrides) {
    for (String pair : overrides.split(",")) {
      int equals = pair.indexOf('=');
      if (equals < 1) throw new IllegalArgumentException("--overrides k=v: " + pair);
      String key = pair.substring(0, equals).strip(), value = pair.substring(equals + 1).strip();
      JsonElement old = tree.get(key);
      if (old == null || !old.isJsonPrimitive())
        throw new IllegalArgumentException("Unknown or non-scalar config field: " + key);
      JsonPrimitive primitive = old.getAsJsonPrimitive();
      if (primitive.isBoolean()) {
        if (!value.equalsIgnoreCase("true") && !value.equalsIgnoreCase("false"))
          throw new IllegalArgumentException(key + " must be true or false: " + value);
        tree.addProperty(key, Boolean.parseBoolean(value));
      } else if (primitive.isNumber()) {
        JsonElement number = JsonParser.parseString(value);
        if (!number.isJsonPrimitive() || !number.getAsJsonPrimitive().isNumber())
          throw new IllegalArgumentException(key + " must be a number: " + value);
        tree.add(key, number);
      }
      else tree.addProperty(key, value);
    }
  }

  @Override
  public String summarize(Path snapshot) {
    return "refit -> " + snapshot;
  }
}
