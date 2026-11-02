package com.epsilon.prism.run;

import ai.djl.Model;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.learn.CohortStore;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.learn.Teacher;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import java.nio.file.Files;
import java.nio.file.Path;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * run の行（rows-v2/&lt;t&gt;）のコホート A..B の TRAIN のラベル行と根の記録（{@link Selection#LABELS_AND_ROOTS}）で、白紙の網を
 * 生の Y・錨なしで当て、版の形で out/snapshot に書く（版の番号は B + 1）。
 *
 * <ul>
 *   <li>学習器の数（バッチ・学習率 lrPeak → lrFloor・勾配の上限・温度 λ_c・周回数 fitEpochs・学習のデバイス・展開のスレッド・種）は run の
 *       config.json から読む（{@link PrismConfig#read}）。run は読むだけ。
 *   <li>--cohorts: 学ぶコホートの範囲（A-B か A..B、両端を含む）。
 *   <li>--depth: 背骨の層の数（2 か 3）。幅は標準の形（H 256・F 512・G 256）。
 *   <li>--aux-result: 候補の結果の頭の重み α_r。読みの頭の重み α_o は config.json の auxOracle（--overrides で変えられる）。
 *   <li>--frontier: 前線の腕（on・off）。on なら網の形を前線の腕にし（{@link PrismShape#frontier()}）、前線は行の O から学習のときに
 *       作る。
 *   <li>--replicate R: 腕の複製の番号（網の初期値と行の並べ替えの種を変える。既定 0）。
 *   <li>--overrides k=v,...: config.json の欄の上書き（{@link #override}。例 lrPeak=0.0012,lrFloor=0.00006）。
 * </ul>
 */
public final class FitWorkflow implements CommandWorkflow<Path> {

  private static final Logger LOG = LoggerFactory.getLogger(FitWorkflow.class);

  /** 腕の種の座標（"v2ar"）。 */
  private static final long ARM = 0x76326172L;

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "refit",
        WorkflowKind.TRAINING,
        "refit --rows RUN --cohorts A-B --depth 2|3 --aux-result W --frontier off|on --out DIR"
            + " [--replicate R] [--overrides k=v,...]",
        12,
        16);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("rows");
    String cohorts = options.optionalText("cohorts", null);
    String depth = options.optionalText("depth", null);
    String auxResult = options.optionalText("aux-result", null);
    String frontier = options.optionalText("frontier", null);
    Path out = options.path("out");
    int replicate = Integer.parseInt(options.optionalText("replicate", "0"));
    String overrides = options.optionalText("overrides", null);
    options.requireEmpty();
    if (cohorts == null || depth == null || auxResult == null || frontier == null) {
      throw new IllegalArgumentException(
          "refit requires --cohorts, --depth, --aux-result and --frontier");
    }
    String[] range = cohorts.split("\\.\\.|-");
    if (range.length != 2) {
      throw new IllegalArgumentException("--cohorts A-B or A..B: " + cohorts);
    }
    int from = Integer.parseInt(range[0].strip());
    int to = Integer.parseInt(range[1].strip());
    if (from < 1 || to < from || replicate < 0) {
      throw new IllegalArgumentException(
          "invalid --cohorts " + cohorts + " or --replicate " + replicate);
    }
    PrismShape shape =
        PrismShape.standard(Integer.parseInt(depth)).withFrontier(switchValue(frontier));
    JsonObject tree =
        JsonParser.parseString(Files.readString(run.resolve("config.json"))).getAsJsonObject();
    if (overrides != null) {
      override(tree, overrides);
    }
    PrismConfig config = PrismConfig.read(tree);
    config.validate();
    AuxiliaryLoss auxiliary = new AuxiliaryLoss((float) config.auxOracle(), Float.parseFloat(auxResult));
    long seed = CrnSeeds.derive(config.seed(), CrnSeeds.Domain.LEARN, to, ARM, replicate);
    try (CohortStore store = new CohortStore()) {
      for (int cohort = from; cohort <= to; cohort++) {
        store.addCohort(CohortWriter.directory(run, cohort));
      }
      int[] rows = store.select(from, to, Selection.LABELS_AND_ROOTS);
      if (rows.length == 0) {
        throw new IllegalStateException(
            "No training rows in cohorts " + from + ".." + to + " of " + run);
      }
      int steps = PrismConfig.steps(Integer.MAX_VALUE, config.fitEpochs(), rows.length, config.batchSize());
      FitPlan plan =
          new FitPlan(config.fitSettings(steps), shape, auxiliary, FitPlan.TOKEN_BUDGET, seed);
      NetworkDevices devices = config.learnerDevices();
      LOG.info(
          "refit cohorts={}..{} rows={} steps={} shape={} auxiliary={} devices={}",
          from,
          to,
          rows.length,
          steps,
          shape,
          auxiliary,
          devices.asList());
      try (Model model = Fit.blank(shape, devices.primary(), seed)) {
        FitResult result =
            Fit.run(store, rows, model, plan, Teacher.RAW, devices, config.decodeThreads());
        Path snapshot = out.resolve("snapshot");
        Snapshot.write(
            snapshot,
            (PrismNetwork) model.getBlock(),
            PolicyParams.of(to + 1),
            new Snapshot.FitRecord(run.toString(), from, to, replicate, plan, result));
        LOG.info(
            "refit steps={} loss {} -> {} clipped={} {}s -> {}",
            result.steps(),
            result.firstLoss(),
            result.lastLoss(),
            result.clippedSteps(),
            String.format("%.1f", result.seconds()),
            snapshot);
        return snapshot;
      }
    }
  }

  /**
   * config.json の木の欄を k=v（読点区切り）で上書きする。値は {@link PrismConfig} の欄の型（数・真偽・文字列・enum）で、設定と同じに読む。
   *
   * @param tree config.json の木
   * @param overrides k=v の列
   * @throws IllegalArgumentException 書き方の誤り、知らない欄か列の欄か形式、型の合わない値のとき
   */
  static void override(JsonObject tree, String overrides) {
    for (String pair : overrides.split(",")) {
      int equals = pair.indexOf('=');
      if (equals < 1) throw new IllegalArgumentException("--overrides k=v: " + pair);
      String key = pair.substring(0, equals).strip();
      tree.add(key, PrismConfig.scalarValue(key, pair.substring(equals + 1).strip()));
    }
  }

  /** on・off を読む。 */
  private static boolean switchValue(String value) {
    return switch (value) {
      case "on" -> true;
      case "off" -> false;
      default -> throw new IllegalArgumentException("--frontier must be on or off: " + value);
    };
  }

  @Override
  public String summarize(Path snapshot) {
    return "refit -> " + snapshot;
  }
}
