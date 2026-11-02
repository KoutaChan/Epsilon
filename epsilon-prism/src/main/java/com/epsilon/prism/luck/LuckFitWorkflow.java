package com.epsilon.prism.luck;

import ai.djl.Device;
import ai.djl.engine.Engine;
import ai.djl.ndarray.NDManager;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 標本のコホートから運の網 Ω を白紙で当て、保留の R² を報告する。
 *
 * <ul>
 *   <li>--samples: 標本のコホートのディレクトリ（luck/ を封印したもの）。複数はコンマで区切る。
 *   <li>--out: 書く先。網は out/network/（weights.bin と model.json）、報告は out/luck-fit.json（{@link LuckFit.Report}）。
 *   <li>--width: 網の幅（256 か 384。既定 256）。深さは 2 に固定。
 *   <li>--epochs: 当てはめの標本の周回数（既定 1）。
 *   <li>--seed: 網の初期値と並べ替えの種（既定 1）。
 *   <li>--threads: 符号化のスレッドの数（既定は論理 CPU の半分）。
 * </ul>
 *
 * <p>デバイスはエンジンの既定（GPU があれば GPU 0、BF16）。塊は 8192、AdamW は学習率 2e-4 一定・減衰 0.01、保留は標本の半荘の 5%。
 */
public final class LuckFitWorkflow implements CommandWorkflow<Path> {

  /** 報告のファイルの名前。 */
  public static final String REPORT = "luck-fit.json";

  /** 網のディレクトリの名前。 */
  public static final String NETWORK = "network";

  private static final Set<Integer> WIDTHS = Set.of(256, 384);
  private static final Logger LOG = LoggerFactory.getLogger(LuckFitWorkflow.class);
  private static final Gson GSON =
      new GsonBuilder().setPrettyPrinting().serializeSpecialFloatingPointValues().create();

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "luck-fit",
        WorkflowKind.TRAINING,
        "luck-fit --samples COHORT[,COHORT...] --out DIR [--width 256|384] [--epochs E]"
            + " [--seed S] [--threads T]",
        4,
        12);
  }

  @Override
  public Path execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    String samplesOption = options.optionalText("samples", null);
    Path out = options.path("out");
    int width = Integer.parseInt(options.optionalText("width", "256"));
    double epochs = Double.parseDouble(options.optionalText("epochs", "1"));
    long seed = Long.parseLong(options.optionalText("seed", "1"));
    int threads =
        Integer.parseInt(
            options.optionalText(
                "threads",
                Integer.toString(Math.max(1, Runtime.getRuntime().availableProcessors() / 2))));
    options.requireEmpty();
    if (samplesOption == null) {
      throw new IllegalArgumentException("luck-fit requires --samples");
    }
    if (!WIDTHS.contains(width)) {
      throw new IllegalArgumentException("--width must be 256 or 384: " + width);
    }
    List<Path> cohorts = new ArrayList<>();
    for (String cohort : samplesOption.split(",")) {
      cohorts.add(Path.of(cohort.strip()));
    }
    LuckFit.Plan plan = LuckFit.Plan.standard(width, epochs, seed);
    LuckSamples samples = LuckSamples.load(cohorts);
    Device device = Engine.getInstance().defaultDevice();
    LOG.info("luck-fit cohorts={} samples={} device={}", cohorts, samples.size(), device);
    try (NDManager manager = NDManager.newBaseManager(device)) {
      LuckNetwork network = LuckNetwork.blank(plan.shape(), manager, plan.seed());
      LuckFit.Report report = LuckFit.run(samples, network, manager, plan, threads);
      Files.createDirectories(out);
      network.save(out.resolve(NETWORK));
      Files.writeString(out.resolve(REPORT), GSON.toJson(report));
      LOG.info(
          "luck-fit steps={} loss {} -> {} holdout R2={} reduction={} {} actorMeanZ={} {}s -> {}",
          report.steps(),
          report.firstLoss(),
          report.lastLoss(),
          report.holdout().all().r2(),
          report.holdout().all().reduction(),
          report.holdout().byEvent(),
          report.holdout().actorMeanZ(),
          String.format("%.1f", report.seconds()),
          out);
    }
    return out;
  }

  @Override
  public String summarize(Path out) {
    return "luck-fit -> " + out;
  }
}
