package com.epsilon.prism.workflow;

import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.model.PrismModel;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 版の網の重みを要素ごとに平均して 1 つの版にする（{@link PrismModel#setToMean}）。版ごとの揺れを平らにした網を evaluate で測るのに使う。
 *
 * <p>--snapshots は版のディレクトリを読点で区切ったもの（全て同じ網の形）。--version は書く版の番号（既定は最後の版の番号）。
 */
public final class AverageWorkflow implements CommandWorkflow<String> {
  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "average",
        WorkflowKind.TRAINING,
        "average --snapshots DIR,DIR,... --out DIR [--version N] [--device D]",
        4,
        8);
  }

  @Override
  public String execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    List<Path> snapshots =
        Arrays.stream(options.optionalText("snapshots", "").split(","))
            .map(String::strip)
            .filter(text -> !text.isEmpty())
            .map(Path::of)
            .toList();
    if (snapshots.size() < 2) throw new IllegalArgumentException("--snapshots needs two or more");
    Path out = options.path("out");
    String version = options.optionalText("version", null);
    NetworkDevices devices =
        NetworkDevices.of(PrismConfig.device(options.optionalText("device", "cpu")));
    options.requireEmpty();
    List<PrismModel> models = new ArrayList<>(snapshots.size());
    try {
      for (Path snapshot : snapshots)
        models.add(PrismModel.load(Snapshot.model(snapshot), devices, false));
      try (var average = PrismModel.load(Snapshot.model(snapshots.getFirst()), devices, false)) {
        average.setToMean(models);
        int number =
            version == null
                ? Snapshot.params(snapshots.getLast()).version()
                : Integer.parseInt(version);
        Snapshot.write(out, average, PolicyParams.of(number));
      }
    } finally {
      for (PrismModel model : models) model.close();
    }
    return "averaged=" + snapshots.size() + " to=" + out;
  }

  @Override
  public String summarize(String result) {
    return result;
  }
}
