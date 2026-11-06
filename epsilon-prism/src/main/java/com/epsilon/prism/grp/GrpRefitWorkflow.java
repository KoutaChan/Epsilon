package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.RunFiles;
import com.epsilon.workflow.CommandWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import com.epsilon.workflow.WorkflowDefinition;
import com.epsilon.workflow.WorkflowKind;
import com.epsilon.workflow.WorkflowOptions;
import com.google.gson.JsonObject;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Stream;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 手動の GRP の学習の要求（grp-refit）。run の config.json を読み、grp/refit-request.json に取り置き h（既定は対局の記録がある最新の封印済みの
 * コホート）を {@code {"holdout": h}} で書くだけで、学習は学習の JVM の GRP の裏の作業（{@link GrpBackground}）が次に頼まれたとき（次の封印の
 * 後か起動の初め）に行う。前の要求がまだ読まれていなければ置き換える。phiSource = SELF_GRP の run だけ。
 */
public final class GrpRefitWorkflow implements CommandWorkflow<Integer> {
  private static final Logger LOG = LoggerFactory.getLogger(GrpRefitWorkflow.class);

  /** refit-request.json の取り置きの鍵。 */
  public static final String HOLDOUT = "holdout";

  private static final Pattern GAME_LOG = Pattern.compile("c(\\d+)\\.bin\\.gz");

  @Override
  public WorkflowDefinition definition() {
    return WorkflowDefinition.range(
        "grp-refit", WorkflowKind.TRAINING, "grp-refit --run DIR [--holdout T]", 2, 4);
  }

  @Override
  public Integer execute(WorkflowArguments arguments) throws Exception {
    WorkflowOptions options = arguments.namedOptions();
    Path run = options.path("run");
    String holdoutText = options.optionalText("holdout", null);
    options.requireEmpty();
    PrismConfig config = selfGrpConfig(run);
    var files = new GrpFiles(run);
    int holdout = holdoutText == null ? latestRecorded(files) : Integer.parseInt(holdoutText);
    if (!GrpRefitter.recorded(files, holdout))
      throw new IllegalArgumentException("Cohort " + holdout + " of " + run + " is not sealed with a game log");
    RefitWindow window =
        RefitWindow.ending(holdout, config.grpWindow())
            .orElseThrow(() -> new IllegalArgumentException("No usable cohort before holdout " + holdout));
    var request = new JsonObject();
    request.addProperty(HOLDOUT, holdout);
    RunFiles.writeAtomically(files.refitRequest(), request + "\n");
    LOG.info("grp-refit requested holdout {} window {}..{} in {}", holdout, window.first(), window.last(), files.refitRequest());
    return 0;
  }

  @Override
  public String summarize(Integer status) {
    return "status=" + status;
  }

  @Override
  public boolean successful(Integer status) {
    return status == 0;
  }

  /**
   * run の config.json（phiSource = SELF_GRP であること）。
   *
   * @throws IllegalArgumentException SELF_GRP の run でないとき
   */
  static PrismConfig selfGrpConfig(Path run) throws IOException {
    PrismConfig config = PrismConfig.read(Files.readString(run.resolve("config.json")));
    config.validate();
    if (config.phiSource() != PhiSource.SELF_GRP)
      throw new IllegalArgumentException(run + " has phiSource = " + config.phiSource() + ", not SELF_GRP");
    return config;
  }

  /** 対局の記録がある最新の封印済みのコホート。 */
  private static int latestRecorded(GrpFiles files) throws IOException {
    Path directory = files.games(1).getParent();
    int latest = 0;
    if (Files.isDirectory(directory)) {
      try (Stream<Path> names = Files.list(directory)) {
        for (Path name : names.toList()) {
          Matcher matcher = GAME_LOG.matcher(name.getFileName().toString());
          if (matcher.matches()) {
            int c = Integer.parseInt(matcher.group(1));
            if (c > latest && GrpRefitter.recorded(files, c)) latest = c;
          }
        }
      }
    }
    if (latest == 0) throw new IllegalArgumentException("No sealed cohort with a game log in " + files.run());
    return latest;
  }
}
