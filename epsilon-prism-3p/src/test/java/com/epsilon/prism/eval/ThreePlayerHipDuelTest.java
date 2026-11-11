package com.epsilon.prism.eval;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.eval.DuelResult;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Main;
import com.epsilon.prism.ThreePlayerSettings;
import com.epsilon.prism.app.Evaluate;
import com.epsilon.prism.backend.PerturbedNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PolicyParams;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism3p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.Assert;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 同じバージョンどうしの三人打ちの決闘を HIP で打つ（rocm のグループ。GPU 1 枚）。乱数で初期化した三人打ちのバージョン（v2 の形と V3）を
 * NATIVE の推論で開くと、HIP のバックエンドで動く。evaluate で自分自身を相手に、1v2・2v1 の全ての席の割り当てを打つ（ツモの引き方は、席ごとのツモ列と山の順の 2 つ）。
 * 同じ行の s はバッチの組み方に依らずビット単位で同じなので、全ての牌山で全ての割り当ての半荘が同じ点で終わり、牌山ごとの差 δ は 0 になる。
 */
public class ThreePlayerHipDuelTest {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerHipDuelTest.class);

  private static final int WALLS = 24;

  /** 1 つの牌山で打つ半荘の数。1v2 と 2v1 の全ての席の割り当て（3 + 3）。 */
  private static final int GAMES_PER_WALL = 6;

  /** 三人打ちの平均の着順。 */
  private static final double MEAN_RANK = 2.0;

  @DataProvider
  public Object[][] shapes() {
    return new Object[][] {{"v2", PrismShape.standard(2)}, {"v3", ShapeArm.V3.shape(ShapeArm.V3_DEPTH)}};
  }

  @Test(groups = "rocm", dataProvider = "shapes", timeOut = 600_000)
  public void theSameVersionTiesOnEveryWall(String label, PrismShape shape) throws Exception {
    Path root = Files.createTempDirectory("prism-3p-hip-duel");
    try {
      Path snapshot = root.resolve("1");
      try (NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        Snapshot.write(snapshot, PerturbedNetwork.of(manager, InputSchema.FORMAT, shape, 71), PolicyParams.of(1));
      }
      InferenceSettings settings = ThreePlayerSettings.of().inference();
      try (var runtime = new NetworkRuntime(settings.encoders(), settings.slots(), settings.readyBatches());
          NetworkPolicy policy =
              Snapshot.open(snapshot, List.of(Device.gpu(0)), runtime, 64, Precision.BFLOAT16, InferencePath.HIP)) {
        Assert.assertEquals(policy.inference(), InferencePath.HIP, label + ": the three-player version opens on HIP");
      }
      for (String draws : List.of("SEAT_STREAMS", "WALL_ORDER")) {
        String table = draws.equals("SEAT_STREAMS") ? "streams" : "order";
        Path out = root.resolve(table);
        Assert.assertEquals(Main.run(arguments(snapshot, out, table, draws)), Exit.OK.code(), label + " " + table);
        Path job = out.resolve("jobs").resolve(RunDirectory.jobName(Evaluate.MODEL_VERSION, table));
        DuelResult result = Jobs.result(job, DuelResult.class);
        Map<String, Double> deltas = new TreeMap<>();
        List<String> lines = Files.readAllLines(job.resolve(JobOutputs.GAMES));
        for (String line : lines) {
          // 列: 牌山の番号・割り当て・席ごとのスロット・席ごとの点 3 つ・候補の平均の着順・…
          String[] columns = line.split("\t");
          deltas.merge(columns[0], Double.parseDouble(columns[6]) - MEAN_RANK, Double::sum);
        }
        List<String> nonzero = new ArrayList<>();
        deltas.forEach(
            (wall, delta) -> {
              if (Math.abs(delta) > 1e-9) {
                nonzero.add(wall + "=" + delta);
              }
            });
        log.info(
            "PRISM-3P-HIP-DUEL {} {}: {} games on {} walls, identical {}, walls with a nonzero delta {}",
            label,
            table,
            result.games(),
            deltas.size(),
            result.identical(),
            nonzero.size());
        Assert.assertEquals(result.games(), WALLS * GAMES_PER_WALL, label + " " + table);
        Assert.assertEquals(lines.size(), WALLS * GAMES_PER_WALL, label + " " + table);
        Assert.assertEquals(deltas.size(), WALLS, label + " " + table);
        Assert.assertTrue(nonzero.isEmpty(), label + " " + table + " walls with a nonzero delta: " + nonzero);
        Assert.assertEquals(result.identical(), 1.0, label + " " + table);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** 自分自身を相手にする卓を、GPU 0 の NATIVE（HIP）の推論で打つための evaluate の引数。 */
  private static String[] arguments(Path snapshot, Path out, String table, String draws) {
    String prefix = "evaluation.tables." + table + ".";
    List<String> arguments =
        new ArrayList<>(
            List.of(
                "evaluate", "--model", snapshot.toString(), "--table", table, "--out", out.toString()));
    for (String assignment :
        List.of(
            "run.variant=TENHOU_3P",
            "run.seed=1",
            "inference.devices=[\"gpu0\"]",
            "inference.precision=BFLOAT16",
            "inference.kernels=NATIVE",
            "inference.maxBatch=2048",
            prefix + "opponent=prism:" + snapshot,
            prefix + "seating={kind = \"EVERY_PLACEMENT\", splits = [\"1v2\", \"2v1\"]}",
            prefix + "draws=" + draws,
            prefix + "wallSet=FIXED",
            prefix + "walls=" + WALLS,
            prefix + "stop=FIXED",
            prefix + "score=RANK",
            prefix + "every=1",
            prefix + "from=1")) {
      arguments.addAll(List.of("--set", assignment));
    }
    return arguments.toArray(String[]::new);
  }
}
