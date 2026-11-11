package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;
import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertTrue;

import ai.djl.Device;
import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Parameter;
import ai.djl.training.ParameterStore;
import ai.djl.util.Pair;
import com.epsilon.agent.app.Exit;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.run.JobOutputs;
import com.epsilon.agent.run.Jobs;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.prism.Limits;
import com.epsilon.prism.Main;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.TestSettings;
import com.epsilon.prism.app.Evaluate;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism4p.input.InputSchema;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Random;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * ネットワークのバージョンの方策のテスト（CPU、FP32、小さなネットワーク）。
 *
 * <ul>
 *   <li>乱数を足したネットワークのバージョンを {@link Snapshot#open} で開くと、ネットワークの方策になる。乱数の種の違う半荘を並べた受付の
 *       s は、同じ O を同じ順に 1 つのバッチにして直に順伝播した s とビットで同じ。トークンの上限でバッチを分けても（S の順に並べ替える）、
 *       要求ごとの s はバッチの組み方による丸めの差しか変わらない。つまり行どうしが混ざらず、入力のライターとネットワークが関係の属性の
 *       並び方をそろえている。p0 は agent の事実から求めた参照（{@link PriorReference}）とビットで同じで、σ は貪欲に選んだ手の one-hot。
 *       受付の区分は 1 つで、受付には鳴きの行も混ざる。
 *   <li>初期化したままのバージョン（s ≡ 0。p0 が最大の手を打つ）で、evaluate がルールベースの打ち手と数半荘を打ち終え、半荘ごとの結果を
 *       書く。同じ --out の同じ卓は打たずに同じ結果を返し、定義の違う卓は拒む。
 * </ul>
 */
public class NetworkPolicyTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, null);
  private static final int GAMES = 6;
  private static final int BATCHES = 60;

  /** CPU の方策の行の上限（推論のセクションの既定値）。 */
  private static final int MAX_BATCH = 8;

  @Test(timeOut = 900_000)
  public void servedScoresMatchTheNetwork() throws Exception {
    Path root = Files.createTempDirectory("prism-policy");
    try {
      Path snapshot = root.resolve("1");
      write(snapshot, SHAPE, true);
      PriorSettings prior = PriorSettings.STANDARD;
      List<Device> cpu = List.of(Device.cpu());
      int smallest = InputSchema.FORMAT.fixedTokens() + Limits.MAX_EVENT_TOKENS;
      PriorReference reference = new PriorReference();
      Observation reader = new Observation(TENHOU_4P);
      int[] counts = new int[4];
      try (PolicyService served =
              Snapshot.open(snapshot, cpu, null, MAX_BATCH, Precision.FLOAT32, InferencePath.EAGER);
          NetworkPolicy split =
              NetworkPolicy.eager(
                  PolicyParams.of(1),
                  snapshot.resolve(Snapshot.MODEL),
                  cpu,
                  null,
                  MAX_BATCH,
                  Precision.FLOAT32,
                  prior,
                  smallest);
          NDManager manager = NDManager.newBaseManager(Device.cpu())) {
        assertTrue(served instanceof NetworkPolicy, served.getClass().getName());
        assertEquals(((NetworkPolicy) served).inference(), InferencePath.EAGER);
        Direct direct =
            new Direct(
                PrismNetwork.load(snapshot.resolve(Snapshot.MODEL), manager), Precision.FLOAT32);
        walk(
            GAMES,
            BATCHES,
            requests -> {
              List<PolicyRequest> copies = new ArrayList<>(requests.size());
              for (PolicyRequest request : requests) {
                copies.add(PolicyRequests.copy(request));
              }
              ask(served, requests);
              ask(split, copies);
              float[][] expected = direct.scores(requests, manager);
              for (int row = 0; row < requests.size(); row++) {
                PolicyRequest request = requests.get(row);
                int n = request.legalCount();
                String where = "row " + row + " " + Arrays.toString(request.legal());
                for (int a = 0; a < n; a++) {
                  assertEquals(
                      Float.floatToRawIntBits(request.scores[a]),
                      Float.floatToRawIntBits(expected[row][a]),
                      "service s " + a + " " + where);
                  assertEquals(copies.get(row).scores[a], request.scores[a], 1e-5f, "split s " + where);
                }
                Observation o = request.bind(reader);
                float[] expectedPrior = reference.prior(o, prior);
                for (int a = 0; a < n; a++) {
                  assertEquals(
                      Float.floatToRawIntBits(request.prior[a]),
                      Float.floatToRawIntBits(expectedPrior[a]),
                      "p0 " + a + " " + where);
                }
                assertEquals(request.sigma[Greedy.argmax(request.scores, request.prior, n)], 1f, where);
                assertEquals(served.admission(o), 0, where);
                counts[1] += n;
                if (Capacity.of(o) == Capacity.MULTIPLE_TRANSITIONS) {
                  counts[3]++;
                }
              }
              counts[0]++;
              counts[2] = Math.max(counts[2], requests.size());
            });
        NetworkPolicy.Throughput whole = ((NetworkPolicy) served).throughput();
        NetworkPolicy.Throughput parts = split.throughput();
        log.info(
            "PRISM-POLICY batches={} actions={} widest={} chunks {} -> {} decisions={}",
            counts[0],
            counts[1],
            counts[2],
            whole.chunks(),
            parts.chunks(),
            whole.decisions());
        assertEquals(whole.chunks(), whole.batches(), "one chunk per small batch");
        assertTrue(parts.chunks() > parts.batches(), "the small budget splits batches");
        assertTrue(counts[2] > 3, "batches must hold several rows: " + counts[2]);
        assertTrue(counts[3] > 0, "batches must mix call rows into the one admission key");
        assertTrue(direct.nonzero(), "the perturbed network must give nonzero scores");
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  @Test(timeOut = 900_000)
  public void evaluateFinishesGamesWithABlankVersion() throws Exception {
    Path root = Files.createTempDirectory("prism-evaluate");
    try {
      Path snapshot = root.resolve("1");
      write(snapshot, SHAPE, false);
      Path out = root.resolve("out");
      assertEquals(Main.run(evaluate(snapshot, out, 4)), Exit.OK.code());
      Path job = out.resolve("jobs").resolve(RunDirectory.jobName(Evaluate.MODEL_VERSION, "t"));
      List<String> lines = Files.readAllLines(job.resolve(JobOutputs.GAMES));
      assertEquals(lines.size(), 4);
      for (String line : lines) {
        // 列: 牌山の番号・割り当て・席ごとのスロット・席ごとの点 4 つ・候補の平均着順
        double rank = Double.parseDouble(line.split("\t")[7]);
        assertTrue(rank >= 1 && rank <= 4, line);
      }
      // 同じ --out と卓でもう一度回す。済んだジョブは打たずに同じ結果を返し、卓の値が違えば拒む
      String games = Files.readString(job.resolve(JobOutputs.GAMES));
      String result = Files.readString(job.resolve(Jobs.RESULT));
      assertEquals(Main.run(evaluate(snapshot, out, 4)), Exit.OK.code());
      assertEquals(Files.readString(job.resolve(JobOutputs.GAMES)), games);
      assertEquals(Files.readString(job.resolve(Jobs.RESULT)), result);
      assertEquals(Main.run(evaluate(snapshot, out, 8)), Exit.USAGE.code());
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /** evaluate の引数を返す。卓 t で、バージョンとルールベースの打ち手が、指定した数の牌山を 1 半荘ずつ打つ。 */
  private static String[] evaluate(Path snapshot, Path out, int walls) {
    List<String> arguments =
        new ArrayList<>(
            List.of(
                "evaluate", "--model", snapshot.toString(), "--table", "t", "--out", out.toString(), "--set",
                "run.variant=TENHOU_4P", "--set", "run.seed=1", "--set", "table.activeGames=4", "--set",
                "table.workers=2", "--set", "inference.encoders=2", "--set", "table.batchWaitMicros=1000"));
    for (String assignment : TestSettings.table("t", EvaluationTable.HEURISTIC, walls, 1, 1)) {
      arguments.addAll(List.of("--set", assignment));
    }
    return arguments.toArray(String[]::new);
  }

  /**
   * 初期化したままのネットワークを、バージョン番号 1 のバージョンとして書く。乱数を足すときは、全てのパラメーターに足す。
   *
   * @param snapshot バージョンのディレクトリ
   * @param shape ネットワークの形
   * @param perturbed 乱数を足すなら true
   */
  static void write(Path snapshot, PrismShape shape, boolean perturbed) throws Exception {
    FitPlan plan =
        new FitPlan(
            new FitSettings(1, 4096, 3e-3, 3e-4, 1f, 0.01, 0.02),
            shape,
            AuxiliaryLoss.standard(),
            FitPlan.TOKEN_BUDGET,
            7);
    try (Model model = Fit.blank(InputSchema.FORMAT, shape, Device.cpu(), plan.seed())) {
      PrismNetwork network = (PrismNetwork) model.getBlock();
      if (perturbed) {
        Random random = new Random(13);
        for (Pair<String, Parameter> pair : network.getParameters()) {
          NDArray array = pair.getValue().getArray();
          float[] values = array.toFloatArray();
          for (int index = 0; index < values.length; index++) {
            values[index] += (float) (random.nextGaussian() * 0.05);
          }
          array.set(values);
        }
      }
      Snapshot.write(
          snapshot,
          network,
          PolicyParams.of(1),
          new Snapshot.FitRecord(snapshot.toString(), 1, 1, 0, plan, FitResult.empty(0)));
    }
  }

  /**
   * 乱数の種の違う半荘を打ち、各半荘の判断（合法手が 2 つ以上のもの）を半荘の中の順に 1 つずつ取って、1 回の受付として渡す。O は
   * 受付ごとに新しいバッチに置く。鳴きの行を混ぜるため、半荘は鳴き・槓を選ぶ打ち方で打つ。
   *
   * @param games 並べる半荘の数
   * @param batches 受付の回数
   * @param visitor 受付の要求を受け取るもの
   */
  static void walk(int games, int batches, Consumer<List<PolicyRequest>> visitor) {
    List<List<long[]>> perGame = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      perGame.add(PolicyRequests.observations(77 + game, TableWalk.Policy.CALLS, batches));
    }
    for (int batch = 0; batch < batches; batch++) {
      List<long[]> rows = new ArrayList<>(games);
      for (List<long[]> game : perGame) {
        if (batch < game.size()) {
          rows.add(game.get(batch));
        }
      }
      if (!rows.isEmpty()) {
        visitor.accept(PolicyRequests.requests(rows));
      }
    }
  }

  /** 1 回の受付の要求を方策に送り、答えを待つ。 */
  static void ask(PolicyService policy, List<PolicyRequest> requests) {
    var attempt = policy.tryAcquireIngress(0);
    assertTrue(attempt.acquired());
    policy.submit(requests, attempt.ingress()).join();
  }

  /** 方策を通さずに、同じ O を要求の順に 1 つのバッチにして直に順伝播する。 */
  static final class Direct {
    private final PrismNetwork network;
    private final Precision precision;
    private final RequestBatch batch;
    private boolean nonzero;

    Direct(PrismNetwork network, Precision precision) {
      this.network = network;
      this.precision = precision;
      batch = new RequestBatch();
    }

    float[][] scores(List<PolicyRequest> requests, NDManager manager) {
      HostInputs inputs = batch.encode(requests);
      try (NDManager scope = manager.newSubManager()) {
        NDArray scores =
            network
                .forward(
                    new ParameterStore(scope, false),
                    NetworkInputs.of(scope, PinnedUploads.on(scope), inputs.input()),
                    Pass.inference(precision))
                .scores()
                .toType(DataType.FLOAT32, false);
        int width = Math.toIntExact(scores.getShape().get(1));
        float[] flat = scores.toFloatArray();
        float[][] result = new float[requests.size()][];
        for (int row = 0; row < requests.size(); row++) {
          int n = requests.get(row).legalCount();
          result[row] = new float[n];
          System.arraycopy(flat, row * width, result[row], 0, n);
          for (float value : result[row]) {
            nonzero |= value != 0f;
          }
        }
        return result;
      }
    }

    /** 0 でない s を出したか。 */
    boolean nonzero() {
      return nonzero;
    }
  }
}
