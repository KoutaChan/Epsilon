package com.epsilon.prism.policy;

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
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.prism.Limits;
import com.epsilon.prism.config.settings.EpsilonSettings;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.learn.Fit;
import com.epsilon.prism.learn.FitPlan;
import com.epsilon.prism.learn.FitResult;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.NetworkInputs;
import com.epsilon.prism.model.Pass;
import com.epsilon.prism.model.Precision;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.play.SnapshotScorer;
import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.workflow.EvaluateWorkflow;
import com.epsilon.workflow.WorkflowArguments;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.SplittableRandom;
import java.util.function.Consumer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.DataProvider;
import org.testng.annotations.Test;

/**
 * 網の版の方策（CPU、FP32、小さな網）。
 *
 * <ul>
 *   <li>乱数で動かした網の版を {@link Snapshot#open} で開くと網の方策になり、種の違う半荘を並べた受け付けの s が、同じ O を
 *       同じ並びの 1 つの塊にして直に順伝播した s とビットで同じ。詰めたトークンの上限で塊を分けても（S の順に並べ替える）、問い合わせごとの s は
 *       変わらない（塊の組み方の丸めの差だけ。行どうしが混ざらない＝入力の書き手と網が関係の属性の並びの約束を共有している）。p0 は Nano の符号化の
 *       遷移特徴（{@link R0Features}）から {@link Prior} で求めた参照とビットで同じで、σ は貪欲の one-hot。受付の区分は 1 つで、受け付けには鳴きの
 *       行も混ざる（{@link NetworkPolicy#admissionKey}）。
 *   <li>白紙の版（s ≡ 0。p0 の最大で打つ）で evaluate が数局を打ち終え、半荘ごとの結果を書く。
 * </ul>
 */
public class NetworkPolicyTest {
  private static final Logger log = LoggerFactory.getLogger(NetworkPolicyTest.class);

  private static final PrismShape SHAPE = new PrismShape(64, 128, 64, 2, false);
  private static final int GAMES = 6;
  private static final int BATCHES = 60;

  @DataProvider
  public Object[][] frontierArms() {
    return new Object[][] {{false}, {true}};
  }

  @Test(timeOut = 900_000, dataProvider = "frontierArms")
  public void servedScoresMatchTheNetwork(boolean frontier) throws Exception {
    Path root = Files.createTempDirectory("prism-policy");
    try {
      Path snapshot = root.resolve("1");
      write(snapshot, SHAPE.withFrontier(frontier), true);
      PrismConfig config = PrismConfig.fromSettings(EpsilonSettings.defaults());
      PriorSettings prior = config.priorSettings();
      List<Device> cpu = List.of(Device.cpu());
      int smallest = BatchPlan.FIXED_TOKENS + Limits.MAX_EVENT_TOKENS;
      var nano = new DecisionBatchBuilder.EncodingWorkspacePool(1);
      int[] counts = new int[4];
      try (PolicyService served =
              Snapshot.open(
                  snapshot,
                  cpu,
                  null,
                  config.inferBatch(),
                  prior,
                  config.inference());
          NetworkPolicy split =
              NetworkPolicy.eager(
                  PolicyParams.of(1),
                  snapshot.resolve(Snapshot.MODEL),
                  cpu,
                  null,
                  config.inferBatch(),
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
                copies.add(copy(request));
              }
              ask(served, requests);
              ask(split, copies);
              float[][] expected = direct.scores(requests, manager);
              for (int row = 0; row < requests.size(); row++) {
                PolicyRequest request = requests.get(row);
                List<Action> legal = request.legal();
                int n = legal.size();
                String where = "row " + row + " " + legal;
                for (int a = 0; a < n; a++) {
                  assertEquals(
                      Float.floatToRawIntBits(request.scores[a]),
                      Float.floatToRawIntBits(expected[row][a]),
                      "service s " + a + " " + where);
                  assertEquals(copies.get(row).scores[a], request.scores[a], 1e-5f, "split s " + where);
                }
                float[] reference = nanoPrior(request, nano, prior);
                for (int a = 0; a < n; a++) {
                  assertEquals(
                      Float.floatToRawIntBits(request.prior[a]),
                      Float.floatToRawIntBits(reference[a]),
                      "p0 " + a + " " + where);
                }
                assertEquals(request.sigma[Greedy.argmax(request.scores, request.prior, n)], 1f, where);
                assertEquals(served.admissionKey(request), Capacity.SINGLE_TRANSITION, where);
                counts[1] += n;
                if (request.capacity() == Capacity.MULTIPLE_TRANSITIONS) {
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
      Path perGame = root.resolve("per-game.tsv");
      var settings =
          EpsilonSettings.of(
              Map.of(
                  "epsilon.prism.collectDevices", "cpu",
                  "epsilon.prism.maxActiveGames", "4",
                  "epsilon.prism.advanceWorkers", "2",
                  "epsilon.prism.encoders", "2",
                  "epsilon.prism.batchWaitMicros", "1000"));
      Arena.Result result =
          new EvaluateWorkflow()
              .execute(
                  new WorkflowArguments(
                      "evaluate",
                      new String[] {
                        "--snapshot", snapshot.toString(),
                        "--games", "4",
                        "--opponent", "heuristic",
                        "--per-game", perGame.toString()
                      },
                      settings));
      assertEquals(result.games(), 4);
      List<String> lines = Files.readAllLines(perGame);
      assertEquals(lines.size(), 4);
      for (String line : lines) {
        int rank = Integer.parseInt(line.split("\t")[1]);
        assertTrue(rank >= 1 && rank <= 4, line);
      }
    } finally {
      Snapshot.deleteTree(root);
    }
  }

  /**
   * 白紙（perturbed なら全てのパラメーターに乱数を足した）網の版を書く（版の番号は 1）。
   *
   * @param snapshot 版のディレクトリ
   * @param shape 網の形
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
    try (Model model = Fit.blank(shape, Device.cpu(), plan.seed())) {
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
   * 種の違う半荘を並べて打ち、段ごとに網に問う全ての判断（合法手 2 つ以上）を 1 つの受け付けとして渡す（O は段ごとに使い直す置き場に捕捉して
   * 載せる。受け口が戻るまで有効）。手は種で決まる乱数で選ぶ。
   *
   * @param games 並べる半荘の数
   * @param batches 段の数
   * @param visitor 受け付けを受ける口
   */
  static void walk(int games, int batches, Consumer<List<PolicyRequest>> visitor) {
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(4096);
    GameEngine[] engines = new GameEngine[games];
    GameStepResult[] steps = new GameStepResult[games];
    SplittableRandom[] randoms = new SplittableRandom[games];
    for (int game = 0; game < games; game++) {
      engines[game] = new GameEngine(77 + game);
      randoms[game] = new SplittableRandom(5 + game);
      steps[game] = engines[game].stepHanchan();
    }
    var selections = new EngineSelectionBuffer();
    for (int batch = 0; batch < batches; batch++) {
      List<PolicyRequest> requests = new ArrayList<>();
      arena.clear();
      for (int game = 0; game < games; game++) {
        while (!(steps[game] instanceof GameStepResult.AwaitingDecisions)
            && !(steps[game] instanceof GameStepResult.HanchanEnded)) {
          steps[game] = engines[game].stepHanchan();
        }
        if (steps[game] instanceof GameStepResult.AwaitingDecisions awaiting) {
          for (EngineDecisionPoint point : awaiting.decisions()) {
            List<Action> legal = point.legalActions();
            if (legal.size() > 1) {
              PolicyRequest request = retained(engines[game].observation(point.player()), legal);
              request.capture(writer, arena);
              requests.add(request);
            }
          }
        }
      }
      if (!requests.isEmpty()) {
        visitor.accept(requests);
      }
      for (int game = 0; game < games; game++) {
        if (steps[game] instanceof GameStepResult.AwaitingDecisions awaiting) {
          selections.clear();
          for (EngineDecisionPoint point : awaiting.decisions()) {
            List<Action> legal = point.legalActions();
            selections.add(point.id(), legal.get(randoms[game].nextInt(legal.size())));
          }
          steps[game] = engines[game].commitDecisions(selections);
        }
      }
    }
  }

  /** retain の問い合わせ（p0 を全ての問い合わせに書かせる）。 */
  static PolicyRequest retained(PublicObservation observation, List<Action> legal) {
    return copy(SnapshotScorer.request(observation, legal));
  }

  /** 同じ観測・合法手・捕捉した O の新しい retain の問い合わせ（答えの欄は空）。 */
  static PolicyRequest copy(PolicyRequest request) {
    return request.copy(true);
  }

  /** 1 回の受け付けを方策に問い、答えを待つ。 */
  static void ask(PolicyService policy, List<PolicyRequest> requests) {
    var attempt = policy.tryAcquireIngress();
    assertTrue(attempt.acquired());
    policy.submit(requests, attempt.ingress()).join();
  }

  /** 参照の p0: Nano の符号化の遷移特徴（{@link R0Features}）から読んだ打牌の後の向聴と受け入れで {@link Prior} を通す。 */
  private static float[] nanoPrior(
      PolicyRequest request, DecisionBatchBuilder.EncodingWorkspacePool workspaces, PriorSettings prior) {
    List<Action> legal = request.legal();
    DecisionBatchBuilder builder =
        DecisionBatchBuilder.inference(1, DecisionBatchBuilder.selectInferenceBucket(legal), workspaces);
    try (DecisionBatchBuilder.EncodingSession session = builder.openEncoding()) {
      session.encodeObservedInferenceRow(0, request.observation(), legal);
    }
    DecisionHostBatch host = builder.buildEncodedInferenceRows();
    int n = legal.size();
    float[] shanten = new float[n], ukeire = new float[n];
    for (int a = 0; a < n; a++) {
      if (ActionGroups.of(legal.get(a).type()) != ActionGroups.DISCARD) continue;
      shanten[a] = R0Features.shanten(host, 0, a);
      ukeire[a] = R0Features.ukeire(host, 0, a);
    }
    return Prior.compute(legal, shanten, ukeire, prior);
  }

  /** 方策を通さずに、同じ O を問い合わせの順の 1 つの塊にして直に順伝播する。 */
  static final class Direct {
    private final PrismNetwork network;
    private final Precision precision;
    private final RequestBatch batch;
    private boolean nonzero;

    Direct(PrismNetwork network, Precision precision) {
      this.network = network;
      this.precision = precision;
      batch = new RequestBatch(network.shape().frontier());
    }

    float[][] scores(List<PolicyRequest> requests, NDManager manager) {
      HostInputs inputs = batch.encode(requests);
      try (NDManager scope = manager.newSubManager()) {
        NDArray scores =
            network
                .forward(
                    new ParameterStore(scope, false),
                    NetworkInputs.of(scope, inputs.input(), inputs.frontier()),
                    Pass.inference(precision))
                .scores()
                .toType(DataType.FLOAT32, false);
        int width = Math.toIntExact(scores.getShape().get(1));
        float[] flat = scores.toFloatArray();
        float[][] result = new float[requests.size()][];
        for (int row = 0; row < requests.size(); row++) {
          int n = requests.get(row).legal().size();
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
