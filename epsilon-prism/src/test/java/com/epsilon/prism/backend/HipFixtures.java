package com.epsilon.prism.backend;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.nn.Parameter;
import ai.djl.util.Pair;
import com.epsilon.core.Action;
import com.epsilon.core.PublicObservation;
import com.epsilon.prism.DecisionWalk;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputWriter;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.policy.BatchSamples;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.ByteBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Random;
import java.util.function.IntFunction;
import java.util.stream.Stream;

/** HIP の試験の材料: 乱数で動かした網と、エンジンで打った半荘の判断から Tokenizer で作った塊。 */
public final class HipFixtures {

  /** 公開の出来事が 100 を超える局（どちらも 103）を含む CALLS の半荘の種。 */
  private static final long[] LONG_GAMES = {8_285, 8_441};

  /** 長い行の公開の出来事の下限（T = 39 + 出来事 > 128 で、K2 の波の 2 つ目の問い合わせの塊が LDS の q を通る）。 */
  private static final int LONG_ROW_EVENTS = 90;

  /** 速さの試験が回す本番の塊のディレクトリの系の性質（{@link BatchSamples} が書いた *.batch。なければ模擬の半荘の行）。 */
  static final String SPEED_BATCHES = "prism.speedBatches";

  /**
   * 速さの試験の 1 塊の行。
   *
   * @param name 名前（模擬なら synthetic、本番の塊ならファイルの名前）
   * @param rows O の語の列
   */
  record SpeedBatch(String name, List<long[]> rows) {}

  private HipFixtures() {}

  /**
   * 標準の幅の網を作り、全てのパラメーターを動かす（0 で始める層も効かせ、関係の偏り・符号の表は大きくして式の取り違えを見えるようにする）。
   *
   * @param manager パラメーターの管理元
   * @param depth 背骨の層の数
   * @param seed 乱数の種
   * @return 網
   */
  public static PrismNetwork network(NDManager manager, int depth, long seed) {
    manager.getEngine().setRandomSeed((int) seed);
    PrismNetwork network = new PrismNetwork(PrismShape.standard(depth));
    network.initialize(manager);
    Random random = new Random(seed);
    for (Pair<String, Parameter> pair : network.getParameters()) {
      String name = pair.getKey();
      NDArray array = pair.getValue().getArray();
      float[] values = array.toFloatArray();
      double sigma = sigma(name.substring(name.indexOf('_') + 1), name);
      for (int i = 0; i < values.length; i++) {
        values[i] += (float) (random.nextGaussian() * sigma);
      }
      array.set(values);
    }
    return network;
  }

  /** パラメーターごとの揺らしの大きさ。 */
  private static double sigma(String local, String name) {
    if (local.equals("relation")) {
      return 1.0;
    }
    if (local.equals("codeKey") || local.equals("codeValue")) {
      return 0.3;
    }
    boolean head = name.contains("discardValue") || name.contains("gates");
    if (head
        && (local.equals("output")
            || local.equals("exact")
            || local.equals("kindOffset")
            || local.equals("exactProjection")
            || local.equals("outputBias")
            || local.equals("exactOutput"))) {
      return 0.1;
    }
    return 0.02;
  }

  /**
   * 半荘を打ち、網に問う判断の記録 O を集める。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param policy 打ち方
   * @param limit 集める判断の上限
   * @return O の語の列（判断ごと）
   */
  static List<long[]> observations(long seed, int games, DecisionWalk.Policy policy, int limit) {
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(1 << 12);
    Observation read = new Observation();
    List<long[]> rows = new ArrayList<>();
    new DecisionWalk(policy)
        .run(
            seed,
            games,
            (engine, point, round) -> {
              if (rows.size() >= limit) {
                return;
              }
              PublicObservation o = engine.observation(point.player());
              List<Action> legal = point.legalActions();
              arena.clear();
              int base = writer.capture(o, point.kind(), legal, arena);
              int longs = read.bind(arena.words(), base).longs();
              rows.add(Arrays.copyOfRange(arena.words(), base, base + longs));
            });
    return rows;
  }

  /**
   * 価値グラフの全ての作り方（打牌・立直・応答の鳴きとロン・手番の槓とツモと九種・PASS）が出る行を集める。GREEDY と CALLS の半荘を
   * 交互に打ち、まだ数の足りない役割を含む判断を先に取り、残りを普通の判断で埋める。
   *
   * @param rows 行の数
   * @param seed 最初の半荘の種
   * @return O の語の列
   */
  public static List<long[]> mixedRows(int rows, long seed) {
    EnumMap<ActionRole, Integer> quota = new EnumMap<>(ActionRole.class);
    quota.put(ActionRole.RIICHI, 3);
    quota.put(ActionRole.TSUMO, 2);
    quota.put(ActionRole.RON, 2);
    quota.put(ActionRole.KAN, 2);
    quota.put(ActionRole.CALL, 3);
    quota.put(ActionRole.KYUSHU, 1);
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(1 << 12);
    Observation read = new Observation();
    List<long[]> special = new ArrayList<>();
    List<long[]> plain = new ArrayList<>();
    int[] seen = new int[1];
    int specialLimit = rows * 2 / 3;
    for (int game = 0; game < 12 && (special.size() < specialLimit || plain.size() < rows); game++) {
      DecisionWalk.Policy policy = game % 2 == 0 ? DecisionWalk.Policy.GREEDY : DecisionWalk.Policy.CALLS;
      new DecisionWalk(policy)
          .run(
              seed + game,
              1,
              (engine, point, round) -> {
                List<Action> legal = point.legalActions();
                boolean wanted = false;
                for (Action action : legal) {
                  ActionRole role = ActionRole.of(action.type());
                  Integer left = quota.get(role);
                  wanted |= left != null && left > 0;
                }
                boolean keepSpecial = wanted && special.size() < specialLimit;
                if (!keepSpecial && plain.size() >= rows) {
                  return;
                }
                arena.clear();
                int base =
                    writer.capture(engine.observation(point.player()), point.kind(), legal, arena);
                long[] words = Arrays.copyOfRange(arena.words(), base, base + read.bind(arena.words(), base).longs());
                if (keepSpecial) {
                  special.add(words);
                  for (Action action : legal) {
                    quota.computeIfPresent(
                        ActionRole.of(action.type()), (role, left) -> Math.max(left - 1, 0));
                  }
                } else if (plain.size() < rows && seen[0]++ % 5 == 0) {
                  plain.add(words);
                }
              });
    }
    List<long[]> mixed = new ArrayList<>(special);
    for (int i = 0; mixed.size() < rows && i < plain.size(); i++) {
      mixed.add(plain.get(i));
    }
    return mixed;
  }

  /**
   * 公開の出来事の多い行: {@link #LONG_GAMES} の半荘ごとに、公開の出来事が最も多い判断を 1 つ取る。
   *
   * @return O の語の列（半荘の順）
   * @throws IllegalStateException 判断の公開の出来事が {@link #LONG_ROW_EVENTS} に届かないとき
   */
  static List<long[]> longRows() {
    ObservationWriter writer = new ObservationWriter();
    ObservationArena arena = new ObservationArena(1 << 12);
    Observation read = new Observation();
    List<long[]> rows = new ArrayList<>();
    for (long seed : LONG_GAMES) {
      long[][] longest = {null};
      int[] events = {-1};
      new DecisionWalk(DecisionWalk.Policy.CALLS)
          .run(
              seed,
              1,
              (engine, point, round) -> {
                arena.clear();
                int base =
                    writer.capture(
                        engine.observation(point.player()),
                        point.kind(),
                        point.legalActions(),
                        arena);
                Observation o = read.bind(arena.words(), base);
                if (o.publicEvents() > events[0]) {
                  events[0] = o.publicEvents();
                  longest[0] = Arrays.copyOfRange(arena.words(), base, base + o.longs());
                }
              });
      if (events[0] < LONG_ROW_EVENTS) {
        throw new IllegalStateException("game " + seed + " has at most " + events[0] + " public events");
      }
      rows.add(longest[0]);
    }
    return rows;
  }

  /** 長い行の試験の塊の行: 普通の行 2 つの後に {@link #longRows()}（長い行の出来事のトークンは塊の途中から始まる）。 */
  public static List<long[]> longMixedRows() {
    List<long[]> rows = new ArrayList<>(mixedRows(2, 900));
    rows.addAll(longRows());
    return rows;
  }

  /**
   * GREEDY と CALLS の半荘から交互に集めた行。
   *
   * @param count 行の数
   * @param seed 最初の半荘の種
   * @return O の語の列
   */
  static List<long[]> simulatedRows(int count, long seed) {
    List<long[]> greedy = observations(seed, 1 + count / 600, DecisionWalk.Policy.GREEDY, count);
    List<long[]> calls = observations(seed + 1, 1 + count / 600, DecisionWalk.Policy.CALLS, count);
    List<long[]> rows = new ArrayList<>();
    for (int i = 0; rows.size() < count && (i < greedy.size() || i < calls.size()); i++) {
      if (i < greedy.size()) {
        rows.add(greedy.get(i));
      }
      if (i < calls.size() && rows.size() < count) {
        rows.add(calls.get(i));
      }
    }
    return rows;
  }

  /**
   * 速さの試験の塊。系の性質 {@value #SPEED_BATCHES} があればそのディレクトリの本番の塊（名前の順）、なければ模擬の半荘の行の塊 1 つ。
   *
   * @param count 模擬の塊の行の数
   * @param seed 模擬の塊の最初の半荘の種
   * @return 塊（1 つ以上）
   * @throws IllegalStateException ディレクトリに塊がないとき
   */
  static List<SpeedBatch> speedBatches(int count, long seed) {
    String directory = System.getProperty(SPEED_BATCHES);
    if (directory == null || directory.isBlank()) {
      return List.of(new SpeedBatch("synthetic", simulatedRows(count, seed)));
    }
    try (Stream<Path> files = Files.list(Path.of(directory))) {
      List<SpeedBatch> batches = new ArrayList<>();
      for (Path file :
          files.filter(f -> f.getFileName().toString().endsWith(".batch")).sorted().toList()) {
        batches.add(new SpeedBatch(file.getFileName().toString(), BatchSamples.read(file)));
      }
      if (batches.isEmpty()) {
        throw new IllegalStateException("no *.batch files in " + directory);
      }
      return batches;
    } catch (IOException failure) {
      throw new UncheckedIOException(failure);
    }
  }

  /**
   * O の列を 1 つの塊に符号化する（blob は直の ByteBuffer）。
   *
   * @param rows O の語の列
   * @return 封じた塊
   */
  public static HostBatch encode(List<long[]> rows) {
    return encode(rows, ByteBuffer::allocateDirect);
  }

  /**
   * O の列を 1 つの塊に符号化する。
   *
   * @param rows O の語の列
   * @param memory blob の置き場を取る口（バイト数を受ける）
   * @return 封じた塊
   */
  static HostBatch encode(List<long[]> rows, IntFunction<ByteBuffer> memory) {
    int n = rows.size();
    BatchPlan.Capacity capacity =
        new BatchPlan.Capacity(
            n,
            Math.max(n * Limits.MAX_EVENT_TOKENS, Limits.MAX_EVENT_TOKENS),
            Math.max(n * Limits.MAX_TRANSITIONS, Limits.MAX_TRANSITIONS),
            Math.max(n * Limits.MAX_LEGAL_ACTIONS, Limits.MAX_LEGAL_ACTIONS));
    HostBatch batch = new HostBatch(capacity, memory.apply(BatchPlan.maxBytes(capacity)));
    BatchPlan plan = batch.plan();
    plan.clear();
    Observation read = new Observation();
    for (long[] words : rows) {
      if (plan.add(read.bind(words, 0)) < 0) {
        throw new IllegalStateException("batch capacity is too small");
      }
    }
    plan.seal();
    Tokenizer tokenizer = new Tokenizer();
    InputWriter writer = new InputWriter();
    for (int row = 0; row < n; row++) {
      tokenizer.write(read.bind(rows.get(row), 0), writer.bind(batch, row));
    }
    return batch;
  }

  /** 行 r の最初の手の番号（塊の手の順）。 */
  static int firstAction(HostBatch batch, int row) {
    return batch.plan().firstAction(row);
  }

  static int actions(HostBatch batch, int row) {
    return batch.plan().actions(row);
  }
}
