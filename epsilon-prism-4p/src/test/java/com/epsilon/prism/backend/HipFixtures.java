package com.epsilon.prism.backend;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import ai.djl.ndarray.NDManager;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.prism.Limits;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.BatchPlan;
import com.epsilon.prism.input.HostBatch;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.model.ShapeArm;
import com.epsilon.prism4p.input.InputSchema;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import java.util.function.IntFunction;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/** HIP のテストの材料。パラメーターを乱数でずらしたネットワークと、エンジンで打った半荘の判断から入力のライターで作ったバッチを用意する。 */
public final class HipFixtures {

  private static final Logger log = LoggerFactory.getLogger(HipFixtures.class);

  /** 長い行を探すときに打つ CALLS の半荘の、最初の乱数の種。 */
  private static final long LONG_SEARCH_FROM = 8_000;

  /** 長い行を探す半荘の数の上限。 */
  private static final int LONG_SEARCH_GAMES = 400;

  /** 長い行とみなす公開イベントの数の下限。T = 39 + イベント数 > 128 になるので、K2・K4 は 10 ウェーブ構成のカーネルを通る。 */
  private static final int LONG_ROW_EVENTS = 90;

  /** {@link #longRows(int)} が 1 つの半荘から取る判断の上限。 */
  private static final int LONG_PER_GAME = 12;

  /** {@link #longRows(int)} が探す半荘の数の上限。 */
  private static final int LONG_ROWS_SEARCH_GAMES = 3_000;

  private HipFixtures() {}

  /** 通しのテストで比べる形のアーム。v2 の構造の深さ 2（本番）と 3、v3 の構造の 3 つ。 */
  public enum Arm {
    V2_DEPTH_2(ShapeArm.V2, 2),
    V2_DEPTH_3(ShapeArm.V2, 3),
    V3(ShapeArm.V3, ShapeArm.V3_DEPTH),
    V3_ONE_STAGE(ShapeArm.V3_ONE_STAGE, ShapeArm.V3_DEPTH),
    V3_CONTEXT_512(ShapeArm.V3_CONTEXT_512, ShapeArm.V3_DEPTH);

    /** テストで使うアームを選ぶシステムプロパティ（名前をカンマで区切る。無ければ全てのアーム）。 */
    static final String PROPERTY = "prism.arms";

    private final ShapeArm arm;
    private final int depth;

    Arm(ShapeArm arm, int depth) {
      this.arm = arm;
      this.depth = depth;
    }

    /** アームのネットワークの形。 */
    public PrismShape shape() {
      return arm.shape(depth);
    }

    /** ログの行に書く名前（v2-depth-2 など）。 */
    public String label() {
      return name().toLowerCase(Locale.ROOT).replace('_', '-');
    }

    /**
     * システムプロパティ {@value #PROPERTY} で選んだアームを返す。
     *
     * @return アーム（指定が無ければ宣言の順、指定があれば書いた順）
     * @throws IllegalArgumentException 知らない名前があるとき
     */
    public static List<Arm> selected() {
      String given = System.getProperty(PROPERTY);
      if (given == null || given.isBlank()) {
        return List.of(values());
      }
      List<Arm> arms = new ArrayList<>();
      for (String name : given.split(",")) {
        arms.add(valueOf(name.strip().toUpperCase(Locale.ROOT).replace('-', '_')));
      }
      return arms;
    }
  }

  /**
   * 標準の幅で v2 の構造のネットワークを作り、全てのパラメーターを乱数でずらす（{@link #network(NDManager, PrismShape, long)}）。
   *
   * @param manager パラメーターのマネージャー
   * @param depth バックボーンの層の数
   * @param seed 乱数の種
   * @return ネットワーク
   */
  public static PrismNetwork network(NDManager manager, int depth, long seed) {
    return network(manager, PrismShape.standard(depth), seed);
  }

  /**
   * 四人打ちのネットワークを作り、全てのパラメーターを乱数でずらす（{@link PerturbedNetwork}）。
   *
   * @param manager パラメーターのマネージャー
   * @param shape 形
   * @param seed 乱数の種
   * @return ネットワーク
   */
  public static PrismNetwork network(NDManager manager, PrismShape shape, long seed) {
    return PerturbedNetwork.of(manager, InputSchema.FORMAT, shape, seed);
  }

  /** アームのネットワークと、価値グラフの全ての作り方が現れる {@link GpuCase#ROWS} 行のバッチから、GPU のテストの材料を作る。 */
  static GpuCase gpuCase(Arm arm, long seed) {
    return gpuCase(arm.label(), arm.shape(), seed, HipReferenceTest.mixedRows(GpuCase.ROWS));
  }

  /** 指定した形のネットワークと、指定した行のバッチ（{@link GpuCase#ROWS} の 4 倍まで）から、GPU のテストの材料を作る。 */
  static GpuCase gpuCase(String name, PrismShape shape, long seed, List<long[]> rows) {
    return new GpuCase(name, manager -> network(manager, shape, seed), encode(rows));
  }

  /** 判断する席の O を取り込み、そのワード列の写しを返すヘルパー（スレッドセーフでない）。 */
  private static final class Capture {
    private final ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    private final ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    private final Observation read = new Observation(TENHOU_4P);

    /** 席の O を取り込み、読み込み用の {@link Observation} に結び付けて返す。 */
    Observation observe(Engine engine, int slot) {
      Decisions decisions = engine.decisions();
      batch.clear();
      int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
      return read.bind(batch, index);
    }

    /** 直前に取り込んだ O のワード列の写しを返す。 */
    long[] words() {
      int index = batch.size() - 1;
      return Arrays.copyOfRange(batch.words(), batch.start(index), batch.start(index) + batch.longs(index));
    }
  }

  /**
   * 半荘を打ち、ネットワークに判断させる局面の記録 O を集める。
   *
   * @param seed 最初の半荘の乱数の種
   * @param games 半荘の数
   * @param policy 打ち方
   * @param limit 集める判断の上限
   * @return 判断ごとの O のワード列
   */
  static List<long[]> observations(long seed, int games, TableWalk.Policy policy, int limit) {
    Capture capture = new Capture();
    List<long[]> rows = new ArrayList<>();
    new TableWalk(TENHOU_4P, policy)
        .run(
            seed,
            games,
            (engine, slot) -> {
              if (rows.size() < limit) {
                capture.observe(engine, slot);
                rows.add(capture.words());
              }
            });
    return rows;
  }

  /**
   * 価値グラフの全ての作り方（打牌・立直・応答の鳴きとロン・手番の槓とツモと九種・PASS）が現れる行を集める。GREEDY と CALLS の半荘を
   * 交互に打つ。まだ数の足りない種類の手を含む判断を先に取り、残りを普通の判断で埋める。
   *
   * @param rows 行の数
   * @param seed 最初の半荘の乱数の種
   * @return 行ごとの O のワード列
   */
  public static List<long[]> mixedRows(int rows, long seed) {
    EnumMap<ActionRole, Integer> quota = new EnumMap<>(ActionRole.class);
    quota.put(ActionRole.RIICHI, 3);
    quota.put(ActionRole.TSUMO, 2);
    quota.put(ActionRole.RON, 2);
    quota.put(ActionRole.KAN, 2);
    quota.put(ActionRole.CALL, 3);
    quota.put(ActionRole.KYUSHU, 1);
    Capture capture = new Capture();
    List<long[]> special = new ArrayList<>();
    List<long[]> plain = new ArrayList<>();
    int[] seen = new int[1];
    int specialLimit = rows * 2 / 3;
    for (int game = 0; game < 12 && (special.size() < specialLimit || plain.size() < rows); game++) {
      TableWalk.Policy policy = game % 2 == 0 ? TableWalk.Policy.GREEDY : TableWalk.Policy.CALLS;
      new TableWalk(TENHOU_4P, policy)
          .run(
              seed + game,
              1,
              (engine, slot) -> {
                if (special.size() >= specialLimit && plain.size() >= rows) {
                  return;
                }
                Observation o = capture.observe(engine, slot);
                boolean wanted = false;
                for (int a = 0; a < o.legalActions(); a++) {
                  Integer left = quota.get(ActionRole.of(o.action(a).type()));
                  wanted |= left != null && left > 0;
                }
                if (wanted && special.size() < specialLimit) {
                  special.add(capture.words());
                  for (int a = 0; a < o.legalActions(); a++) {
                    quota.computeIfPresent(
                        ActionRole.of(o.action(a).type()), (role, left) -> Math.max(left - 1, 0));
                  }
                } else if (plain.size() < rows && seen[0]++ % 5 == 0) {
                  plain.add(capture.words());
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
   * 公開イベントの多い行を集める。{@link #LONG_SEARCH_FROM} から順に CALLS の半荘を打ち、公開イベントが {@link #LONG_ROW_EVENTS} 以上の
   * 判断を持つ半荘を 2 つ見つける。それぞれの半荘から、いちばん長い判断を 1 つ取る。見つけた種はログに書く。
   *
   * @return 半荘の順に並べた O のワード列
   * @throws IllegalStateException 探す範囲に長い半荘が 2 つないとき
   */
  static List<long[]> longRows() {
    Capture capture = new Capture();
    List<long[]> rows = new ArrayList<>();
    for (long seed = LONG_SEARCH_FROM; rows.size() < 2 && seed < LONG_SEARCH_FROM + LONG_SEARCH_GAMES; seed++) {
      long[][] longest = {null};
      int[] events = {-1};
      new TableWalk(TENHOU_4P, TableWalk.Policy.CALLS)
          .run(
              seed,
              1,
              (engine, slot) -> {
                Observation o = capture.observe(engine, slot);
                if (o.publicEvents() > events[0]) {
                  events[0] = o.publicEvents();
                  longest[0] = capture.words();
                }
              });
      if (events[0] >= LONG_ROW_EVENTS) {
        rows.add(longest[0]);
        log.info("PRISM-HIP-FIXTURE long row: CALLS game {} has {} public events", seed, events[0]);
      }
    }
    if (rows.size() < 2) {
      throw new IllegalStateException(
          "no two CALLS games with " + LONG_ROW_EVENTS + " public events from seed " + LONG_SEARCH_FROM);
    }
    return rows;
  }

  /**
   * 公開イベントが {@link #LONG_ROW_EVENTS} 以上の判断を集める。{@link #LONG_SEARCH_FROM} から順に CALLS の半荘を打ち、半荘ごとに
   * イベントの多い順に {@link #LONG_PER_GAME} 個まで取る。
   *
   * @param count 集める行の数
   * @return O のワード列（半荘の順、半荘の中はイベントの多い順）
   * @throws IllegalStateException 探す範囲に {@code count} 個ないとき
   */
  static List<long[]> longRows(int count) {
    Capture capture = new Capture();
    List<long[]> rows = new ArrayList<>();
    for (long seed = LONG_SEARCH_FROM; rows.size() < count && seed < LONG_SEARCH_FROM + LONG_ROWS_SEARCH_GAMES; seed++) {
      List<long[]> words = new ArrayList<>();
      List<Integer> events = new ArrayList<>();
      new TableWalk(TENHOU_4P, TableWalk.Policy.CALLS)
          .run(
              seed,
              1,
              (engine, slot) -> {
                Observation o = capture.observe(engine, slot);
                if (o.publicEvents() >= LONG_ROW_EVENTS) {
                  words.add(capture.words());
                  events.add(o.publicEvents());
                }
              });
      Integer[] order = new Integer[words.size()];
      for (int i = 0; i < order.length; i++) {
        order[i] = i;
      }
      Arrays.sort(order, (a, b) -> Integer.compare(events.get(b), events.get(a)));
      for (int i = 0; i < Math.min(order.length, LONG_PER_GAME) && rows.size() < count; i++) {
        rows.add(words.get(order[i]));
      }
    }
    if (rows.size() < count) {
      throw new IllegalStateException(
          "found " + rows.size() + " of " + count + " decisions with " + LONG_ROW_EVENTS + " public events");
    }
    return rows;
  }

  /** 長い行のテストで使うバッチの行。普通の行 2 つの後に {@link #longRows()} の行を続ける（長い行のイベントのトークンはバッチの途中から始まる）。 */
  public static List<long[]> longMixedRows() {
    List<long[]> rows = new ArrayList<>(mixedRows(2, 900));
    rows.addAll(longRows());
    return rows;
  }

  /**
   * GREEDY と CALLS の半荘から交互に集めた行。
   *
   * @param count 行の数
   * @param seed 最初の半荘の乱数の種
   * @return 行ごとの O のワード列
   */
  static List<long[]> simulatedRows(int count, long seed) {
    List<long[]> greedy = observations(seed, 1 + count / 600, TableWalk.Policy.GREEDY, count);
    List<long[]> calls = observations(seed + 1, 1 + count / 600, TableWalk.Policy.CALLS, count);
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
   * O の列を 1 つのバッチに符号化する（blob はダイレクトの ByteBuffer）。
   *
   * @param rows 行ごとの O のワード列
   * @return 確定したバッチ
   */
  public static HostBatch encode(List<long[]> rows) {
    return encode(rows, ByteBuffer::allocateDirect);
  }

  /**
   * O の列を 1 つのバッチに符号化する。
   *
   * @param rows 行ごとの O のワード列
   * @param memory blob のバッファを確保する関数（バイト数を受け取る）
   * @return 確定したバッチ
   */
  static HostBatch encode(List<long[]> rows, IntFunction<ByteBuffer> memory) {
    int n = rows.size();
    BatchPlan.Capacity capacity =
        new BatchPlan.Capacity(
            n,
            Math.max(n * Limits.MAX_EVENT_TOKENS, Limits.MAX_EVENT_TOKENS),
            Math.max(n * Limits.MAX_TRANSITIONS, Limits.MAX_TRANSITIONS),
            Math.max(n * Limits.MAX_LEGAL_ACTIONS, Limits.MAX_LEGAL_ACTIONS));
    HostBatch batch = new HostBatch(InputSchema.FORMAT, capacity, memory.apply(BatchPlan.maxBytes(InputSchema.FORMAT, capacity)));
    BatchPlan plan = batch.plan();
    plan.clear();
    Observation read = new Observation(TENHOU_4P);
    for (long[] words : rows) {
      if (plan.add(read.bind(words, 0)) < 0) {
        throw new IllegalStateException("batch capacity is too small");
      }
    }
    plan.seal();
    InputFormat.RowWriter writer = InputSchema.FORMAT.writer();
    HostInputs inputs = new HostInputs(batch);
    for (int row = 0; row < n; row++) {
      writer.write(read.bind(rows.get(row), 0), inputs, row);
    }
    return batch;
  }

  /** 行の最初の手の番号を返す（バッチの中の手の順）。 */
  static int firstAction(HostBatch batch, int row) {
    return batch.plan().firstAction(row);
  }

  static int actions(HostBatch batch, int row) {
    return batch.plan().actions(row);
  }
}
