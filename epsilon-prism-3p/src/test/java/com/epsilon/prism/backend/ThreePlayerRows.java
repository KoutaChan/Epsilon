package com.epsilon.prism.backend;

import static com.epsilon.mahjong.rules.Variant.TENHOU_3P;

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
import com.epsilon.prism3p.input.InputSchema;
import com.epsilon.prism3p.input.InputSchema.ActionRole;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.EnumMap;
import java.util.List;
import java.util.Locale;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 三人打ちの HIP のテスト用のヘルパー。三人打ちの入力形式で乱数で初期化したネットワークと、エンジンで打った三人打ちの半荘の判断を
 * ライターで書いたバッチを作る。学習のテスト（model と learn のパッケージ）も使う。
 */
public final class ThreePlayerRows {
  private static final Logger log = LoggerFactory.getLogger(ThreePlayerRows.class);

  /** 長い行を探すときに打つ、CALLS の半荘の最初の種。 */
  private static final long LONG_SEARCH_FROM = 8_000;

  /** 長い行を探す半荘の数の上限。 */
  private static final int LONG_SEARCH_GAMES = 3_000;

  /**
   * 長い行に求める、公開のイベントの数の下限。三人打ちのイベントの上限は 91 なので、行のトークン（局・席・牌の 31 ＋ イベント）は 128 を
   * 超えない。下限は、イベントの範囲が行の大半を占める長さにする。
   */
  static final int LONG_ROW_EVENTS = 70;

  /** {@link #longRows(int)} が 1 つの半荘から取る判断の上限。 */
  private static final int LONG_PER_GAME = 12;

  private ThreePlayerRows() {}

  /** 三人打ちのテストで比べる形のアーム。v2 の形の深さ 2（本番の形）と、v3 の形。 */
  public enum Arm {
    V2(PrismShape.standard(2)),
    V3(ShapeArm.V3.shape(ShapeArm.V3_DEPTH));

    private final PrismShape shape;

    Arm(PrismShape shape) {
      this.shape = shape;
    }

    public PrismShape shape() {
      return shape;
    }

    /** ログの行に書く名前。 */
    public String label() {
      return name().toLowerCase(Locale.ROOT);
    }
  }

  /**
   * 三人打ちの入力形式でネットワークを作り、全てのパラメーターを乱数でずらす（{@link PerturbedNetwork}）。
   *
   * @param manager パラメーターを持つマネージャー
   * @param shape ネットワークの形
   * @param seed 乱数の種
   * @return ネットワーク
   */
  public static PrismNetwork network(NDManager manager, PrismShape shape, long seed) {
    return PerturbedNetwork.of(manager, InputSchema.FORMAT, shape, seed);
  }

  /** 判断する席の O を取り込み、そのワードの写しを返すヘルパー（スレッドセーフではない）。 */
  private static final class Capture {
    private final ObservationWriter writer = new ObservationWriter(TENHOU_3P);
    private final ObservationBatch batch = new ObservationBatch(TENHOU_3P, 1 << 12);
    private final Observation read = new Observation(TENHOU_3P);

    /** 席の O を取り込み、リーダーに結び付けて返す。 */
    Observation observe(Engine engine, int slot) {
      Decisions decisions = engine.decisions();
      batch.clear();
      int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
      return read.bind(batch, index);
    }

    /** 直前に取り込んだ O のワードの写し。 */
    long[] words() {
      int index = batch.size() - 1;
      return Arrays.copyOfRange(batch.words(), batch.start(index), batch.start(index) + batch.longs(index));
    }
  }

  /**
   * 半荘を打ち、ネットワークに評価させる判断の記録 O を集める。
   *
   * @param seed 最初の半荘の種
   * @param games 半荘の数
   * @param policy 打ち方
   * @param limit 集める判断の上限
   * @return 判断ごとの O のワードの並び
   */
  static List<long[]> observations(long seed, int games, TableWalk.Policy policy, int limit) {
    Capture capture = new Capture();
    List<long[]> rows = new ArrayList<>();
    new TableWalk(TENHOU_3P, policy)
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
   * 価値グラフの全ての作り方（打牌・立直・応答の鳴きとロン・手番の槓とツモと九種九牌と北抜き・PASS）が出る行を集める。GREEDY と CALLS の
   * 半荘を交互に打つ。まだ数が足りない手の扱いを含む判断を、行の数の 2/3 まで先に取り、残りをふつうの判断で埋める。
   *
   * @param rows 行の数
   * @param seed 最初の半荘の種
   * @return O のワードの並び
   */
  public static List<long[]> mixedRows(int rows, long seed) {
    EnumMap<ActionRole, Integer> quota = new EnumMap<>(ActionRole.class);
    quota.put(ActionRole.RIICHI, 3);
    quota.put(ActionRole.TSUMO, 2);
    quota.put(ActionRole.RON, 2);
    quota.put(ActionRole.KAN, 2);
    quota.put(ActionRole.CALL, 3);
    quota.put(ActionRole.KYUSHU, 1);
    quota.put(ActionRole.NUKI, 3);
    Capture capture = new Capture();
    List<long[]> special = new ArrayList<>();
    List<long[]> plain = new ArrayList<>();
    int[] seen = new int[1];
    int specialLimit = rows * 2 / 3;
    for (int game = 0; game < 40 && (special.size() < specialLimit || plain.size() < rows); game++) {
      TableWalk.Policy policy = game % 2 == 0 ? TableWalk.Policy.GREEDY : TableWalk.Policy.CALLS;
      new TableWalk(TENHOU_3P, policy)
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
                    quota.computeIfPresent(ActionRole.of(o.action(a).type()), (role, left) -> Math.max(left - 1, 0));
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
   * 公開のイベントが {@link #LONG_ROW_EVENTS} 個以上ある判断を集める。{@link #LONG_SEARCH_FROM} の種から順に CALLS の半荘を打ち、
   * 半荘ごとにイベントの多い順で {@link #LONG_PER_GAME} 個まで取る。
   *
   * @param count 行の数
   * @return O のワードの並び（半荘の順。半荘の中はイベントの多い順）
   * @throws IllegalStateException 探す範囲で、行の数だけ見つからないとき
   */
  public static List<long[]> longRows(int count) {
    Capture capture = new Capture();
    List<long[]> rows = new ArrayList<>();
    int most = 0;
    for (long seed = LONG_SEARCH_FROM; rows.size() < count && seed < LONG_SEARCH_FROM + LONG_SEARCH_GAMES; seed++) {
      List<long[]> words = new ArrayList<>();
      List<Integer> events = new ArrayList<>();
      new TableWalk(TENHOU_3P, TableWalk.Policy.CALLS)
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
        most = Math.max(most, events.get(order[i]));
      }
    }
    if (rows.size() < count) {
      throw new IllegalStateException(
          "found " + rows.size() + " of " + count + " decisions with " + LONG_ROW_EVENTS + " public events");
    }
    log.info("PRISM-3P-ROWS {} decisions with at least {} public events, most {}", rows.size(), LONG_ROW_EVENTS, most);
    return rows;
  }

  /**
   * GREEDY と CALLS の半荘から交互に集めた行。
   *
   * @param count 行の数
   * @param seed 最初の半荘の種
   * @return O のワードの並び
   */
  static List<long[]> simulatedRows(int count, long seed) {
    List<long[]> greedy = observations(seed, 1 + count / 400, TableWalk.Policy.GREEDY, count);
    List<long[]> calls = observations(seed + 1, 1 + count / 400, TableWalk.Policy.CALLS, count);
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
   * O の並びを、三人打ちの 1 つのバッチに符号化する（blob はダイレクトバッファ）。
   *
   * @param rows O のワードの並び
   * @return 確定したバッチ
   */
  public static HostBatch encode(List<long[]> rows) {
    int n = rows.size();
    BatchPlan.Capacity capacity =
        new BatchPlan.Capacity(
            n,
            Math.max(n * Limits.MAX_EVENT_TOKENS, Limits.MAX_EVENT_TOKENS),
            Math.max(n * Limits.MAX_TRANSITIONS, Limits.MAX_TRANSITIONS),
            Math.max(n * Limits.MAX_LEGAL_ACTIONS, Limits.MAX_LEGAL_ACTIONS));
    HostBatch batch =
        new HostBatch(
            InputSchema.FORMAT, capacity, ByteBuffer.allocateDirect(BatchPlan.maxBytes(InputSchema.FORMAT, capacity)));
    BatchPlan plan = batch.plan();
    plan.clear();
    Observation read = new Observation(TENHOU_3P);
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
}
