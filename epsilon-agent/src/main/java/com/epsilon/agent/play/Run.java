package com.epsilon.agent.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.Resolution;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 卓の 1 本の経路。半荘の主経路か、ある判断で選ぶ手を差し替えて分けた続き（{@link #fork}）のどちらか。分けた続きは局の終わりで止まる。
 *
 * <p>判断の時点ごとに、強制でない判断の記録 O を、その時点を進めているスレッドで 1 回ずつ取り込む。系列の処理（{@link RunHooks}）に
 * 問い合わせを作らせ、席の打ち手に答えさせる。答えがそろったら全席の手を共通乱数で引き、系列の処理が打つ手に替えて渡してから
 * エンジンを進める。
 *
 * <p>共通乱数: 手は判断のキー（{@link CrnKeys}）で引く。キーは局のキー、席、判断のクラスと、席と判断のクラスごとに数える序数で決まる。
 * 序数は強制の判断でも進める。分けた続きは親の牌山、局のキー、序数を引き継ぐので、親と同じ手を打った続きはその後も親と同じ乱数を引く。どれも実行の順、同時に進める
 * 半荘の数、ワーカーの数に依らない。
 *
 * <p>一度に触るのは 1 つのスレッドだけ（進行のスレッドかドライバ）で、受け渡しはドライバのイベントの列を通す。分けた続きは {@link #spawned}
 * に入れ、親を進めた進行のワーカーが同じタスクの中で続けて進める。
 *
 * @param <Q> 問い合わせの型
 */
public final class Run<Q extends Query> {

  enum Status {
    /** 非同期の打ち手の答えを待つ。 */
    NEEDS_POLICY,
    /** 経路の終わり（主経路は半荘の終わり、分けた続きは局の終わり）。 */
    FINISHED
  }

  private final Table<Q> table;
  private final boolean main;
  private final GameKey game;
  private final Engine engine;
  private final CrnKeys.Cursor cursor;
  private final List<TableSeat<Q>> seats;

  /** 判断ごとの問い合わせ（強制の判断は null）。1 つの判断の時点で判断がいちばん多いのは他家の応答で、そのときも席の数 − 1 個まで。 */
  private final Query[] queries;

  /** 判断ごとに手を引いた分布（強制でない判断だけ。{@link RunHooks#decided} の間だけ読む）。 */
  private final float[][] distributions;

  /** 判断ごとの判断のキー（強制でない判断だけ。{@link RunHooks#decided} の間だけ読む）。 */
  private final long[] keys;

  /** 判断ごとに選んだ合法手の位置（エンジンにそのまま渡す）。 */
  private final int[] chosen;

  private final List<Q> outstanding;
  private final List<Run<Q>> spawned = new ArrayList<>(0);
  private final RunRecords records = new RunRecords();
  private final RunHooks<Q> hooks;
  private Boundary step;
  private int handOrdinal;
  private long handKey;
  private boolean prepared;
  private int pending;

  /** {@link #advance} の 1 回の呼び出しで数えた判断の数。呼び出しの終わりに卓の件数へ足す。 */
  private long queried, forced;

  /**
   * 半荘の主経路を作る。牌山は run の種と牌山の番号だけで決まり、局のキーはコホートの種、牌山の番号、局の序数だけで決まる。割り当ての
   * 番号は、席に座る打ち手だけを決める。
   */
  Run(Table<Q> table, GameKey game) {
    this.table = table;
    main = true;
    this.game = game;
    TablePlan<Q> plan = table.plan;
    Variant variant = plan.variant();
    cursor = new CrnKeys.Cursor(variant.seats(), table.crn.kindClasses());
    engine = new Engine(variant);
    engine.startGame(TableSeeds.wall(plan.runSeed(), game.wall()), plan.drawOrder());
    seats = plan.seating().seats(game);
    int slots = variant.seats() - 1;
    queries = new Query[slots];
    distributions = new float[slots][];
    keys = new long[slots];
    chosen = new int[slots];
    outstanding = new ArrayList<>(slots);
    handKey = TableSeeds.handKey(plan.iterationSeed(), game.wall(), 0);
    hooks = table.hooks.main(this);
    step = engine.advance(table.tools().workspace);
    if (plan.observer() != null) plan.observer().advanced(game, engine.full());
  }

  /** 親の判断の時点で写したエンジンを、差し替えた手で進めた続きを作る。 */
  private Run(Run<Q> parent, Engine fork, int[] choices, RunHooks<Q> hooks) {
    table = parent.table;
    main = false;
    game = parent.game;
    engine = fork;
    cursor = parent.cursor.copy();
    seats = parent.seats;
    int slots = parent.queries.length;
    queries = new Query[slots];
    distributions = new float[slots][];
    keys = new long[slots];
    chosen = new int[slots];
    outstanding = new ArrayList<>(slots);
    handKey = parent.handKey;
    handOrdinal = parent.handOrdinal;
    this.hooks = hooks;
    step = fork.commit(choices, table.tools().workspace);
  }

  /** 経路の半荘（分けた続きは親の半荘）。 */
  public GameKey game() {
    return game;
  }

  /** 半荘の主経路なら true（分けた続きは false）。 */
  public boolean main() {
    return main;
  }

  /** 半荘の中の今の局の序数（0 始まり）。 */
  public int handOrdinal() {
    return handOrdinal;
  }

  /** 席から見た局面（次にエンジンを進めるまで有効）。 */
  public SeatView view(int seat) {
    return engine.view(seat);
  }

  /** 全ての席の局面（次にエンジンを進めるまで有効）。 */
  public FullView full() {
    return engine.full();
  }

  /**
   * 直前の判断の時点で選んだ手でエンジンを進めた結果。席ごとに、選んだ手が行われたかと、自分だけ選び替えると結果が変わりえたかを持つ。
   * 手を進めた後に呼ばれる {@link RunHooks#boundary} と {@link RunHooks#roundEnded} で読み、次にエンジンを進めるまで有効。
   */
  public Resolution resolution() {
    return engine.resolution();
  }

  /**
   * 今の局が始まったときの卓。次にエンジンを進めるまで有効で、呼び出し側は書き換えない。分けた続きでは親と同じ卓を返す。
   *
   * @throws IllegalStateException 最初の局を配る前（主経路の処理を作っている間）
   */
  public TableState handStart() {
    return engine.handStart();
  }

  /** {@code slot} 番目の判断の問い合わせ（強制の判断は null）。今の判断の時点の間だけ有効。 */
  @SuppressWarnings("unchecked")
  public Q query(int slot) {
    return (Q) queries[slot];
  }

  /** {@code slot} 番目の判断で手を引いた分布（強制でない判断だけ。{@link RunHooks#decided} の間だけ有効）。 */
  public float[] distribution(int slot) {
    return distributions[slot];
  }

  /**
   * {@code slot} 番目の判断で、行動方策 b から引いた手（{@link RunHooks#play} で替えた後）。主経路の探索で引いた判断では、選んだ手と同じ
   * 判断のキーで b から引き直すので、選んだ手と違うことがある。ほかの判断では選んだ手と同じ。{@link RunHooks#decided} の間だけ有効。
   */
  public int behaviourChoice(int slot) {
    Query query = queries[slot];
    Behaviour behaviour = table.behaviour;
    if (query == null || behaviour == null || floored(query)) return chosen[slot];
    float[] b = behaviour.policy(query.sigma, query.legalCount());
    return hooks.play(query(slot), table.crn.select(keys[slot], query.legal(), query.sigma, b, behaviour.epsilon()));
  }

  /** 呼んだスレッドのリーダーを問い合わせの O に付けて返す。同じスレッドで次に読むか取り込むまで有効。 */
  public Observation read(Q query) {
    return query.bind(table.tools().reader);
  }

  /**
   * この判断の時点から、判断ごとに選ぶ手を差し替えた続きを作る。{@link RunHooks#decided} の中で呼ぶ。エンジンは今の局面で写し、続きは
   * 局の終わりで止まる。
   *
   * @param choices 判断ごとに選ぶ合法手の位置（渡した後は使わない）
   * @param hooks 続きの系列の処理
   */
  public void fork(int[] choices, RunHooks<Q> hooks) {
    spawned.add(new Run<>(this, engine.copy(), choices, hooks));
  }

  RunHooks<Q> hooks() {
    return hooks;
  }

  /** 次に止まるところまで進める。進行のスレッドで呼ぶ。 */
  Status advance() {
    long started = System.nanoTime();
    Table.Tools tools = table.tools();
    try {
      while (true) {
        switch (step) {
          case DECISIONS -> {
            if (!prepared && prepare(tools)) return Status.NEEDS_POLICY;
            commit(tools.workspace);
          }
          case ROUND_END -> {
            if (endRound(false, tools.workspace)) return Status.FINISHED;
          }
          case GAME_END -> {
            endRound(true, tools.workspace);
            return Status.FINISHED;
          }
        }
      }
    } finally {
      hooks.advanced(this, System.nanoTime() - started);
      table.count(queried, forced);
      queried = 0;
      forced = 0;
      records.flush(table.timer);
    }
  }

  /**
   * 今の判断の時点にある判断ごとに問い合わせを作る。合法手が 1 つしかない強制の判断は問い合わせない。ほかの判断は O を取り込み、系列の処理に
   * 問い合わせを作らせる。その場で答える打ち手にはすぐ答えさせ、そうでない打ち手への問い合わせは答えを待つ一覧に足す。答えを待つ
   * 問い合わせが残れば true。
   */
  private boolean prepare(Table.Tools tools) {
    outstanding.clear();
    records.clear();
    Decisions decisions = engine.decisions();
    hooks.boundary(this, decisions, tools.workspace);
    Variant variant = table.plan.variant();
    for (int slot = 0; slot < decisions.size(); slot++) {
      if (decisions.legalCount(slot) == 1) {
        queries[slot] = null;
        forced++;
        continue;
      }
      int seat = decisions.seat(slot);
      SeatView view = engine.view(seat);
      TableSeat<Q> policy = seats.get(seat);
      int record = records.capture(tools, variant, view, decisions, slot);
      Observation observation = tools.reader;
      Q query =
          hooks.query(this, decisions, slot, view, policy.admission(observation), records.batch(), record, observation);
      queried++;
      queries[slot] = query;
      if (policy.inline()) {
        long started = System.nanoTime();
        policy.evaluateInline(query);
        table.timer.add(TableStage.INLINE_POLICY, System.nanoTime() - started);
      } else {
        outstanding.add(query);
      }
    }
    prepared = true;
    pending = outstanding.size();
    return pending > 0;
  }

  /**
   * 全席の手を共通乱数で引き（{@link #choose}）、系列の処理が打つ手に替える（{@link RunHooks#play}）。替えた手を、主経路なら観測に渡し、
   * 系列の処理にも渡してからエンジンを進める。
   */
  private void commit(RuleWorkspace ws) {
    Decisions decisions = engine.decisions();
    int count = decisions.size();
    for (int slot = 0; slot < count; slot++) {
      int seat = decisions.seat(slot);
      int kindClass = table.crn.kindClass(decisions.kind(slot));
      // 序数は強制の判断でも進める。こうすると主経路と分けた続きで同じだけ進み、同じ局面では同じキーになる
      int ordinal = cursor.next(seat, kindClass);
      Query query = queries[slot];
      if (query == null) {
        chosen[slot] = 0;
        continue;
      }
      float[] distribution = distribution(query);
      distributions[slot] = distribution;
      keys[slot] = table.crn.decisionKey(handKey, seat, kindClass, ordinal);
      chosen[slot] = hooks.play(query(slot), choose(keys[slot], query, distribution));
    }
    GameObserver observer = table.plan.observer();
    if (main && observer != null)
      for (int slot = 0; slot < count; slot++)
        observer.decided(game, engine.view(decisions.seat(slot)), decisions, slot, chosen[slot]);
    hooks.decided(this, decisions, chosen, ws);
    step = engine.commit(chosen, ws);
    if (main && observer != null) observer.advanced(game, engine.full());
    Arrays.fill(queries, null);
    prepared = false;
  }

  /**
   * 分布（{@link #distribution}）から共通乱数で手を引く。一様分布を混ぜた分布は {@link CrnScheme#select} で引く（{@link CrnScheme#SINGLE_RACE}
   * はその分布の上で 1 回の指数競争、{@link CrnScheme#TWO_STAGE_FLOOR} は 2 段の乱数）。行動方策が無いときの打ち手の分布と、探索の分布は、1 回の指数競争
   * で引く。キーは、局のキー・席・判断のクラス・序数から作る。
   */
  private int choose(long key, Query query, float[] distribution) {
    int[] legal = query.legal();
    return floored(query)
        ? table.crn.select(key, legal, query.sigma, distribution, table.behaviour.epsilon())
        : CrnKeys.select(key, legal, distribution, 0);
  }

  /** 一様分布を混ぜた分布で引く判断なら true（行動方策があり、主経路の探索でない）。 */
  private boolean floored(Query query) {
    Behaviour behaviour = table.behaviour;
    return behaviour != null && !(main && behaviour.temperature() > 0 && query.scores != null);
  }

  /**
   * 手を引く分布。行動方策が無ければ打ち手の分布を、あれば打ち手の分布に一様分布を混ぜた分布（{@link Behaviour#policy}）を使う。分けた続きも同じ分布で引く（手を差し替える
   * のは続きを作るときだけ）。主経路の探索（温度が正で、打ち手が手の価値を出すとき）は {@link Behaviour#explore} の分布を使う。
   */
  private float[] distribution(Query query) {
    int n = query.legalCount();
    Behaviour behaviour = table.behaviour;
    if (behaviour == null) return query.sigma;
    return floored(query) ? behaviour.policy(query.sigma, n) : behaviour.explore(query.scores, n);
  }

  /**
   * 局の終わりを系列の処理に渡し、主経路なら観測にも渡す。主経路で半荘が続くなら次局へ進める。
   *
   * @return 経路が止まるなら true（分けた続きは局の終わり、主経路は半荘の終わり）
   */
  private boolean endRound(boolean gameEnded, RuleWorkspace ws) {
    RoundEnd end = engine.roundEnd();
    hooks.roundEnded(this, end, gameEnded);
    GameObserver observer = table.plan.observer();
    if (main) {
      table.rounds.increment();
      if (observer != null) observer.roundEnded(game, engine.full(), end);
      if (gameEnded) {
        if (observer != null) observer.gameEnded(game, end.next());
        table.games.increment();
      }
    }
    if (gameEnded || !main) return true;
    handOrdinal++;
    handKey = TableSeeds.handKey(table.plan.iterationSeed(), game.wall(), handOrdinal);
    cursor.reset();
    step = engine.advance(ws);
    if (observer != null) observer.advanced(game, engine.full());
    return false;
  }

  /** 問い合わせの席に座る打ち手。 */
  TableSeat<Q> seatOf(Query query) {
    return seats.get(query.seat());
  }

  List<Q> outstanding() {
    return outstanding;
  }

  /** 答えを待つ問い合わせが 1 つ終わったことを数える。全ての答えがそろえば true。 */
  boolean requestCompleted() {
    return --pending == 0;
  }

  /** {@link #advance} の中で作った続き。受け取った側が空にする。 */
  List<Run<Q>> spawned() {
    return spawned;
  }

  /** 診断用の文字列。 */
  @Override
  public String toString() {
    return (main ? "main" : "branch")
        + " wall=" + game.wall()
        + " placement=" + game.placement()
        + " hand=" + handOrdinal;
  }
}
