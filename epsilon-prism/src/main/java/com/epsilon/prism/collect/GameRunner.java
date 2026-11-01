package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.core.CrnKeys;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.luck.LuckBatch;
import com.epsilon.prism.luck.LuckSampling;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.policy.SeatRouter;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 1本の半荘の進行（主経路）と、その判断から分かれた写しと分岐（{@link Role}）。
 *
 * <p>主経路（{@link Role#MAIN}）の判断ごと（合法手が双子だけでない判断。{@link LabelPlan#decide}）:
 *
 * <ol>
 *   <li>prepare（推論の前）: 類 c と予定（ラベル行か・ロールインか）を硬貨で決め、どちらかなら retain の問い合わせにする（方策が p0 を書き、
 *       記録 O を捕捉する）。
 *   <li>commit（推論の後、engine の commit の前）: 全席の手を共通乱数で選ぶ（{@link #choose}）。retain の判断では候補集合 C(o)（{@link
 *       CandidateSet#build}）を作り、
 *       <ul>
 *         <li>ラベル行なら {@link PendingLabel}（source MAIN）を作り、打った手の枠をこの局の targets に足し、C ∪ {played} のほかの手ごとに分岐
 *             （{@link Role#FORK}）を作る。
 *         <li>ロールインなら、b が貪欲の手（σ の argmax。双子を含む）を打った判断でだけ d ~ q（{@link RollIn#draw}）を引き、{@link
 *             RootRecord} を作って打った手の枠を targets に足す。d ≠ played なら写し（{@link Role#COPY}）を作り、写しの targets に d の枠を入れる
 *             （同じ判断のラベル行が d を測るなら、その d の枠も写しに入れて d の分岐を作らない）。床 ε_b で別の手を打った判断は根も写しも作らず、
 *             件数（floorRoots）だけ数える。根の y_main を π_t(o) の値にするため（床 ε_b の手の値を根に混ぜない）。選別は手の抽選の共通乱数
 *             だけで決まり、以後の進行の乱数とは独立なので、残った根の y_main の期待値は Q_b(o, π_t(o))。
 *       </ul>
 *   <li>engine の commit。主経路は逸脱しない。
 * </ol>
 *
 * <p>写し（{@link Role#COPY}）: 主経路の境界で engine を複製し、逸脱した席の手だけを d に差し替えて commit し、局末まで全席 b で打って止まる。
 * 逸脱した席の後の判断は予定（率 ρ_br、最大 maxCopyLabels）に当たればラベル行（source ROLLIN、rootDecisionId は写しの根の判断 ID）にし、写しの中から
 * 分岐を作る。ほかの席の判断は行にしない。ロールインはしない。
 *
 * <p>分岐（{@link Role#FORK}）: 主経路か写しの境界で engine を複製し、判断の席の手だけを候補 a に差し替えて commit し、局末まで全席 b で打って
 * 止まる。行も分岐も作らない。局末の値を親の記録の 1 枠に渡すだけの測定。
 *
 * <p>値: 局が終わると、この局の targets の全ての枠に局末の 4 席の値を渡す（{@link HandEnd}。ドライバが Φ を求めて {@link Pending#accept} を呼ぶ）。
 *
 * <p>運の精算（{@link CollectContext#eventLedger} のときだけ）: 進行ごとに {@link EventLedger.Tracker} を 1 つ持ち、境界ごと（prepare）に前の境界
 * からのツモと槓ドラの表示を、強制でない判断ごと（commit の手を選ぶ輪）に方策の偶然を、局末（endHand）に裏ドラを、この局の targets の枠の帳簿に
 * 足す。枠は判断の後に作るので o の前の出来事は数えない。写しと分岐は親の計算器を写して始める。帳簿を付けない収集では計算も乱数の消費も手も変えない。
 *
 * <p>記録 O: 進行ごとに {@link QueryRecords} を 1 つ持ち、prepare で、O を読む方策（{@link PolicyService#readsRecord}）に出す問い合わせと行に残す
 * 判断の O だけを、その境界を進めているスレッドで 1 回捕捉する（強制の判断は捕捉しない）。符号化のスレッドはその O だけを読み、局面に触れない。
 * 置き場は次の境界の prepare まで生きる。
 *
 * <p>行（{@link CollectContext#rows} のときだけ）: 進行ごとに {@link RowRecorder} を 1 つ持ち、行を作る判断の record で問い合わせの O の写しと
 * 神託を作り（{@link RowRecorder#decision}）、境界と局末で聴牌への到達を追い、枠のある局末で局末の事実を {@link HandEnd} に載せる。写しと分岐は親の
 * 記録を写して始める。状態を読むだけなので、進行と乱数は変えない。
 *
 * <p>運の網 Ω の標本（{@link CollectContext#luck} のときだけ）: 進行ごとに {@link LuckRecorder} を 1 つ持ち、境界ごと（prepare）に
 * {@link EventLedger.Tracker#observe} が復元したツモと槓ドラの表示を、帳簿の枠の有無に依らず硬貨（{@link LuckSampling}）で引いて O° を捕捉し、
 * 局末（endHand）にその局の標本を {@link HandEnd} に載せる（ドライバがその経路の局末の値を Y にして書く）。主経路で枠のない局も、標本があれば
 * HandEnd を作る（局末の値を求めない局の標本を捨てると、局の結果で標本を選ぶことになる）。硬貨の鍵の根（経路と根の判断）の置き方:
 *
 * <ul>
 *   <li>主経路: 局の始めに (MAIN_PATH, handRoot(局の序数))。写しか分岐を作った判断 D の後は (MAIN_PATH, D)（ラベル行の分岐に加えて、中立
 *       ロールインの写しも兄弟にする）。1 つの境界で複数の判断が経路を分けたら、境界の判断の並びで先の判断を根にする。
 *   <li>主経路の判断 D の写しと分岐: (MAIN_PATH, D) で始める。
 *   <li>写し（根の判断 R）: 写しの中で写しか分岐を作った判断 D′ の後は (R, D′)。その分岐は (R, D′) で始める（写しの判断 ID は主経路と同じ数を
 *       使い回すので、経路 R で分ける）。
 * </ul>
 *
 * <p>同じ判断から分かれた経路と、その判断の後の親の続きは同じ根なので、同じ序数の出来事を一緒に引く（兄弟の間で中心化した c のばらつきを測れる）。
 * 鍵は出来事の前の経路だけで決まり、結果を含まない。状態を読むだけなので、進行・乱数・行は変えない。
 *
 * <p>共通乱数: 手の抽選は局のキーと席×種別類の序数（{@link CrnKeys}）で、写しと分岐は複製時に山と序数の進みを引き継ぐ。同じ手を打ち直した経路は
 * 主経路と同じ値になる（候補の間の CRN）。硬貨は進行の硬貨の鍵（主経路は {@link CollectContext#mainKey}、写しは {@link CollectContext#copyKey}）
 * と判断 ID のハッシュ。実行順・同時進行数・ワーカー数に依らない。判断 ID は応答できる席の数で枝ごとにずれる（他家の伏せた手牌に依る）ので、硬貨は
 * 席 i の観測だけの関数ではない。ただし鍵は山と独立なハッシュなので、隠れた状態 H を固定しても硬貨は一様で、採否の確率は H に依らない（{@link
 * CrnSeeds}）。
 *
 * <p>一度に触るのは1スレッドだけ（進行ワーカーかドライバ）で、受け渡しはドライバのイベントキューを通す。写しと分岐は {@link #spawned} に入れ、
 * ドライバが同じ仕事の中で深さ優先に進める（写しが作った分岐も同じ）。
 */
final class GameRunner {
  enum Status {
    /** 非同期の方策の結果を待つ。 */
    NEEDS_POLICY,
    /** 半荘の終わり（写しと分岐は局の終わり）。 */
    FINISHED
  }

  /** 1 つの境界の判断の数の上限（手番 1 つか、応答 3 つ）。 */
  static final int MAX_POINTS = 3;

  private final CollectContext context;
  private final Role role;
  private final long gameId;
  private final GameEngine engine;

  /** 判断の共通乱数の使い方（b がなければ {@link CrnScheme#V1}）。 */
  private final CrnScheme crn;

  private final CrnKeys.Cursor cursor;

  /** この半荘の行の用途と組。 */
  private final Row.Purpose purpose;

  private final Row.Group group;
  private final CollectionMetrics metrics;

  /** prepare で数える判断の件数（advance の終わりに metrics へ足す）。 */
  private final DecisionCounts decisionCounts;

  private final PolicyService[] seats;

  /** 進行の硬貨の鍵（主経路は mainKey、写しは copyKey、分岐は 0 で使わない）。 */
  private final long coinKey;

  /** 写しの逸脱した席（主経路と分岐は -1）。 */
  private final int deviator;

  /** 写しの根の主経路の判断 ID（主経路と分岐は -1）。 */
  private final long rootDecisionId;

  private final EngineSelectionBuffer selections = new EngineSelectionBuffer();
  private final PolicyRequest[] requests = new PolicyRequest[MAX_POINTS];

  /** 判断 index を打つ方策（ふつうは席の方策。席が {@link SeatRouter} なら判断ごとに選んだ方策）。 */
  private final PolicyService[] policies = new PolicyService[MAX_POINTS];
  private final LabelPlan[] plans = new LabelPlan[MAX_POINTS];
  private final int[] chosen = new int[MAX_POINTS];
  private final List<PolicyRequest> outstanding = new ArrayList<>(MAX_POINTS);
  private final List<HandEnd> handEnds = new ArrayList<>(1);

  /** この境界で作った写しと分岐（ドライバが引き取る）。 */
  private final List<GameRunner> spawned = new ArrayList<>(0);

  /** この局の値を受ける枠。写しと分岐は作られたときに親が入れた枠で始まる。主経路は局ごとに新しい一覧で始める。 */
  private List<Slot> targets;

  /** 運の精算の帳簿の計算器と出来事の復元（帳簿も運の網の標本も付けない収集は null）。 */
  private final EventLedger.Tracker ledger;

  /** 境界の問い合わせの記録 O（次の境界の prepare まで）。 */
  private final QueryRecords records = new QueryRecords();

  /** 行の記録（行を書かない評価は null）。 */
  private final RowRecorder rows;

  /** 運の網の標本の捕捉（標本を集めない収集は null）。 */
  private final LuckRecorder luck;

  private GameStepResult step;
  private RootState root;

  /** いま準備している境界の段階（境界の全判断で同じ）。 */
  private Stage stage;

  /** 写しでこれまでに予約したラベル行の数。 */
  private int copyLabels;

  private int handOrdinal;
  private long handKey;
  private boolean prepared;
  private int pendingRequests;

  /** 半荘の主経路。牌山は run の種と半荘 ID だけで決まる。 */
  GameRunner(CollectContext context, long gameId) {
    this.context = context;
    this.role = Role.MAIN;
    decisionCounts = new DecisionCounts(Role.MAIN);
    this.gameId = gameId;
    crn = context.behaviour == null ? CrnScheme.V1 : context.behaviour.scheme();
    cursor = new CrnKeys.Cursor(crn.kindClasses());
    root = RootState.initial();
    engine =
        new GameEngine(
            CrnSeeds.derive(context.runSeed, CrnSeeds.Domain.WALL, gameId),
            context.observer == null ? null : context.observer.recorder(gameId));
    engine.getState().enableSeatStreamDraws();
    purpose = context.purpose(gameId);
    group = context.group(gameId);
    metrics = context.metrics;
    seats =
        context.seating == null
            ? new PolicyService[] {context.policy, context.policy, context.policy, context.policy}
            : context.seating.seats(gameId);
    coinKey = context.mainKey(gameId);
    deviator = -1;
    rootDecisionId = -1;
    targets = new ArrayList<>();
    luck =
        context.luck == null
            ? null
            : new LuckRecorder(context.luck, gameId, LuckSampling.MAIN_PATH, LuckSampling.handRoot(0));
    ledger = context.eventLedger || luck != null ? new EventLedger.Tracker() : null;
    rows = context.rows == null ? null : new RowRecorder();
    handKey = CrnKeys.handKey(context.iterationSeed, gameId, 0);
    step = engine.stepHanchan();
  }

  /**
   * 写しか分岐。親の境界で複製した engine に差し替えた選択を commit した続き。序数（cursor）・局の root・局のキー・席の方策は親の今の状態を
   * 引き継ぐ。
   *
   * @param role {@link Role#COPY} か {@link Role#FORK}
   * @param point 経路を分けた親の判断（写しの逸脱した席と根の判断 ID。運の網の標本の根の判断）
   * @param slots この経路の局末の値を受ける枠
   */
  private GameRunner(
      GameRunner parent,
      Role role,
      GameEngine fork,
      EngineSelectionBuffer forked,
      EngineDecisionPoint point,
      List<Slot> slots) {
    if (role == Role.MAIN) throw new IllegalArgumentException("Spawned runners are copies or forks");
    context = parent.context;
    this.role = role;
    decisionCounts = new DecisionCounts(role);
    gameId = parent.gameId;
    crn = parent.crn;
    cursor = parent.cursor.copy();
    root = parent.root;
    engine = fork;
    purpose = parent.purpose;
    group = parent.group;
    metrics = parent.metrics;
    seats = parent.seats;
    deviator = role == Role.COPY ? point.player() : -1;
    rootDecisionId = role == Role.COPY ? point.id() : -1;
    coinKey = role == Role.COPY ? context.copyKey(gameId, rootDecisionId) : 0L;
    targets = new ArrayList<>(slots);
    handKey = parent.handKey;
    handOrdinal = parent.handOrdinal;
    // 運の網の標本は、親の経路で経路を分けた判断を根にして始める（兄弟の経路と、その判断の後の親の続きと同じ鍵）
    luck = parent.luck == null ? null : new LuckRecorder(context.luck, gameId, parent.luckPath(), point.id());
    // 計算器は親の commit の前の状態（o の境界）から写す。o の打牌の後の表示とツモは、この経路の最初の prepare で数える
    ledger = parent.ledger == null ? null : new EventLedger.Tracker(parent.ledger, parent.engine.getState());
    // 行の記録の到達も親の o の境界から写す（o の打牌はこの経路の最初の境界で数える）
    rows = parent.rows == null ? null : new RowRecorder(parent.rows);
    step = fork.commitDecisions(forked);
  }

  Role role() {
    return role;
  }

  /** 次に止まるところまで進める。進行ワーカーで呼ぶ。写しと分岐は局の終わりで止まる。 */
  Status advance() {
    long started = System.nanoTime();
    try {
      while (true) {
        if (step instanceof GameStepResult.AwaitingDecisions awaiting) {
          List<EngineDecisionPoint> points = awaiting.decisions();
          if (!prepared && prepare(points)) return Status.NEEDS_POLICY;
          commit(points);
        } else if (step instanceof GameStepResult.RoundSettled settled) {
          endHand(settled.settlement(), false);
          if (role != Role.MAIN) return Status.FINISHED;
        } else if (step instanceof GameStepResult.HanchanEnded ended) {
          endHand(ended.settlement(), true);
          return Status.FINISHED;
        } else {
          throw new IllegalStateException("Unexpected engine boundary " + step);
        }
      }
    } finally {
      context.timer.add(
          switch (role) {
            case MAIN -> StageTimer.Stage.MAIN_ADVANCE;
            case COPY -> StageTimer.Stage.COPY_ADVANCE;
            case FORK -> StageTimer.Stage.FORK_ADVANCE;
          },
          System.nanoTime() - started);
      decisionCounts.flush(metrics);
      if (ledger != null) ledger.flush(context.timer);
      records.flush(context.timer);
      if (rows != null) rows.flush(context.timer);
    }
  }

  /**
   * 境界の各判断の方策を求める。合法手 1 つは強制手で問い合わせない（requests[index] = null、plans[index] = {@link LabelPlan#NONE}）。ほかの判断は
   * 種別・段階・脅威の数・局の番号から類を決め（{@link DecisionClass#of}）、予定を決め（{@link LabelPlan#decide}。写しのラベル行の
   * 予約はここで copyLabels を数える）、{@link PolicyRequest}（retain = 予定の retain）を作り、記録 O を要る問い合わせだけ捕捉する（{@link
   * #capture}）。inline の方策はその場で評価し、そうでなければ outstanding に足す。件数（modelDecisions・役割別・類ごとの主経路の判断）もここで
   * 経路の {@link DecisionCounts} に数える（advance の終わりに metrics へ足す）。非同期の問い合わせが残れば true。
   */
  private boolean prepare(List<EngineDecisionPoint> points) {
    outstanding.clear();
    records.clear();
    GameState state = engine.getState();
    if (ledger != null) ledger.observe(state, points, targets, luck);
    if (rows != null) rows.boundary(state, targets);
    stage = Stage.of(state.remainingWallTiles());
    for (int index = 0; index < points.size(); index++) {
      EngineDecisionPoint point = points.get(index);
      List<Action> legal = point.legalActions();
      if (legal.size() == 1) {
        requests[index] = null;
        plans[index] = LabelPlan.NONE;
        decisionCounts.countForced();
        continue;
      }
      int seat = point.player();
      Kind kind = Kind.classify(point.kind(), legal);
      int threats = Threats.count(state, seat);
      DecisionClass decisionClass =
          DecisionClass.of(kind, stage, threats, state.getKyokuIndex());
      decisionCounts.countModel(decisionClass);
      LabelPlan plan =
          LabelPlan.decide(
              context,
              role,
              coinKey,
              point.id(),
              seat,
              deviator,
              copyLabels,
              decisionClass,
              threats,
              legal);
      // 写しのラベル行は予約した時点で上限に数える（写しの進行は決定的なので、どの判断が当たるかも実行順に依らない）
      if (role == Role.COPY && plan.label()) copyLabels++;
      plans[index] = plan;
      var request =
          new PolicyRequest(
              this,
              gameId,
              index,
              seat,
              engine.observation(seat),
              point.kind(),
              legal,
              kind,
              decisionClass,
              plan.retain());
      requests[index] = request;
      PolicyService policy = seats[seat];
      if (policy instanceof SeatRouter router) policy = router.route(request);
      policies[index] = policy;
      capture(request, policy, plan);
      if (policy.inline()) {
        long started = System.nanoTime();
        policy.evaluateInline(state, point, request);
        context.timer.add(StageTimer.Stage.INLINE_POLICY, System.nanoTime() - started);
      } else {
        outstanding.add(request);
      }
    }
    prepared = true;
    pendingRequests = outstanding.size();
    return pendingRequests > 0;
  }

  /**
   * 記録 O を要る問い合わせだけ、経路の置き場に 1 回捕捉する（強制の判断はここに来ない）: O を読む方策に出す問い合わせ（符号化のため）と、行に
   * 残す判断（行の O のため）。失敗はこの進行の失敗にする。
   */
  private void capture(PolicyRequest request, PolicyService policy, LabelPlan plan) {
    if (policy.readsRecord() || (rows != null && plan.retain())) records.capture(request);
  }

  /**
   * 全席の手を共通乱数で選び（{@link #choose}）、retain の判断を記録にし（{@link #record}）、engine を commit する。observer があれば主経路の
   * 判断を通知する（commit の前の state）。
   */
  private void commit(List<EngineDecisionPoint> points) {
    int count = points.size();
    selections.clear();
    for (int index = 0; index < count; index++) {
      EngineDecisionPoint point = points.get(index);
      int kindClass = crn.kindClass(point.kind());
      // 序数は強制手でも進める（主経路・写し・分岐で同じだけ進み、同じ局面では同じ鍵になる）
      int ordinal = cursor.next(point.player(), kindClass);
      PolicyRequest request = requests[index];
      if (request == null) {
        chosen[index] = 0;
      } else {
        float[] distribution = distribution(point, request);
        chosen[index] = choose(point, kindClass, ordinal, request, distribution);
        // 方策の偶然（record より前なので、この境界で作る o の枠には足さない）
        if (ledger != null)
          ledger.policy(point.player(), request.scores, distribution, chosen[index], point.legalActions().size(), targets);
      }
      if (role == Role.MAIN
          && request != null
          && request.sigma != null
          && !Twins.same(point.legalActions(), chosen[index], greedy(request.sigma)))
        metrics.mainNonGreedy.increment();
      selections.add(point.id(), point.legalActions().get(chosen[index]));
    }
    if (role == Role.MAIN && context.observer != null)
      for (int index = 0; index < count; index++)
        context.observer.decided(
            gameId,
            engine.getState(),
            points.get(index),
            points.get(index).legalActions().get(chosen[index]));
    // 記録（候補集合・写し・分岐）は engine の commit の前に行う（複製は判断の境界から作る）
    long branched = -1;
    for (int index = 0; index < count; index++) {
      if (!plans[index].retain()) continue;
      int before = spawned.size();
      record(points, index);
      if (branched < 0 && spawned.size() > before) branched = points.get(index).id();
    }
    // 運の網の標本: 経路を分けた判断の後の続きは、作った写しと分岐と同じ根で引く
    if (luck != null && branched >= 0) luck.root(luckPath(), branched);
    step = engine.commitDecisions(selections);
    Arrays.fill(requests, null);
    Arrays.fill(plans, LabelPlan.NONE);
    prepared = false;
  }

  /**
   * 手の選び: 分布（{@link #distribution}）から共通乱数で引く。床を混ぜた b は {@link CrnScheme#select}（{@link CrnScheme#V1} は b の上の 1 回
   * の競争、{@link CrnScheme#V2} は 2 段の硬貨）、σ と探索の分布は 1 回の競争。
   *
   * <pre>
   * crn.select(crn.decisionKey(handKey, 席, 種別類, 序数), legal, σ, b, ε_b)      （床を混ぜた b）
   * CrnKeys.select(crn.decisionKey(handKey, 席, 種別類, 序数), legal, distribution, 0)
   * </pre>
   */
  private int choose(
      EngineDecisionPoint point, int kindClass, int ordinal, PolicyRequest request, float[] distribution) {
    long key = crn.decisionKey(handKey, point.player(), kindClass, ordinal);
    List<Action> legal = point.legalActions();
    return floored(request)
        ? crn.select(key, legal, request.sigma, distribution, context.behaviour.epsilon())
        : CrnKeys.select(key, legal, distribution, 0);
  }

  /** 床を混ぜた b で引く判断か（b があり、主経路の探索でない）。 */
  private boolean floored(PolicyRequest request) {
    Behaviour behaviour = context.behaviour;
    return behaviour != null
        && !(role == Role.MAIN && behaviour.temperature() > 0 && request.scores != null);
  }

  /**
   * 手を引く分布。b（{@link Behaviour}: σ の貪欲に床 ε_b）があれば b、なければ σ（評価）。写しと分岐も同じ（写しの逸脱は作るときに差し替えるだけ
   * で、写しの中は b）。主経路の探索（温度が正）は {@link Behaviour#explore}。
   */
  private float[] distribution(EngineDecisionPoint point, PolicyRequest request) {
    int n = point.legalActions().size();
    Behaviour behaviour = context.behaviour;
    if (behaviour == null) return request.sigma;
    return floored(request) ? behaviour.policy(request.sigma, n) : behaviour.explore(request.scores, n);
  }

  /**
   * retain の判断 index の記録。state は commit の前。候補集合を 1 度だけ作り（played = chosen[index]、ε_C の硬貨はこの進行の硬貨の鍵）、予定の
   * ラベル行・ロールインを {@link GameRunner} の説明どおりに作る。分岐と写しは {@link #spawn} で作る。件数（labelRows・copyLabelRows・roots・
   * floorRoots・copies・sameDeviations・forks・measured・explored・playedOutside）もここで数える。時間は {@link StageTimer.Stage#CANDIDATES}。
   */
  private void record(List<EngineDecisionPoint> points, int index) {
    EngineDecisionPoint point = points.get(index);
    PolicyRequest request = requests[index];
    LabelPlan plan = plans[index];
    if (request.scores == null || request.prior == null)
      throw new IllegalStateException("Retained decisions need s and p0 from the policy: " + this);
    List<Action> legal = point.legalActions();
    int played = chosen[index];
    long id = point.id();
    int classIndex = plan.decisionClass().ordinal();
    boolean exploring =
        role == Role.MAIN && context.behaviour != null && context.behaviour.temperature() > 0;
    List<Slot> playedSlots = new ArrayList<>(2);
    long started = System.nanoTime();
    CandidateSet candidates =
        context.sampling.measuresAll(plan.decisionClass())
            ? CandidateSet.allActions(legal, played)
            : CandidateSet.build(
                state(),
                point.player(),
                legal,
                request.scores,
                request.prior,
                plan.threats(),
                played,
                context.sampling.maxCandidates(),
                context.sampling.epsilonC(),
                context.exploreCoin(coinKey, id),
                context.exploreDraw(coinKey, id));
    // ロールインは b が貪欲の手を打った判断だけ（床 ε_b の手を打った根は作らない。クラスの Javadoc）
    boolean rollIn = plan.rollIn() && Twins.same(legal, played, greedy(request.sigma));
    if (plan.rollIn() && !rollIn) metrics.floorRoots.increment();
    RollIn.Deviation deviation =
        rollIn
            ? RollIn.draw(
                candidates.support(),
                request.scores,
                request.prior,
                context.sampling.epsilonQ(),
                context.deviationCoin(coinKey, id))
            : null;
    context.timer.add(StageTimer.Stage.CANDIDATES, System.nanoTime() - started);
    byte[] support = candidates.supportBytes();
    int[] measured = candidates.measured();
    boolean labelled = plan.label();
    if (candidates.explored()) metrics.explored.increment();
    // 行の記録（行を作る判断だけ。O の写しと神託）
    RowRecorder.Decision recorded = labelled || deviation != null ? rows.decision(request, state()) : null;
    PendingLabel label = null;
    if (labelled) {
      boolean main = role == Role.MAIN;
      label =
          new PendingLabel(
              context,
              draft(
                  point,
                  request,
                  plan,
                  recorded,
                  main ? Row.Source.MAIN : Row.Source.ROLLIN,
                  played,
                  support,
                  -1,
                  Float.NaN,
                  Float.NaN,
                  main ? -1 : rootDecisionId),
              measured.length);
      // 打った手はこの経路（主経路か写し）の局末の値がただで付く。主経路が探索するとき（Behaviour#explore）は続きが b でないので、打った手も
      // 分岐（b の続き）で測る（主経路の値を使うと打った手だけ続きが違い、差が片側にずれ、候補の間の共通乱数も切れる）
      if (exploring) playedSlots.add(slot(label, played));
      else targets.add(slot(label, played));
      if (main) metrics.labelRows[classIndex].increment();
      else metrics.copyLabelRows[classIndex].increment();
      metrics.measured.add(measured.length);
      if (!candidates.playedInside(played)) metrics.playedOutside.increment();
    }

    // 写しが値を測るラベル行の手（同じ判断のラベル行が d を測るなら d の分岐は作らない）
    int coveredByCopy = -1;
    if (deviation != null) {
      int d = deviation.action();
      double rollinRate = context.sampling.rollinRate(plan.decisionClass());
      var rootRecord =
          new RootRecord(
              context,
              draft(
                  point,
                  request,
                  plan,
                  recorded,
                  Row.Source.ROOT,
                  played,
                  support,
                  d,
                  deviation.probability(),
                  (float) (rollinRate * deviation.probability()),
                  -1));
      if (exploring) playedSlots.add(slot(rootRecord, played));
      else targets.add(slot(rootRecord, played));
      metrics.roots[classIndex].increment();
      if (rootRecord.sameAsPlayed()) {
        // d = played: 写しは主経路と同じ過程なので作らない（y_br = y_main）
        metrics.sameDeviations.increment();
      } else {
        List<Slot> slots = new ArrayList<>(2);
        slots.add(slot(rootRecord, d));
        if (label != null && contains(measured, d)) {
          slots.add(slot(label, d));
          coveredByCopy = d;
        }
        spawn(points, index, d, Role.COPY, slots);
        metrics.copies.increment();
      }
    }

    if (label != null) {
      for (int a : measured) {
        if (a == played || a == coveredByCopy) continue;
        spawn(points, index, a, Role.FORK, List.of(slot(label, a)));
        metrics.forks.increment();
      }
    }
    if (!playedSlots.isEmpty()) {
      spawn(points, index, played, Role.FORK, playedSlots);
      metrics.forks.increment();
    }
  }

  /**
   * 行の下書き（ラベル行は deviation = −1・q と傾向は NaN、根の記録は d・q(d|o)・β_c(o)·q(d|o)。recorded はこの判断の行の記録）。
   *
   * @param root ROLLIN は写しの根の判断 ID、ほかは −1
   */
  private RowDraft draft(
      EngineDecisionPoint point,
      PolicyRequest request,
      LabelPlan plan,
      RowRecorder.Decision recorded,
      Row.Source source,
      int played,
      byte[] support,
      int deviation,
      float deviationProbability,
      float propensity,
      long root) {
    return new RowDraft(
        gameId,
        point.id(),
        point.player(),
        request.kind(),
        plan.decisionClass(),
        purpose,
        group,
        source,
        played,
        point.legalActions().size(),
        request.prior,
        support,
        deviation,
        deviationProbability,
        propensity,
        root,
        recorded);
  }

  private static boolean contains(int[] actions, int action) {
    for (int a : actions) if (a == action) return true;
    return false;
  }

  /** 手 action の経路の局末の値を target に渡す枠（帳簿を付ける収集なら空の帳簿つき）。 */
  private Slot slot(Pending target, int action) {
    return new Slot(target, action, context.eventLedger ? new double[EventLedger.FEATURES] : null);
  }

  /** 方策の貪欲の手（σ の最大の手。網の方策の σ は one-hot。同点は番号の小さい手）。 */
  private static int greedy(float[] sigma) {
    int best = 0;
    for (int a = 1; a < sigma.length; a++) if (sigma[a] > sigma[best]) best = a;
    return best;
  }

  /**
   * 判断 index の席の手だけ action に差し替えた写しか分岐を作り、{@link #spawned} に足す。ほかの判断は主経路（この進行）と同じ手。engine の複製
   * （{@link GameEngine#forkAtDecisionBoundary()}）は commit の前に行う。
   *
   * @param role {@link Role#COPY}（逸脱した席 = 判断 index の席、根の判断 ID = その判断の ID）か {@link Role#FORK}
   * @param slots 作った経路の局末の値を受ける枠
   */
  private void spawn(
      List<EngineDecisionPoint> points, int index, int action, Role role, List<Slot> slots) {
    GameEngine fork = engine.forkAtDecisionBoundary();
    var forked = new EngineSelectionBuffer();
    for (int other = 0; other < points.size(); other++) {
      EngineDecisionPoint p = points.get(other);
      forked.add(p.id(), p.legalActions().get(other == index ? action : chosen[other]));
    }
    spawned.add(new GameRunner(this, role, fork, forked, points.get(index), slots));
  }

  /** 運の網の標本の根の経路（{@link LuckSampling}）: 写しは根の判断 ID、主経路は MAIN_PATH（分岐は経路を分けない）。 */
  private long luckPath() {
    return role == Role.COPY ? rootDecisionId : LuckSampling.MAIN_PATH;
  }

  /**
   * 局の終わり。枠か運の網の標本があれば局末の値を待つ先として渡し（行を書く収集なら局末の事実も載せる）、半荘が続くなら（主経路は）次局へ進む。
   */
  private void endHand(RoundSettlement settlement, boolean hanchanEnded) {
    // 裏ドラの項と局末の事実は次局を始める前（局末の状態）に作る
    if (ledger != null && !targets.isEmpty()) ledger.handEnd(engine.getState(), settlement, targets);
    HandEndFacts facts =
        rows != null && !targets.isEmpty() ? rows.handEnd(engine.getState(), settlement, targets) : null;
    LuckBatch samples = luck == null ? null : luck.detach();
    int[] scores = settlement.snapshotFinalScores();
    List<Slot> slots = targets;
    if (role == Role.MAIN) {
      metrics.hands.increment();
      if (context.observer != null)
        context.observer.handEnded(gameId, engine.getState(), settlement.result());
    }
    if (hanchanEnded) {
      if (role == Role.MAIN) {
        if (context.observer != null) {
          context.observer.handValued(gameId, null, scores);
          context.observer.gameEnded(gameId, scores);
        }
        metrics.games.increment();
      }
      if (!slots.isEmpty() || samples != null)
        handEnds.add(
            new HandEnd(
                null, scores, HandValue.terminal(scores, context.profile), slots, facts, samples));
      return;
    }
    var next = (RoundTransition.NextRound) settlement.transition();
    RootState nextRoot = root.next(next, scores);
    if (!slots.isEmpty() || samples != null)
      handEnds.add(new HandEnd(nextRoot.publicPrefix(), scores, null, slots, facts, samples));
    if (role != Role.MAIN) return;
    if (context.observer != null) context.observer.handValued(gameId, nextRoot.publicPrefix(), scores);
    targets = new ArrayList<>();
    if (ledger != null) ledger.reset();
    if (rows != null) rows.reset();
    root = nextRoot;
    handOrdinal++;
    if (luck != null) luck.root(LuckSampling.MAIN_PATH, LuckSampling.handRoot(handOrdinal));
    handKey = CrnKeys.handKey(context.iterationSeed, gameId, handOrdinal);
    cursor.reset();
    step = engine.stepHanchan();
  }

  PolicyService policyFor(PolicyRequest request) {
    return policies[request.point()];
  }

  List<PolicyRequest> outstanding() {
    return outstanding;
  }

  /** 非同期の問い合わせが1件終わった。全件そろえば true。 */
  boolean requestCompleted() {
    return --pendingRequests == 0;
  }

  List<HandEnd> handEnds() {
    return handEnds;
  }

  /** この進行で作った写しと分岐。受け取った側が空にする。 */
  List<GameRunner> spawned() {
    return spawned;
  }

  /** 判断の時点の state（候補集合と体制の代表が公開情報を読む）。 */
  private GameState state() {
    return engine.getState();
  }

  /** 枠を受ける一覧の写し（試験と診断用）。 */
  List<Slot> targets() {
    return List.copyOf(targets);
  }

  /** 診断用の文字列。 */
  @Override
  public String toString() {
    return role
        + " game="
        + gameId
        + " hand="
        + handOrdinal
        + " deviator="
        + deviator
        + " root="
        + rootDecisionId
        + " copyLabels="
        + copyLabels
        + " group="
        + group
        + " purpose="
        + purpose
        + " pending="
        + Arrays.toString(chosen);
  }
}
