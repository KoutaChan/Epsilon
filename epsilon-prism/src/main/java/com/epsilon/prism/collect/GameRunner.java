package com.epsilon.prism.collect;

import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.engine.EngineDecisionPoint;
import com.epsilon.engine.EngineSelectionBuffer;
import com.epsilon.engine.GameEngine;
import com.epsilon.engine.GameStepResult;
import com.epsilon.engine.RoundSettlement;
import com.epsilon.engine.RoundTransition;
import com.epsilon.prism.core.CrnKeys;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.PrismRow;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 1本の半荘の進行（主経路）と、その判断から分かれた写しと分岐（{@link Role}）。
 *
 * <p>主経路（{@link Role#MAIN}）の判断ごと（合法手が双子だけでない判断。{@link LabelPlan#decide}）:
 *
 * <ol>
 *   <li>prepare（推論の前）: 類 c と予定（ラベル行か・ロールインか）を硬貨で決め、どちらかなら retain の問い合わせにする（方策が payload と p0 を書く）。
 *   <li>commit（推論の後、engine の commit の前）: 全席の手を共通乱数で選ぶ（{@link #choose}）。retain の判断では候補集合 C(o)（{@link
 *       CandidateSet#build}）を作り、
 *       <ul>
 *         <li>ラベル行なら {@link PendingLabel}（source MAIN）を作り、打った手の枠をこの局の targets に足し、C ∪ {played} のほかの手ごとに分岐
 *             （{@link Role#FORK}）を作る。
 *         <li>ロールインなら、b が貪欲の手（σ の argmax。双子を含む）を打った判断でだけ d ~ q（{@link RollIn#draw}）を引き、{@link
 *             RootRecord} を作って打った手の枠を targets に足す。d ≠ played なら写し（{@link Role#COPY}）を作り、写しの targets に d の枠を入れる
 *             （同じ判断のラベル行が d を測るなら、その d の枠も写しに入れて d の分岐を作らない）。床 ε_b で別の手を打った判断は根も写しも作らず、
 *             件数（floorRoots）だけ数える。根の y_main を π_t(o) の値にして、認証の乗り換えを「生徒の手 ≠ π_t(o)」と比べるため（床を外しただけの
 *             改善を生徒の手柄に数えない）。選別は手の抽選の共通乱数だけで決まり、以後の進行の乱数とは独立なので、残った根の y_main の期待値は
 *             Q_b(o, π_t(o))。
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
 * <p>測定器（{@link CandidateProbe}。probe-nano だけ）があれば、ラベル行は候補集合の代わりに {played, 測定器の手} の 2 手で作る（{@link #probe}）。
 *
 * <p>値: 局が終わると、この局の targets の全ての枠に局末の 4 席の値を渡す（{@link HandEnd}。ドライバが Φ を求めて {@link Pending#accept} を呼ぶ）。
 *
 * <p>共通乱数: 手の抽選は局のキーと席×種別類の序数（{@link CrnKeys}）で、写しと分岐は複製時に山と序数の進みを引き継ぐ。同じ手を打ち直した経路は
 * 主経路と同じ値になる（候補の間の CRN）。硬貨は進行の硬貨の鍵（主経路は {@link CollectContext#mainKey}、写しは {@link CollectContext#copyKey}）
 * と判断 ID のハッシュ。実行順・同時進行数・ワーカー数に依らない。判断 ID は応答できる席の数で枝ごとにずれる（他家の伏せた手牌に依る）ので、硬貨は
 * 席 i の観測だけの関数ではない。ただし鍵は山と独立なハッシュなので、隠れた状態 H を固定しても硬貨は一様で、採否の確率は H に依らない（{@link
 * com.epsilon.prism.core.CrnSeeds}）。
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

  private static final int MAX_POINTS = 3;

  private final CollectContext context;
  private final Role role;
  private final long gameId;
  private final GameEngine engine;
  private final CrnKeys.Cursor cursor;

  /** この半荘の行の用途と組。 */
  private final PrismRow.Purpose purpose;

  private final PrismRow.Group group;
  private final CollectionMetrics metrics;
  private final PolicyService[] seats;

  /** 進行の硬貨の鍵（主経路は mainKey、写しは copyKey、分岐は 0 で使わない）。 */
  private final long coinKey;

  /** 写しの逸脱した席（主経路と分岐は -1）。 */
  private final int deviator;

  /** 写しの根の主経路の判断 ID（主経路と分岐は -1）。 */
  private final long rootDecisionId;

  private final EngineSelectionBuffer selections = new EngineSelectionBuffer();
  private final PolicyRequest[] requests = new PolicyRequest[MAX_POINTS];
  private final LabelPlan[] plans = new LabelPlan[MAX_POINTS];
  private final int[] chosen = new int[MAX_POINTS];
  private final List<PolicyRequest> outstanding = new ArrayList<>(MAX_POINTS);
  private final List<HandEnd> handEnds = new ArrayList<>(1);

  /** この境界で作った写しと分岐（ドライバが引き取る）。 */
  private final List<GameRunner> spawned = new ArrayList<>(0);

  /** この局の値を受ける枠。写しと分岐は作られたときに親が入れた枠で始まる。主経路は局ごとに新しい一覧で始める。 */
  private List<Slot> targets;

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
    this.gameId = gameId;
    cursor = new CrnKeys.Cursor();
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
    handKey = CrnKeys.handKey(context.iterationSeed, gameId, 0);
    step = engine.stepHanchan();
  }

  /**
   * 写しか分岐。親の境界で複製した engine に差し替えた選択を commit した続き。序数（cursor）・局の root・局のキーは親の今の状態を引き継ぐ。
   *
   * @param role {@link Role#COPY} か {@link Role#FORK}
   * @param deviator 写しの逸脱した席（分岐は -1）
   * @param rootDecisionId 写しの根の判断 ID（分岐は -1）
   * @param slots この経路の局末の値を受ける枠
   */
  private GameRunner(
      GameRunner parent,
      Role role,
      GameEngine fork,
      EngineSelectionBuffer forked,
      int deviator,
      long rootDecisionId,
      List<Slot> slots) {
    if (role == Role.MAIN) throw new IllegalArgumentException("Spawned runners are copies or forks");
    context = parent.context;
    this.role = role;
    gameId = parent.gameId;
    cursor = parent.cursor.copy();
    root = parent.root;
    engine = fork;
    purpose = parent.purpose;
    group = parent.group;
    metrics = parent.metrics;
    seats = parent.seats;
    this.deviator = deviator;
    this.rootDecisionId = rootDecisionId;
    coinKey = role == Role.COPY ? context.copyKey(gameId, rootDecisionId) : 0L;
    targets = new ArrayList<>(slots);
    handKey = parent.handKey;
    handOrdinal = parent.handOrdinal;
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
    }
  }

  /**
   * 境界の各判断の方策を求める。合法手 1 つは強制手で問い合わせない（requests[index] = null、plans[index] = {@link LabelPlan#NONE}）。ほかの判断は
   * 種別・段階・脅威の数から類を決め（{@link com.epsilon.prism.core.DecisionClass#of}）、予定を決め（{@link LabelPlan#decide}。写しのラベル行の
   * 予約はここで copyLabels を数える）、{@link PolicyRequest}（retain = 予定の retain）を作る。inline の方策はその場で評価し、そうでなければ
   * outstanding に足す。件数（modelDecisions・役割別・類ごとの主経路の判断）もここで数える。非同期の問い合わせが残れば true。
   */
  private boolean prepare(List<EngineDecisionPoint> points) {
    outstanding.clear();
    GameState state = engine.getState();
    stage = Stage.of(state.remainingWallTiles());
    for (int index = 0; index < points.size(); index++) {
      EngineDecisionPoint point = points.get(index);
      List<Action> legal = point.legalActions();
      if (legal.size() == 1) {
        requests[index] = null;
        plans[index] = LabelPlan.NONE;
        metrics.forcedDecisions.increment();
        continue;
      }
      int seat = point.player();
      Kind kind = Kind.classify(point.kind(), legal);
      int threats = Threats.count(state, seat);
      DecisionClass decisionClass = DecisionClass.of(kind, stage, threats);
      metrics.modelDecisions.increment();
      switch (role) {
        case MAIN -> metrics.mainDecisions[decisionClass.ordinal()].increment();
        case COPY -> metrics.copyDecisions.increment();
        case FORK -> metrics.forkDecisions.increment();
      }
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
              legal,
              kind,
              decisionClass,
              plan.retain());
      requests[index] = request;
      PolicyService policy = seats[seat];
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
   * 全席の手を共通乱数で選び（{@link #choose}）、retain の判断を記録にし（{@link #record}）、engine を commit する。observer があれば主経路の
   * 判断を通知する（commit の前の state）。
   */
  private void commit(List<EngineDecisionPoint> points) {
    int count = points.size();
    selections.clear();
    for (int index = 0; index < count; index++) {
      EngineDecisionPoint point = points.get(index);
      int kindClass = CrnKeys.kindClass(point.kind());
      // 序数は強制手でも進める（主経路・写し・分岐で同じだけ進み、同じ局面では同じ鍵になる）
      int ordinal = cursor.next(point.player(), kindClass);
      PolicyRequest request = requests[index];
      chosen[index] = request == null ? 0 : choose(point, request, kindClass, ordinal);
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
    for (int index = 0; index < count; index++) if (plans[index].retain()) record(points, index);
    step = engine.commitDecisions(selections);
    Arrays.fill(requests, null);
    Arrays.fill(plans, LabelPlan.NONE);
    prepared = false;
  }

  /**
   * 手の選び。b（{@link Behaviour}: σ の貪欲に床 ε_b）があれば b から、なければ σ から共通乱数で引く（評価）。写しと分岐も同じ（写しの逸脱は
   * 作るときに差し替えるだけで、写しの中は b）。
   *
   * <pre>
   * CrnKeys.select(CrnKeys.decisionKey(handKey, 席, 種別類, 序数), legal, behaviour == null ? σ : behaviour.policy(σ, n), 0)
   * </pre>
   */
  private int choose(EngineDecisionPoint point, PolicyRequest request, int kindClass, int ordinal) {
    List<Action> legal = point.legalActions();
    Behaviour behaviour = context.behaviour;
    return CrnKeys.select(
        CrnKeys.decisionKey(handKey, point.player(), kindClass, ordinal),
        legal,
        behaviour == null ? request.sigma : behaviour.policy(request.sigma, legal.size()),
        0);
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
    if (request.scores == null || request.prior == null || request.payload == null)
      throw new IllegalStateException(
          "Retained decisions need s, p0 and the payload from a network policy (inline policies"
              + " cannot collect rows): "
              + this);
    if (context.probe != null) {
      probe(points, index);
      return;
    }
    List<Action> legal = point.legalActions();
    int played = chosen[index];
    long id = point.id();
    int classIndex = plan.decisionClass().ordinal();
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
    if (candidates.explored()) metrics.explored.increment();
    byte[] support = candidates.supportBytes();
    int[] measured = candidates.measured();

    PendingLabel label = null;
    if (plan.label()) {
      boolean main = role == Role.MAIN;
      label =
          new PendingLabel(
              context,
              draft(
                  point,
                  request,
                  plan,
                  main ? PrismRow.Source.MAIN : PrismRow.Source.ROLLIN,
                  played,
                  support,
                  main ? -1 : rootDecisionId),
              measured.length);
      // 打った手はこの経路（主経路か写し）の局末の値がただで付く
      targets.add(new Slot(label, played));
      if (main) metrics.labelRows[classIndex].increment();
      else metrics.copyLabelRows[classIndex].increment();
      metrics.measured.add(measured.length);
      if (!candidates.playedInside(played)) metrics.playedOutside.increment();
    }

    // 写しが値を測るラベル行の手（同じ判断のラベル行が d を測るなら d の分岐は作らない）
    int coveredByCopy = -1;
    if (deviation != null) {
      int d = deviation.action();
      var rootRecord =
          new RootRecord(
              context,
              draft(point, request, plan, PrismRow.Source.ROOT, played, support, -1),
              deviation,
              context.sampling.rollinRate(plan.decisionClass()));
      targets.add(new Slot(rootRecord, played));
      metrics.roots[classIndex].increment();
      if (rootRecord.sameAsPlayed()) {
        // d = played: 写しは主経路と同じ過程なので作らない（y_br = y_main）
        metrics.sameDeviations.increment();
      } else {
        List<Slot> slots = new ArrayList<>(2);
        slots.add(new Slot(rootRecord, d));
        if (label != null && contains(measured, d)) {
          slots.add(new Slot(label, d));
          coveredByCopy = d;
        }
        spawn(points, index, d, Role.COPY, slots);
        metrics.copies.increment();
      }
    }

    if (label != null) {
      for (int a : measured) {
        if (a == played || a == coveredByCopy) continue;
        spawn(points, index, a, Role.FORK, List.of(new Slot(label, a)));
        metrics.forks.increment();
      }
    }
  }

  /**
   * 測定器つきの収集（{@link CandidateProbe}）の記録。ラベル行に予約した主経路の判断だけを扱う（ロールインの率は 0。{@link
   * CollectionDriver#collect(CollectionDriver.Cohort, PolicyService, HandValue, com.epsilon.prism.rows.RowSink,
   * CandidateProbe)}）。床 ε_b の手を打った判断は数えるだけ、測定器の手が打った手か双子なら一致を数えるだけで、違えば測る手を {played, 測定器の手}
   * の 2 手にしたラベル行（MAIN）を作り、打った手の枠を targets に、測定器の手の分岐を作る。根と写しは作らない。時間は {@link
   * StageTimer.Stage#PROBE}。
   */
  private void probe(List<EngineDecisionPoint> points, int index) {
    EngineDecisionPoint point = points.get(index);
    PolicyRequest request = requests[index];
    LabelPlan plan = plans[index];
    if (role != Role.MAIN || plan.rollIn())
      throw new IllegalStateException("Probed collection labels main decisions only: " + this);
    List<Action> legal = point.legalActions();
    int played = chosen[index];
    CandidateProbe probe = context.probe;
    if (!Twins.same(legal, played, greedy(request.sigma))) {
      probe.counted(plan.decisionClass(), CandidateProbe.Outcome.FLOOR);
      return;
    }
    long started = System.nanoTime();
    int other = probe.choose(request.observation(), legal, played);
    context.timer.add(StageTimer.Stage.PROBE, System.nanoTime() - started);
    if (other < 0 || other >= legal.size())
      throw new IllegalStateException("Probe chose " + other + " of " + legal.size());
    if (Twins.same(legal, played, other)) {
      probe.counted(plan.decisionClass(), CandidateProbe.Outcome.AGREED);
      return;
    }
    probe.counted(plan.decisionClass(), CandidateProbe.Outcome.DISAGREED);
    byte[] support = {(byte) Math.min(played, other), (byte) Math.max(played, other)};
    var label =
        new PendingLabel(
            context,
            draft(point, request, plan, PrismRow.Source.MAIN, played, support, -1),
            support.length);
    targets.add(new Slot(label, played));
    metrics.labelRows[plan.decisionClass().ordinal()].increment();
    metrics.measured.add(support.length);
    spawn(points, index, other, Role.FORK, List.of(new Slot(label, other)));
    metrics.forks.increment();
  }

  /** 判断の時点で確定する行の値以外の部分。 */
  private RowDraft draft(
      EngineDecisionPoint point,
      PolicyRequest request,
      LabelPlan plan,
      PrismRow.Source source,
      int played,
      byte[] support,
      long root) {
    return new RowDraft(
        context.iteration,
        gameId,
        point.id(),
        point.player(),
        request.kind(),
        stage,
        plan.decisionClass(),
        purpose,
        group,
        source,
        played,
        point.legalActions().size(),
        request.prior,
        support,
        root,
        request.payload);
  }

  private static boolean contains(int[] actions, int action) {
    for (int a : actions) if (a == action) return true;
    return false;
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
    EngineDecisionPoint point = points.get(index);
    spawned.add(
        role == Role.COPY
            ? new GameRunner(this, Role.COPY, fork, forked, point.player(), point.id(), slots)
            : new GameRunner(this, Role.FORK, fork, forked, -1, -1, slots));
  }

  /** 局の終わり。枠があれば局末の値を待つ先として渡し、半荘が続くなら（主経路は）次局へ進む。 */
  private void endHand(RoundSettlement settlement, boolean hanchanEnded) {
    int[] scores = settlement.snapshotFinalScores();
    List<Slot> slots = targets;
    if (role == Role.MAIN) {
      metrics.hands.increment();
      if (context.observer != null)
        context.observer.handEnded(gameId, engine.getState(), settlement.result());
    }
    if (hanchanEnded) {
      if (role == Role.MAIN) {
        if (context.observer != null) context.observer.gameEnded(gameId, scores);
        metrics.games.increment();
      }
      if (!slots.isEmpty())
        handEnds.add(new HandEnd(null, scores, HandValue.terminal(scores, context.profile), slots));
      return;
    }
    var next = (RoundTransition.NextRound) settlement.transition();
    RootState nextRoot = root.next(next, scores);
    if (!slots.isEmpty()) handEnds.add(new HandEnd(nextRoot.publicPrefix(), scores, null, slots));
    if (role != Role.MAIN) return;
    targets = new ArrayList<>();
    root = nextRoot;
    handOrdinal++;
    handKey = CrnKeys.handKey(context.iterationSeed, gameId, handOrdinal);
    cursor.reset();
    step = engine.stepHanchan();
  }

  // ドライバとの受け渡し。

  PolicyService policyFor(PolicyRequest request) {
    return seats[request.seat()];
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
