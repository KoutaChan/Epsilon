package com.epsilon.prism.collect;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.play.Run;
import com.epsilon.agent.play.RunHooks;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.DeviationTarget;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.core.Stage;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.core.Twins;
import com.epsilon.prism.luck.LedgerColumn;
import com.epsilon.prism.luck.LedgerTracker;
import com.epsilon.prism.luck.UraExpectation;
import com.epsilon.prism.luck.UraValues;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * 収集する経路 1 本の、PRISM 側の処理と状態。経路は半荘の主経路（{@link Role#MAIN}）か、主経路の判断から分かれた写しか分岐（{@link Role}）。
 * 経路を進めるのは卓（{@link Run}）で、卓は全席の手を行動方策 b から共通乱数で引く。引いた手が同値な打牌なら、組の中で切る方
 * （{@link TwinDiscard}）を打つ（{@link #play}。乱数の引き方は変えない）。
 *
 * <p>主経路では、同値な打牌をまとめても候補が 2 つ以上残る判断ごとに、次のことをする（{@link LabelPlan#main}）。
 *
 * <ol>
 *   <li>推論の前: 判断のクラス c と、判断する席の手の形を読む。ラベル行にするか、ロールインするかを乱数で決め、どちらかなら行を作る判断として
 *       推論を要求する（方策が p0 も書く）。ラベル行の率はクラスと手の形で、ロールインの率はクラスで決まる（{@link Sampling}）。
 *   <li>手を選んだ後、エンジンを進める前: 行を作る判断では候補集合 C(o) を作る（{@link CandidateSet#build}）。候補と逸脱 d は、同値な
 *       打牌の組ごとに 1 つの手で選ぶ（{@link CandidateSet}・{@link RollIn}）。分岐と写しで打つ手と行に書く手は、組の中で切る方の手に替える。
 *       <ul>
 *         <li>ラベル行なら {@link PendingLabel}（source は MAIN）を作り、打った手の受け取り先をこの局の targets に足す。C ∪ {played} の
 *             ほかの手ごとに分岐（{@link Role#FORK}）を作る。
 *         <li>ロールインは、b が貪欲の手（σ の argmax。同値な手も含む）を打った判断だけで行う。根の判断のクラスで決まる
 *             {@link RollInRule} の台と ε_q で d ~ q を引き（{@link RollIn#draw}）、{@link RootRecord} を作って打った手の受け取り先を targets
 *             に足す。d ≠ played なら写し（{@link Role#COPY}）を作り、写しの targets に d の受け取り先を入れる。同じ判断のラベル行が d を
 *             測るなら、ラベル行の d の受け取り先も写しに入れ、d の分岐は作らない。
 *         <li>b は一様分布を混ぜるので、貪欲でない手を打つことがある。そうした判断では根も写しも作らず、件数（floorRoots）
 *             だけを数える。根の y_main を π_t(o) の値にするためで、一様分布から引いた手の値を根に混ぜない。この選り分けは手を引く共通乱数だけで決まり、以後の進行の
 *             乱数とは独立なので、残った根の y_main の期待値は Q_b(o, π_t(o)) になる。
 *       </ul>
 *   <li>主経路は逸脱しない。
 * </ol>
 *
 * <p>写し（{@link Role#COPY}）は、主経路の判断の時点でエンジンを写し、逸脱した席の手だけを d に差し替えて進める。その後は局末まで全席が b で
 * 打って止まる。逸脱した席の後の判断は、根の判断のクラスで決まる {@link RollInRule} に従い、その判断のクラスの率 ρ_br で当たり、写しごとの
 * 上限に達していなければラベル行にして、写しの中から分岐を作る。この行の source は ROLLIN で、rootDecisionId は写しの根の判断 ID。ほかの席の判断は行にせず、ロールインもしない。
 *
 * <p>分岐（{@link Role#FORK}）は、主経路か写しの判断の時点でエンジンを写し、判断する席の手だけを候補 a に差し替えて進める。その後は局末まで
 * 全席が b で打って止まる。行も分岐も作らず、局末の値を親の記録の受け取り先 1 つに渡すだけの測定になる。
 *
 * <p>局が終わると、この局の targets の全ての受け取り先に、局末の席ごとの値を渡す（{@link HandEnd}）。値は局末を作った進行がその場で求め、
 * ドライバが {@link CollectContext#accept} で渡す。
 *
 * <p>運の精算は {@link CollectContext#ledger} があるときだけ行う。経路ごとに {@link LedgerTracker} を 1 つ持ち、この局の targets の
 * 受け取り先の帳簿に次のものを足す。判断の時点ごとに、前の時点からのツモと槓ドラの表示。強制でない判断で手を選んだ後に、方策の偶然。局末に
 * 裏ドラ。受け取り先は判断の後に作るので、o より前のイベントは数えない。写しと分岐は親の計算器を写して始める。帳簿を付けない収集では、計算も
 * 乱数の消費も手も変わらない。
 *
 * <p>行の記録のため、経路ごとに {@link RowRecorder} を 1 つ持つ。行を作る判断で、推論の要求が運ぶ O の写しと Oracle を作る
 * （{@link RowRecorder#decision}）。判断の時点と局末で聴牌への到達を追い、受け取り先のある局末で局末の事実を {@link HandEnd} に載せる。
 * 写しと分岐は親の記録を写して始める。状態を読むだけなので、進行と乱数は変わらない。
 *
 * <p>採否の乱数は、進行の乱数のキー（主経路は {@link CollectContext#mainKey}、写しは {@link CollectContext#copyKey}）と判断 ID の
 * ハッシュで決まる。実行順・同時に進める数・ワーカー数には依らない。判断 ID は応答できる席の数で枝ごとにずれ、その数は他家の伏せた手牌で
 * 決まるので、この乱数は席 i の観測だけの関数ではない。ただしキーは山と独立なハッシュなので、隠れた状態 H を固定しても乱数は一様で、採否の
 * 確率は H に依らない（{@link CrnSeeds}）。
 */
final class CollectionPath implements RunHooks<PolicyRequest> {

  private final CollectContext context;
  private final Role role;

  /** 推論を要求するときに数える判断の件数。経路を進めた後に metrics へ足す。 */
  private final DecisionCounts decisionCounts;

  /** この半荘の行の用途。 */
  private final Row.Purpose purpose;

  private final CollectionMetrics metrics;

  /** 進行の乱数のキー。主経路は mainKey、写しは copyKey。分岐は使わないので 0。 */
  private final long coinKey;

  /** 写しで逸脱した席。主経路と分岐は -1。 */
  private final int deviator;

  /** 写しの根になった主経路の判断 ID。主経路と分岐は -1。 */
  private final long rootDecisionId;

  /** 写しの根の判断のクラス。主経路と分岐は null。 */
  private final DecisionClass rootClass;

  /** 写しの根の判断のクラスで使う、中立ロールインの決まり。主経路と分岐は null。 */
  private final RollInRule copyRule;

  /** 判断ごとの予定。強制の判断と行を作らない判断は {@link LabelPlan#NONE}。 */
  private final LabelPlan[] plans;

  private final List<HandEnd> handEnds = new ArrayList<>(1);

  /** この局の値の受け取り先。写しと分岐は、作られたときに親が入れた受け取り先で始まる。主経路は局ごとに新しい一覧で始める。 */
  private List<Slot> targets;

  /** 運の精算の帳簿の計算器。帳簿を付けない収集では null。 */
  private final LedgerTracker ledger;

  private final RowRecorder rows;

  /** 半荘の始めから、各局が始まったときの卓。Φ が局の履歴を読むときだけ持ち、ほかは null。写しと分岐は親の履歴の写しを持つ。 */
  private final RoundHistory history;

  /** いま準備している判断の時点の段階。同じ時点の判断では全て同じ。 */
  private Stage stage;

  /** この写しでこれまでに予約したラベル行の数。 */
  private int copyLabels;

  /** 半荘の主経路。 */
  CollectionPath(CollectContext context, long gameId) {
    this.context = context;
    role = Role.MAIN;
    decisionCounts = new DecisionCounts(Role.MAIN, null);
    if (context.phi.needsHistory()) {
      history = new RoundHistory(context.variant);
      history.add(new TableState(context.variant));
    } else {
      history = null;
    }
    purpose = context.purpose(gameId);
    metrics = context.metrics;
    coinKey = context.mainKey(gameId);
    deviator = -1;
    rootDecisionId = -1;
    rootClass = null;
    copyRule = null;
    plans = plans(context);
    targets = new ArrayList<>();
    ledger = context.ledger == null ? null : new LedgerTracker(context.ledger);
    rows = new RowRecorder(context.variant);
  }

  /**
   * 写しか分岐を作る。行の用途と組、Φ の履歴、帳簿の計算器、行の記録は、親の今の状態（親の判断の時点）から引き継ぐ。
   *
   * @param role {@link Role#COPY} か {@link Role#FORK}
   * @param seat 経路を分けた親の判断の席（写しでは逸脱した席）
   * @param decisionId 経路を分けた親の判断 ID（写しでは根の判断 ID）
   * @param decisionClass 経路を分けた親の判断のクラス（写しでは根の判断のクラス）
   * @param slots この経路の局末の値の受け取り先
   */
  private CollectionPath(
      CollectionPath parent,
      Role role,
      long gameId,
      int seat,
      long decisionId,
      DecisionClass decisionClass,
      List<Slot> slots) {
    if (role == Role.MAIN) throw new IllegalArgumentException("Spawned paths are copies or forks");
    context = parent.context;
    this.role = role;
    boolean copy = role == Role.COPY;
    rootClass = copy ? decisionClass : null;
    copyRule = copy ? context.sampling.rollIn(decisionClass) : null;
    decisionCounts = new DecisionCounts(role, rootClass);
    history = parent.history == null ? null : parent.history.copy();
    purpose = parent.purpose;
    metrics = parent.metrics;
    deviator = copy ? seat : -1;
    rootDecisionId = copy ? decisionId : -1;
    coinKey = copy ? context.copyKey(gameId, rootDecisionId) : 0L;
    plans = plans(context);
    targets = new ArrayList<>(slots);
    // 計算器は、親が最後に読んだ判断の時点（o の時点）から写す。o の打牌の後の表示とツモは、この経路の最初の判断の時点で数える
    ledger = parent.ledger == null ? null : new LedgerTracker(parent.ledger);
    // 行の記録の聴牌への到達も、親の o の時点から写す。o の打牌は、この経路の最初の判断の時点で数える
    rows = new RowRecorder(parent.rows);
  }

  private static LabelPlan[] plans(CollectContext context) {
    LabelPlan[] plans = new LabelPlan[context.variant.seats() - 1];
    Arrays.fill(plans, LabelPlan.NONE);
    return plans;
  }

  /** 前の判断の時点からのイベントを帳簿と行の記録に渡し、この時点の段階を決める。 */
  @Override
  public void boundary(Run<PolicyRequest> run, Decisions decisions, RuleWorkspace ws) {
    FullView full = run.full();
    if (ledger != null) ledger.observe(full, targets, ws);
    rows.boundary(full, targets);
    stage = Stage.of(context.variant, full.seat(0).liveLeft());
  }

  /**
   * 種別・段階・脅威の数・局の番号から判断のクラスを決め（{@link DecisionClass#of}）、予定を決める（{@link #plan}）。予定が行を作るなら、
   * 行を作る判断として推論を要求する。
   */
  @Override
  public PolicyRequest query(
      Run<PolicyRequest> run,
      Decisions decisions,
      int slot,
      SeatView view,
      int admission,
      ObservationBatch records,
      int record,
      Observation observation) {
    int n = decisions.legalCount(slot);
    int[] legal = new int[n];
    for (int action = 0; action < n; action++) legal[action] = decisions.legal(slot, action);
    int seat = decisions.seat(slot);
    Kind kind = Kind.classify(decisions.kind(slot), legal);
    int threats = Threats.count(view);
    DecisionClass decisionClass = DecisionClass.of(context.variant, kind, stage, threats, view.round());
    LabelPlan plan = plan(decisions.id(slot), seat, decisionClass, threats, legal, view);
    plans[slot] = plan;
    return new PolicyRequest(
        run, run.game().wall(), slot, kind, decisionClass, plan.retain(), admission, records, record, observation);
  }

  /**
   * 判断の件数を数え、予定を決める（{@link LabelPlan}）。件数は経路の種類ごとに数え、主経路は手の形とクラスごとに、写しは逸脱した席の判断も
   * 別に数える。予定を作るのは、主経路と写しの逸脱した席の判断のうち、同値な打牌をまとめても候補が 2 つ以上残るものだけ。写しのラベル行は
   * 予約した時点で上限に数える。写しの進行は決定的なので、どの判断が当たるかも実行順に依らない。
   */
  private LabelPlan plan(
      long decisionId, int seat, DecisionClass decisionClass, int threats, int[] legal, SeatView view) {
    return switch (role) {
      case MAIN -> {
        Openness hand = Openness.of(view);
        decisionCounts.countMain(decisionClass, hand);
        yield Twins.branchable(legal)
            ? LabelPlan.main(context, coinKey, decisionId, decisionClass, hand, threats)
            : LabelPlan.NONE;
      }
      case COPY -> {
        decisionCounts.countModel();
        if (seat != deviator || !Twins.branchable(legal)) yield LabelPlan.NONE;
        decisionCounts.countDeviator();
        LabelPlan plan =
            LabelPlan.copy(
                context, coinKey, decisionId, copyRule, copyLabels, decisionClass, Openness.of(view), threats);
        if (plan.label()) copyLabels++;
        yield plan;
      }
      case FORK -> {
        decisionCounts.countModel();
        yield LabelPlan.NONE;
      }
    };
  }

  /** 引いた手が同値な打牌なら、組の中で切る方の手を打つ。 */
  @Override
  public int play(PolicyRequest query, int drawn) {
    return context.twinDiscard.play(query.legal(), drawn);
  }

  /**
   * 方策の偶然を帳簿に足し、行を作る判断を記録する（{@link #record}）。帳簿に足すのは記録より前なので、この時点で作る o の受け取り先には
   * 足さない。写しと分岐は記録の中で作る。
   */
  @Override
  public void decided(Run<PolicyRequest> run, Decisions decisions, int[] chosen, RuleWorkspace ws) {
    int count = decisions.size();
    for (int slot = 0; slot < count; slot++) {
      PolicyRequest request = run.query(slot);
      if (request == null) continue;
      if (ledger != null)
        ledger.policy(
            decisions.seat(slot), request.scores, run.distribution(slot), chosen[slot], request.legalCount(), targets);
      if (role == Role.MAIN
          && request.sigma != null
          && !Twins.same(request.legal(), chosen[slot], greedy(request.sigma)))
        metrics.mainNonGreedy.increment();
    }
    for (int slot = 0; slot < count; slot++) {
      if (plans[slot].retain()) record(run, decisions, chosen, slot, ws);
    }
    Arrays.fill(plans, LabelPlan.NONE);
  }

  /**
   * 行を作る判断を 1 つ記録する。局面はエンジンを進める前のもの。候補集合を 1 度だけ作り（played = chosen[slot]、ε_C の乱数はこの進行の
   * 乱数のキーから引く）、予定のラベル行とロールインを {@link CollectionPath} の説明どおりに作る。件数（labelRows・copyLabelRows・roots・
   * floorRoots・copies・sameDeviations・forks・measured・explored・playedOutside）もここで数える。根の記録の候補集合のフィールドには q の台を
   * 書く。打った手のほかから d を引く決まりでは、打った手を除いた集合になる。かかった時間は {@link CollectStage#CANDIDATES} に足す。
   */
  private void record(Run<PolicyRequest> run, Decisions decisions, int[] chosen, int slot, RuleWorkspace ws) {
    PolicyRequest request = run.query(slot);
    LabelPlan plan = plans[slot];
    if (request.scores == null || request.prior == null)
      throw new IllegalStateException("Retained decisions need s and p0 from the policy: " + describe(run));
    int[] legal = request.legal();
    int played = chosen[slot];
    int seat = decisions.seat(slot);
    long id = decisions.id(slot);
    int classIndex = plan.decisionClass().ordinal();
    boolean exploring = role == Role.MAIN && context.behaviour != null && context.behaviour.temperature() > 0;
    List<Slot> playedSlots = new ArrayList<>(2);
    long started = System.nanoTime();
    CandidateSet candidates =
        context.sampling.measuresAll(plan.decisionClass())
            ? CandidateSet.allActions(legal, played)
            : CandidateSet.build(
                run.read(request),
                legal,
                request.scores,
                request.prior,
                plan.threats(),
                played,
                context.sampling.maxCandidates(),
                context.sampling.epsilonC(),
                context.exploreCoin(coinKey, id),
                context.exploreDraw(coinKey, id));
    // ロールインは b が貪欲の手を打った判断だけで行う。一様分布から出た手を打った判断では根を作らない（クラスの Javadoc）
    boolean rollIn = plan.rollIn() && Twins.same(legal, played, greedy(request.sigma));
    if (plan.rollIn() && !rollIn) metrics.floorRoots.increment();
    RollInRule rule = rollIn ? context.sampling.rollIn(plan.decisionClass()) : null;
    // 逸脱の台は同値な打牌の組を番号の最も小さい手で表す（打った手の番号で順位の同点を割らないので、切り方に依らず同じ組を引く）
    int[] deviationSupport =
        rollIn
            ? RollIn.support(Twins.lowest(legal, candidates.support()), Twins.lowest(legal, played), rule.target())
            : null;
    RollIn.Deviation deviation =
        rollIn
            ? RollIn.draw(
                deviationSupport, request.scores, request.prior, rule.epsilonQ(), context.deviationCoin(coinKey, id))
            : null;
    context.timer.add(CollectStage.CANDIDATES, System.nanoTime() - started);
    // 候補と逸脱は同値な打牌の組ごとに 1 つの手で選んだ。分岐と写しで打つ手と行に書く手は、組の中で切る方の手に替える（選んだ組は変えない）
    TwinDiscard twins = context.twinDiscard;
    byte[] support = CandidateSet.bytes(twins.play(legal, candidates.support()));
    int[] measured = twins.play(legal, candidates.measured());
    boolean labelled = plan.label();
    if (candidates.explored()) metrics.explored.increment();
    // 行を作る判断だけ、O の写しと Oracle を記録する
    RowRecorder.Decision recorded =
        labelled || deviation != null ? rows.decision(request, run.read(request), run.full(), ws) : null;
    PendingLabel label = null;
    if (labelled) {
      boolean main = role == Role.MAIN;
      label =
          new PendingLabel(
              context,
              draft(
                  run,
                  seat,
                  id,
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
      // 打った手には、この経路（主経路か写し）の局末の値がそのまま付く。主経路が探索するとき（Behaviour#explore）は、その後を b で打たないので、
      // 打った手も分岐で測り、その後を b で打つ。主経路の値を使うと打った手だけその後の打ち方が違い、差が片側にずれ、候補の間の共通乱数も切れる
      if (exploring) playedSlots.add(slot(label, played));
      else targets.add(slot(label, played));
      if (main) metrics.labelRows[CollectionMetrics.byHand(plan.hand(), plan.decisionClass())].increment();
      else metrics.copyLabelRows[CollectionMetrics.byRoot(rootClass, plan.decisionClass())].increment();
      metrics.measured.add(measured.length);
      if (!candidates.playedInside(played)) metrics.playedOutside.increment();
    }

    // ラベル行の手のうち、値を写しで測る手。同じ判断のラベル行が d を測るなら、d の分岐は作らない
    int coveredByCopy = -1;
    if (deviation != null) {
      int d = twins.play(legal, deviation.action());
      var rootRecord =
          new RootRecord(
              context,
              draft(
                  run,
                  seat,
                  id,
                  request,
                  plan,
                  recorded,
                  Row.Source.ROOT,
                  played,
                  rule.target() == DeviationTarget.ANY ? support : CandidateSet.bytes(twins.play(legal, deviationSupport)),
                  d,
                  deviation.probability(),
                  (float) (rule.rate() * deviation.probability()),
                  -1));
      if (exploring) playedSlots.add(slot(rootRecord, played));
      else targets.add(slot(rootRecord, played));
      metrics.roots[classIndex].increment();
      if (rootRecord.sameAsPlayed()) {
        // d = played なら、写しは主経路と同じ過程なので作らない（y_br = y_main）
        metrics.sameDeviations[classIndex].increment();
      } else {
        List<Slot> slots = new ArrayList<>(2);
        slots.add(slot(rootRecord, d));
        if (label != null && contains(measured, d)) {
          slots.add(slot(label, d));
          coveredByCopy = d;
        }
        spawn(run, decisions, slot, d, Role.COPY, slots);
        metrics.copies[classIndex].increment();
      }
    }

    if (label != null) {
      for (int a : measured) {
        if (a == played || a == coveredByCopy) continue;
        spawn(run, decisions, slot, a, Role.FORK, List.of(slot(label, a)));
        metrics.forks.increment();
      }
    }
    if (!playedSlots.isEmpty()) {
      spawn(run, decisions, slot, played, Role.FORK, playedSlots);
      metrics.forks.increment();
    }
  }

  /**
   * 行の下書きを作る。ラベル行では逸脱を −1、q と傾向スコアを NaN にする。根の記録では d、q(d|o)、β_c(o)·q(d|o) を入れる。
   *
   * @param recorded この判断の行の記録
   * @param root ROLLIN は写しの根の判断 ID、ほかは −1
   */
  private RowDraft draft(
      Run<PolicyRequest> run,
      int seat,
      long decisionId,
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
        run.game().wall(),
        decisionId,
        seat,
        plan.decisionClass(),
        purpose,
        source,
        played,
        request.legalCount(),
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

  /** 手 {@code action} を打った経路の局末の値を {@code target} に渡す受け取り先。帳簿を付ける収集なら空の帳簿を付ける。 */
  private Slot slot(Pending target, int action) {
    return new Slot(target, action, context.ledger == null ? null : new double[LedgerColumn.COUNT]);
  }

  /** 方策の貪欲の手。σ が最大の手で、同点なら番号の小さい手。ネットワークの方策の σ は one-hot。 */
  private static int greedy(float[] sigma) {
    int best = 0;
    for (int a = 1; a < sigma.length; a++) if (sigma[a] > sigma[best]) best = a;
    return best;
  }

  /**
   * 1 つの判断の席の手だけを差し替えた写しか分岐を作る。ほかの判断は、行動方策 b から引いた手を打つ（{@link Run#behaviourChoice}）。
   * 主経路が探索で引いた手は引き継がない。
   *
   * @param slot 手を差し替える判断の位置
   * @param action 差し替える手
   * @param role {@link Role#COPY} か {@link Role#FORK}。写しでは、その判断の席が逸脱した席に、その判断の ID とクラスが根の判断 ID と
   *     クラスになる
   * @param slots 作った経路の局末の値の受け取り先
   */
  private void spawn(Run<PolicyRequest> run, Decisions decisions, int slot, int action, Role role, List<Slot> slots) {
    int[] choices = new int[decisions.size()];
    for (int other = 0; other < choices.length; other++)
      choices[other] = other == slot ? action : run.behaviourChoice(other);
    run.fork(
        choices,
        new CollectionPath(
            this,
            role,
            run.game().wall(),
            decisions.seat(slot),
            decisions.id(slot),
            plans[slot].decisionClass(),
            slots));
  }

  /**
   * 局の終わり。受け取り先があれば、局末の値と局末の事実をまとめて渡す。主経路で半荘が続くなら、次局の記録を始める。裏ドラの項と局末の
   * 事実は、次局を配る前の局末の局面で作る。局末の値は裏ドラの扱いに従って求め（{@link UraValues}）、裏ドラを期待値で求めた局末なら
   * 受け取り先の全てをこの局の行として監査に数える。
   */
  @Override
  public void roundEnded(Run<PolicyRequest> run, RoundEnd end, boolean gameEnded) {
    FullView full = run.full();
    if (ledger != null && !targets.isEmpty()) ledger.handEnd(full, end, targets);
    HandEndFacts facts = !targets.isEmpty() ? rows.handEnd(full, end, targets) : null;
    List<Slot> slots = targets;
    if (!slots.isEmpty()) {
      // 局末の値は、局末を作ったこのスレッドがその場で求める
      double[] values = new double[context.variant.seats()];
      UraExpectation ura =
          context.ura.handEnd(full, end, run.handStart(), history, run.game().wall(), run.handOrdinal(), values);
      context.ura.rows(ura, run.game().wall(), run.handOrdinal(), slots);
      handEnds.add(new HandEnd(values, slots, facts));
    }
    if (gameEnded || role != Role.MAIN) return;
    targets = new ArrayList<>();
    if (ledger != null) ledger.reset();
    rows.reset();
    if (history != null) history.add(end.next());
  }

  /** 進行にかかった時間を経路の種類ごとに足し、経路に貯めた件数と、帳簿・行の記録の計測も足す。 */
  @Override
  public void advanced(Run<PolicyRequest> run, long nanos) {
    context.timer.add(
        switch (role) {
          case MAIN -> CollectStage.MAIN_ADVANCE;
          case COPY -> CollectStage.COPY_ADVANCE;
          case FORK -> CollectStage.FORK_ADVANCE;
        },
        nanos);
    decisionCounts.flush(metrics);
    if (ledger != null) ledger.flush(context.timer, CollectStage.LEDGER);
    rows.flush(context.timer);
  }

  /** この経路が作った局末を、ドライバのスレッドで受け取り先に渡す。 */
  @Override
  public void settle(Run<PolicyRequest> run) {
    for (HandEnd end : handEnds) context.accept(end);
    handEnds.clear();
  }

  /** 診断用の文字列。 */
  private String describe(Run<PolicyRequest> run) {
    return role
        + " "
        + run
        + " deviator="
        + deviator
        + " root="
        + rootDecisionId
        + " copyLabels="
        + copyLabels
        + " purpose="
        + purpose;
  }
}
