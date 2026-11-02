package com.epsilon.prism.collect;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.CrnSeeds.Domain;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.prism.rows.Row;

/**
 * 1コホートの収集で全 run が共有する、変更されない設定と並行に更新する計数。抽選は全て run の種・コホートの種・半荘 ID・判断 ID のハッシュ
 * （{@link CrnSeeds}）で、実行順・同時進行数・ワーカー数に依らない。
 *
 * <p>判断の硬貨は進行ごとの硬貨の鍵（主経路は {@link #mainKey}、写しは {@link #copyKey}）から引く。分岐は硬貨を引かない。
 */
final class CollectContext {
  final long runSeed;
  final long iterationSeed;
  final PolicyService policy;

  final EpsilonUtilityProfile profile;

  /** 収集の率（評価は {@link Sampling#OFF}）。 */
  final Sampling sampling;

  /** ラベルの枠ごとに運の精算の帳簿（{@link EventLedger}）を付けるか（{@link Sampling#eventLedger}）。 */
  final boolean eventLedger;

  /** 全席の行動方策 b（null なら方策の σ から引く。評価）。 */
  final Behaviour behaviour;

  final CollectionMetrics metrics;
  final StageTimer timer;
  final CollectionDriver.SeatPolicies seating;
  final CollectionDriver.GameObserver observer;

  /**
   * 行（{@link Row}）の受け口（行を作らない評価は null）。null でなければ、経路ごとに {@link RowRecorder} を持ち、ラベル行と根の記録が行を
   * 作る。
   */
  final RowSink rows;

  /**
   * 運の網 Ω の標本の受け口（標本を集める収集だけ。ほかは null）。null でなければ、経路ごとに {@link LuckRecorder} を持ち、境界で復元した
   * 出来事から標本を引いて、局末の値を Y にして書く。
   */
  final LuckSampleSink luck;

  CollectContext(
      long runSeed,
      long iterationSeed,
      PolicyService policy,
      EpsilonUtilityProfile profile,
      Sampling sampling,
      Behaviour behaviour,
      CollectionMetrics metrics,
      StageTimer timer,
      CollectionDriver.SeatPolicies seating,
      CollectionDriver.GameObserver observer,
      RowSink rows,
      LuckSampleSink luck) {
    this.runSeed = runSeed;
    this.iterationSeed = iterationSeed;
    this.policy = policy;
    this.profile = profile;
    this.sampling = sampling == null ? Sampling.OFF : sampling;
    eventLedger = this.sampling.eventLedger();
    this.behaviour = behaviour;
    this.metrics = metrics;
    this.timer = timer;
    this.seating = seating;
    this.observer = observer;
    this.rows = rows;
    this.luck = luck;
  }

  /**
   * そろった行を受け口へ渡す（{@link RowDraft#row} で作る）。ドライバのスレッドだけが呼ぶ。
   *
   * @param values 合法手ごとの Y
   * @param ledger 手ごとの運の精算の帳簿（帳簿を付けない収集は null）
   * @param outcomes 手ごとの局末の 3 語
   */
  void write(RowDraft draft, float[] values, float[] ledger, long[] outcomes) {
    metrics.rowsWritten.increment();
    long started = System.nanoTime();
    rows.accept(draft.row(values, ledger, outcomes));
    timer.add(StageTimer.Stage.V2_ROWS, System.nanoTime() - started);
  }

  /** 半荘の行の用途。保留の割り当ては run 全体の種と半荘 ID だけで決め、反復をまたいで変えない。 */
  Row.Purpose purpose(long gameId) {
    return CrnSeeds.unit(CrnSeeds.derive(runSeed, Domain.HOLDOUT, gameId))
            < sampling.holdoutFraction()
        ? Row.Purpose.HOLDOUT
        : Row.Purpose.TRAIN;
  }

  /** 半荘の組（A・B、半々）。run 全体の種と半荘 ID だけで決める。 */
  Row.Group group(long gameId) {
    return CrnSeeds.unit(CrnSeeds.derive(runSeed, Domain.GROUP, gameId)) < 0.5
        ? Row.Group.A
        : Row.Group.B;
  }

  /** 主経路の硬貨の鍵。 */
  long mainKey(long gameId) {
    return CrnSeeds.mainKey(iterationSeed, gameId);
  }

  /** 主経路の判断 rootDecisionId から作った写しの硬貨の鍵。 */
  long copyKey(long gameId, long rootDecisionId) {
    return CrnSeeds.copyKey(iterationSeed, gameId, rootDecisionId);
  }

  /** 主経路の判断をラベル行にするか（率 ρ_c）。 */
  boolean label(long coinKey, long decisionId, DecisionClass c) {
    double rate = sampling.labelRate(c);
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.LABEL, decisionId) < rate;
  }

  /** 主経路の判断で中立ロールインを起こすか（率 β_c）。 */
  boolean rollIn(long coinKey, long decisionId, DecisionClass c) {
    double rate = sampling.rollinRate(c);
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.ROLLIN, decisionId) < rate;
  }

  /** 写しの中の逸脱した席の判断をラベル行にするか（率 ρ_br。上限は呼び出し側が数える）。 */
  boolean copyLabel(long coinKey, long decisionId) {
    double rate = sampling.rhoBranch();
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.COPY_LABEL, decisionId) < rate;
  }

  /** 候補集合に一様な 1 手を足すかの硬貨（ε_C と比べる）。 */
  double exploreCoin(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.EXPLORE, decisionId, 1);
  }

  /** 候補集合に足す一様な 1 手を選ぶ一様値。 */
  double exploreDraw(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.EXPLORE, decisionId, 2);
  }

  /** 逸脱 d ~ q(·|o) を引く一様値。 */
  double deviationCoin(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.DEVIATION, decisionId);
  }

  /** コホートの件数。 */
  CollectionMetrics.Snapshot counts() {
    return metrics.snapshot();
  }
}
