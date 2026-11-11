package com.epsilon.prism.collect;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.play.Behaviour;
import com.epsilon.agent.play.Run;
import com.epsilon.agent.play.RunHooks;
import com.epsilon.agent.play.TableHooks;
import com.epsilon.grp.Phi;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.HandEndFacts;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.CrnSeeds.Domain;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.luck.LedgerRecording;
import com.epsilon.prism.luck.LedgerTracker;
import com.epsilon.prism.luck.UraValues;
import com.epsilon.prism.noise.UraSettings;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.rows.Row;
import java.util.List;

/**
 * 1 コホートを集めるときに、卓のコアに載せる PRISM の処理（{@link TableHooks}）。全ての経路が共有する変わらない設定と、並行に更新する
 * 計数を持ち、主経路ごとに経路の処理（{@link CollectionPath}）を作る。抽選は全て、run の種・コホートの種・半荘 ID・判断 ID のハッシュ
 * （{@link CrnSeeds}）で決める。そのため結果は、実行の順・同時に進める数・ワーカーの数に依らない。
 *
 * <p>判断の乱数は、進行ごとの乱数のキーから引く。主経路のキーは {@link #mainKey}、写しのキーは {@link #copyKey}。分岐は乱数を引かない。
 */
final class CollectContext implements TableHooks<PolicyRequest> {

  /** 書き終えていない行がこれを超えたら新しい半荘を始めない。 */
  private static final long ROW_BACKLOG = 500_000;

  final Variant variant;
  final long runSeed;
  final long iterationSeed;

  /** 局末の値 Φ。局末になった進行が、{@link #ura} を通してその場で呼ぶ。半荘末の効用も、Φ と同じ効用で決める。 */
  final Phi phi;

  final Sampling sampling;

  /** ラベルの値の受け取り先ごとに付ける、運の精算の帳簿（{@link LedgerTracker}）。帳簿を付けない収集では null。 */
  final LedgerRecording ledger;

  /** 局末の値を求める（裏ドラの扱いとその監査も行う）。局末になった進行のスレッドが呼ぶ。 */
  final UraValues ura;

  /** 全席の行動方策 b。 */
  final Behaviour behaviour;

  /** 同値な打牌のうち、全ての経路で切る方。 */
  final TwinDiscard twinDiscard;

  final CollectionMetrics metrics;
  final StageTimer<CollectStage> timer;

  /** 行（{@link Row}）の受け取り先。経路ごとに {@link RowRecorder} を持ち、ラベル行と根の記録が行を作る。 */
  final RowSink rows;

  /** 局末の 3 ワードを書くエンコーダー。ドライバのスレッドだけが使う。 */
  private final OutcomeCodec outcomes;

  CollectContext(
      Variant variant,
      long runSeed,
      long iterationSeed,
      Phi phi,
      Sampling sampling,
      LedgerRecording ledger,
      UraSettings ura,
      Behaviour behaviour,
      TwinDiscard twinDiscard,
      CollectionMetrics metrics,
      StageTimer<CollectStage> timer,
      RowSink rows) {
    this.variant = variant;
    this.runSeed = runSeed;
    this.iterationSeed = iterationSeed;
    this.phi = phi;
    this.sampling = sampling == null ? Sampling.OFF : sampling;
    this.ledger = ledger;
    this.ura = new UraValues(ura, phi, iterationSeed);
    this.behaviour = behaviour;
    this.twinDiscard = twinDiscard;
    this.metrics = metrics;
    this.timer = timer;
    this.rows = rows;
    outcomes = OutcomeCodec.of(variant);
  }

  @Override
  public RunHooks<PolicyRequest> main(Run<PolicyRequest> run) {
    return new CollectionPath(this, run.game().wall());
  }

  /** 書き終えていない行が上限を超えたら新しい半荘を始めない。 */
  @Override
  public boolean backlogged() {
    return rows.backlog() >= ROW_BACKLOG;
  }

  /** 局末の値を待っている受け取り先ごとに、帳簿、局末の事実、値の順に渡す。ドライバのスレッドだけが呼ぶ。 */
  void accept(HandEnd end) {
    double[] values = end.values();
    List<Slot> targets = end.targets();
    HandEndFacts facts = end.facts();
    long outcomeNanos = 0;
    for (Slot slot : targets) {
      if (slot.ledger() != null) slot.target().acceptLedger(slot.action(), slot.ledger());
      if (facts != null) {
        long started = System.nanoTime();
        slot.target().acceptOutcome(slot.action(), facts, outcomes);
        outcomeNanos += System.nanoTime() - started;
      }
      slot.target().accept(slot.action(), values);
    }
    if (facts != null) timer.add(CollectStage.V2_OUTCOMES, outcomeNanos, targets.size());
  }

  /**
   * 値がそろった行を作り（{@link RowDraft#row}）、受け取り先へ渡す。ドライバのスレッドだけが呼ぶ。
   *
   * @param draft 行の下書き
   * @param values 合法手ごとの Y
   * @param ledger 手ごとの運の精算の帳簿（帳簿を付けない収集では null）
   * @param outcomes 手ごとの局末の 3 ワード
   */
  void write(RowDraft draft, float[] values, float[] ledger, long[] outcomes) {
    metrics.rowsWritten.increment();
    long started = System.nanoTime();
    rows.accept(draft.row(values, ledger, outcomes));
    timer.add(CollectStage.V2_ROWS, System.nanoTime() - started);
  }

  /** 半荘の行の用途を返す。ホールドアウトに回すかは run 全体の種と半荘 ID だけで決めるので、反復をまたいでも変わらない。 */
  Row.Purpose purpose(long gameId) {
    return CrnSeeds.unit(CrnSeeds.derive(runSeed, Domain.HOLDOUT, gameId)) < sampling.holdoutFraction()
        ? Row.Purpose.HOLDOUT
        : Row.Purpose.TRAIN;
  }

  /** 主経路の乱数のキー。 */
  long mainKey(long gameId) {
    return CrnSeeds.mainKey(iterationSeed, gameId);
  }

  /** 主経路の判断（{@code rootDecisionId}）から作った写しの、乱数のキー。 */
  long copyKey(long gameId, long rootDecisionId) {
    return CrnSeeds.copyKey(iterationSeed, gameId, rootDecisionId);
  }

  /**
   * 主経路の判断をラベル行にするかを、判断のクラスと手の形で決まる率 ρ で引く。
   *
   * @param coinKey 乱数のキー
   * @param decisionId 判断 ID
   * @param c 判断のクラス
   * @param hand 判断する席の手の形
   * @return ラベル行にするなら true
   */
  boolean label(long coinKey, long decisionId, DecisionClass c, Openness hand) {
    double rate = sampling.labelRate(c, hand);
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.LABEL, decisionId) < rate;
  }

  /** 主経路の判断で中立ロールインを起こすかを、根の判断のクラスの率 β で引く。 */
  boolean rollIn(long coinKey, long decisionId, DecisionClass c) {
    double rate = sampling.rollIn(c).rate();
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.ROLLIN, decisionId) < rate;
  }

  /**
   * 写しの中で、逸脱した席の判断をラベル行にするかを引く。行の数の上限は呼び出し元が数える。
   *
   * @param coinKey 乱数のキー
   * @param decisionId 判断 ID
   * @param rate 写しの根のクラスと判断のクラスで決まる率 ρ_br
   * @return ラベル行にするなら true
   */
  boolean copyLabel(long coinKey, long decisionId, double rate) {
    return rate > 0 && CrnSeeds.coin(coinKey, Domain.COPY_LABEL, decisionId) < rate;
  }

  /** 候補集合に一様に選んだ 1 手を足すかを決める乱数。ε_C と比べる。 */
  double exploreCoin(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.EXPLORE, decisionId, 1);
  }

  /** 候補集合に足す 1 手を一様に選ぶための乱数。 */
  double exploreDraw(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.EXPLORE, decisionId, 2);
  }

  /** 逸脱 d ~ q(·|o) を引くための一様乱数。 */
  double deviationCoin(long coinKey, long decisionId) {
    return CrnSeeds.coin(coinKey, Domain.DEVIATION, decisionId);
  }
}
