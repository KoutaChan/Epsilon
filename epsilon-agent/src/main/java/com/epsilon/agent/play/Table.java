package com.epsilon.agent.play;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import java.util.concurrent.atomic.LongAdder;

/** {@link TableLoop#play} を 1 回呼ぶ間、全ての経路が共有する状態。変わらない計画と、どのスレッドからも足せる件数を持つ。 */
final class Table<Q extends Query> {

  final TablePlan<Q> plan;
  final TableHooks<Q> hooks;

  /** 全席の行動方策。null なら各席の打ち手の分布から手を引く。 */
  final Behaviour behaviour;

  /** 判断に共通乱数をどう使うか。行動方策がなければ {@link CrnScheme#SINGLE_RACE}。 */
  final CrnScheme crn;

  final StageTimer<TableStage> timer = new StageTimer<>(TableStage.class);

  /** 主経路で打ち終えた半荘の数と局の数。 */
  final LongAdder games = new LongAdder(), rounds = new LongAdder();

  /** 打ち手に問い合わせた判断の数と、合法手が 1 つしかなく打ち手に聞かなかった判断の数。主経路・写し・分岐の全てを数える。 */
  final LongAdder queried = new LongAdder(), forced = new LongAdder();

  private final ThreadLocal<Tools> tools;

  Table(TablePlan<Q> plan, TableHooks<Q> hooks, ThreadLocal<Tools> tools) {
    this.plan = plan;
    this.hooks = hooks;
    this.tools = tools;
    behaviour = plan.behaviour();
    crn = behaviour == null ? CrnScheme.SINGLE_RACE : behaviour.scheme();
  }

  /** 呼んだスレッドのヘルパー。 */
  Tools tools() {
    return tools.get();
  }

  /** 経路を 1 回進める間に数えた判断の数を足す。 */
  void count(long queriedDecisions, long forcedDecisions) {
    if (queriedDecisions != 0) queried.add(queriedDecisions);
    if (forcedDecisions != 0) forced.add(forcedDecisions);
  }

  /** 進行のスレッドごとのヘルパー。どれもスレッドセーフでない。経路は判断の時点ごとに別のスレッドへ移るので、経路ではなくスレッドに持たせる。 */
  static final class Tools {
    final RuleWorkspace workspace;
    final ObservationWriter writer;

    /** 直前に取り込んだ記録、または {@link Run#read} で問い合わせた記録を読むリーダー。 */
    final Observation reader;

    Tools(Variant variant) {
      workspace = new RuleWorkspace(variant);
      writer = new ObservationWriter(variant);
      reader = new Observation(variant);
    }
  }
}
