package com.epsilon.prism.collect;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Openness;

/**
 * 経路（{@link CollectionPath}）1 本が推論を要求するときに数える判断の件数。経路の種類ごとに数える。書くのは、その経路を今進めている
 * スレッド 1 本だけ。経路は判断の時点ごとに別の進行スレッドへ移るが、同時に書くのはいつも 1 本。経路を進めた後に、段の時間
 * （{@link CollectStage}）と一緒に {@link #flush} で {@link CollectionMetrics} へまとめて足す。判断ごとに共有の LongAdder を触らないため。
 * 全ての判断の数と強制の判断の数は、卓のコアが数える。
 *
 * <p>経路の種類は作ってから変わらないので、作るときに決める。手の形とクラスごとの件数は、metrics でも主経路の分しか持たない
 * （mainDecisions）ので、その配列は主経路だけが持つ。写しは合計と、逸脱した席の判断（根の判断のクラスごと）を数える。分岐は合計だけを
 * 数え、作るたびに配列を作らない。
 */
final class DecisionCounts {
  private final Role role;

  /** 写しの根の判断のクラス。写しでなければ null。 */
  private final DecisionClass root;

  /** 主経路の NN 推論の判断の数。手の形とクラスごとに、{@link CollectionMetrics#byHand} の番号で引く。写しと分岐では null。 */
  private final long[] mainByHand;

  /** NN 推論の判断（合法手 2 つ以上）の合計。 */
  private long model;

  /** 写しで逸脱した席の NN 推論の判断のうち、同値な打牌をまとめても候補が 2 つ以上残るものの数。 */
  private long deviator;

  /**
   * @param root 写しの根の判断のクラス（写しでなければ null）
   */
  DecisionCounts(Role role, DecisionClass root) {
    if ((role == Role.COPY) != (root != null))
      throw new IllegalArgumentException("Only copies have a root class: " + role + " " + root);
    this.role = role;
    this.root = root;
    mainByHand = role == Role.MAIN ? new long[Openness.COUNT * DecisionClass.COUNT] : null;
  }

  /** 主経路の判断を数える。 */
  void countMain(DecisionClass decisionClass, Openness hand) {
    model++;
    mainByHand[CollectionMetrics.byHand(hand, decisionClass)]++;
  }

  /** 写しか分岐の判断を数える。 */
  void countModel() {
    model++;
  }

  /** 写しで逸脱した席の判断のうち、同値な打牌をまとめても候補が 2 つ以上残るものを数える。{@link #countModel} と合わせて呼ぶ。 */
  void countDeviator() {
    deviator++;
  }

  /** 数えた件数を metrics に足して 0 に戻す。 */
  void flush(CollectionMetrics metrics) {
    if (model == 0) return;
    switch (role) {
      case MAIN -> flushMain(metrics);
      case COPY -> {
        metrics.copyDecisions.add(model);
        if (deviator > 0) metrics.deviatorDecisions[root.ordinal()].add(deviator);
      }
      case FORK -> metrics.forkDecisions.add(model);
    }
    model = 0;
    deviator = 0;
  }

  private void flushMain(CollectionMetrics metrics) {
    for (int i = 0; i < mainByHand.length; i++) {
      long count = mainByHand[i];
      if (count == 0) continue;
      metrics.mainDecisions[i].add(count);
      mainByHand[i] = 0;
    }
  }
}
