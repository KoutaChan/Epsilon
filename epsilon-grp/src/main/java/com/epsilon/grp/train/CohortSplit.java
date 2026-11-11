package com.epsilon.grp.train;

import com.epsilon.mahjong.rules.Variant;
import java.util.BitSet;
import java.util.List;
import java.util.Objects;
import java.util.function.IntToLongFunction;

/**
 * 学習に使う範囲の各コホートで、半荘を学習用の行、早期終了用の検証の行、行にしないものに振り分ける。GrpRows と OutcomeRows の
 * Builder が使う。範囲の中のコホートを 1 度ずつ受け付け、半荘の分け方は {@link GateCohort} に従う。早期終了用の検証の行にするのは、
 * 範囲の新しい方から {@link GrpRows#STOP_COHORTS} 個のコホートにある、早期終了用の半荘だけ。
 */
final class CohortSplit {

  enum Target {
    TRAIN,
    STOP,
    /** 行にしない。関門用の半荘と、古いコホートの早期終了用の半荘。 */
    NONE
  }

  private final Variant variant;
  private final GateCohort gate;
  private final CohortWeights weights;
  private final BitSet added = new BitSet();

  /** 範囲に入るコホートが 1 から 256 個でなければ IllegalArgumentException を投げる。 */
  CohortSplit(Variant variant, RecencyWindow window, GateCohort gate) {
    this.variant = Objects.requireNonNull(variant, "variant");
    this.gate = Objects.requireNonNull(gate, "gate");
    weights = new CohortWeights(Objects.requireNonNull(window, "window"));
  }

  CohortWeights weights() {
    return weights;
  }

  /**
   * コホート 1 つの半荘を振り分ける。
   *
   * @param cohort コホートの番号
   * @param games そのコホートの半荘
   * @return 半荘ごとの行き先（半荘の順）
   * @throws IllegalArgumentException コホートが範囲の外のとき、同じコホートを 2 度渡したとき、ルールの違う半荘があるとき
   */
  Target[] accept(int cohort, List<GameRecord> games) {
    for (GameRecord game : games) {
      if (game.variant() != variant) {
        throw new IllegalArgumentException("A " + game.variant() + " game in " + variant + " rows");
      }
    }
    return accept(cohort, games.size(), index -> games.get(index).gameId());
  }

  /**
   * コホート 1 つの半荘を、半荘 ID だけを見て振り分ける。半荘のルールは呼び出し元が確かめる。
   *
   * @param cohort コホートの番号
   * @param games 半荘の数
   * @param gameIds コホートの中で何番目の半荘かを受け取り、その半荘 ID を返す関数
   * @return 半荘ごとの行き先（半荘の順）
   * @throws IllegalArgumentException コホートが範囲の外のとき、同じコホートを 2 度渡したとき
   */
  Target[] accept(int cohort, int games, IntToLongFunction gameIds) {
    RecencyWindow window = weights.window();
    if (!window.contains(cohort)) {
      throw new IllegalArgumentException("Cohort " + cohort + " is outside " + window);
    }
    int offset = cohort - window.first();
    if (added.get(offset)) {
      throw new IllegalArgumentException("Cohort " + cohort + " was already added");
    }
    added.set(offset);
    boolean gateCohort = cohort == gate.cohort(window);
    boolean stopping = cohort > window.newest() - GrpRows.STOP_COHORTS;
    Target[] targets = new Target[games];
    for (int index = 0; index < targets.length; index++) {
      targets[index] = switch (gate.slice(gameIds.applyAsLong(index), gateCohort)) {
        case TRAIN -> Target.TRAIN;
        case STOP -> stopping ? Target.STOP : Target.NONE;
        case GUARD -> Target.NONE;
      };
    }
    return targets;
  }
}
