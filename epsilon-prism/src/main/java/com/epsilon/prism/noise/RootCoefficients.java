package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import java.util.Objects;
import java.util.Set;

/** 根の情報から帳簿の係数を学ぶ期間と、教師に使う補正の選び方。 */
public record RootCoefficients(
    @Key(doc = "根の補正を記録するか、前向きの保留で選んだ腕を教師に使うか", value = "OFF") Mode mode,
    @Key(doc = "係数の入力にする根の情報", value = "ROOT_HANDS") Information information,
    @Key(doc = "判断の中で係数を共有する範囲", value = "PER_CANDIDATE") Family family,
    @Key(doc = "係数の当てはめに使う連続した世代数", value = "2") @Range(min = 1) int trainingCohorts,
    @Key(doc = "倍率の下限を求める曲線の世代数", value = "3") @Range(min = 1) int curveCohorts,
    @Key(doc = "倍率の向きと区分の選択に使う世代数", value = "4") @Range(min = 1) int selectionCohorts,
    @Key(doc = "点差に対する倍率の向きの候補", value = "[CONSTANT, INCREASING, DECREASING]")
        Set<Direction> directions,
    @Key(doc = "倍率を当てはめる局面の区分の候補", value = "[GLOBAL, RANK, ROUND, RANK_ROUND]")
        Set<Context> contexts) {

  public enum Mode {
    OFF,
    SHADOW,
    APPLY
  }

  public enum Information {
    PUBLIC,
    ROOT_HANDS
  }

  public enum Family {
    SHARED,
    PER_CANDIDATE
  }

  public enum Direction {
    CONSTANT,
    INCREASING,
    DECREASING
  }

  public enum Context {
    GLOBAL,
    RANK,
    ROUND,
    RANK_ROUND
  }

  public RootCoefficients {
    Objects.requireNonNull(mode);
    Objects.requireNonNull(information);
    Objects.requireNonNull(family);
    directions = Set.copyOf(directions);
    contexts = Set.copyOf(contexts);
    if (trainingCohorts < 1
        || curveCohorts < 1
        || selectionCohorts < 1
        || directions.isEmpty()
        || contexts.isEmpty()) {
      throw new IllegalArgumentException(
          "Root coefficient selection needs cohorts, directions and contexts");
    }
  }

  /** 記録と適用で共有する、係数の学習条件。 */
  public RootCoefficients fitSettings() {
    return new RootCoefficients(
        Mode.SHADOW,
        information,
        family,
        trainingCohorts,
        curveCohorts,
        selectionCohorts,
        directions,
        contexts);
  }
}
