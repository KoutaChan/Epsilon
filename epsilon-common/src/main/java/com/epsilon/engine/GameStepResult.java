package com.epsilon.engine;

import java.util.List;

/**
 * {@link GameEngine}が行動の選択待ち、局終了、半荘終了のいずれかに到達した結果。
 *
 * <p>{@link AwaitingDecisions}はエンジンが再利用するため、次の行動反映後も保持してはならない。局終了と半荘終了の結果は、その時点で新しく作る不変の値として保持できる。
 */
public sealed interface GameStepResult
    permits GameStepResult.AwaitingDecisions,
        GameStepResult.RoundEnded,
        GameStepResult.RoundSettled,
        GameStepResult.HanchanEnded {

  /**
   * 外部からの行動選択を待っている。
   *
   * <dl>
   *   <dt>{@code decisions}
   *   <dd>同時に解決する借用した判断時点一覧
   * </dl>
   */
  final class AwaitingDecisions implements GameStepResult {

    private List<EngineDecisionPoint> decisions = List.of();

    AwaitingDecisions() {}

    AwaitingDecisions bind(List<EngineDecisionPoint> decisions) {
      this.decisions = decisions;
      return this;
    }

    public List<EngineDecisionPoint> decisions() {
      return decisions;
    }
  }

  /**
   * 単独局が終了した。点棒精算はまだ行われていない。
   *
   * <dl>
   *   <dt>{@code result}
   *   <dd>和了または流局の未精算結果
   * </dl>
   */
  final class RoundEnded implements GameStepResult {

    private final RoundResult result;

    RoundEnded(RoundResult result) {
      this.result = result;
    }

    public RoundResult result() {
      return result;
    }
  }

  /**
   * 半荘内の局が終了し、点棒精算まで完了した。
   *
   * <dl>
   *   <dt>{@code settlement}
   *   <dd>精算後スコアと次局遷移
   * </dl>
   */
  final class RoundSettled implements GameStepResult {

    private final RoundSettlement settlement;

    RoundSettled(RoundSettlement settlement) {
      this.settlement = settlement;
    }

    public RoundSettlement settlement() {
      return settlement;
    }
  }

  /**
   * 半荘が終了した。
   *
   * <dl>
   *   <dt>{@code settlement}
   *   <dd>最終精算結果
   * </dl>
   */
  final class HanchanEnded implements GameStepResult {

    private final RoundSettlement settlement;

    HanchanEnded(RoundSettlement settlement) {
      this.settlement = settlement;
    }

    public RoundSettlement settlement() {
      return settlement;
    }

    /**
     * 半荘終了時の確定スコアを返す。
     *
     * @return 席番号順の点数配列
     */
    public int[] snapshotFinalScores() {
      return settlement.snapshotFinalScores();
    }
  }
}
