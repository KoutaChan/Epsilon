package com.epsilon.ai.decision;

import com.epsilon.core.Action;
import java.util.List;

/** モデルの既存終了ゲートとの対応。ネットワークの出力は増やさない。 */
public enum DecisionBranchGate {
  RON(0, Action.Group.RON),
  TSUMO(1, Action.Group.TSUMO),
  KYUSHU(2, Action.Group.KYUSHU);

  /** 同時合法なら九種九牌を優先し、一判断に二つの比較教師を混ぜない。 */
  public static final List<DecisionBranchGate> PRIORITY = List.of(KYUSHU, RON, TSUMO);

  private final int scoreIndex;
  private final Action.Group group;

  DecisionBranchGate(int scoreIndex, Action.Group group) {
    this.scoreIndex = scoreIndex;
    this.group = group;
  }

  /** 既存ネットワークの終了ゲートの列位置を返す。 */
  public int scoreIndex() {
    return scoreIndex;
  }

  /** このゲートの宣言に相当する行動かを返す。 */
  public boolean accepts(Action action) {
    return action.type().group() == group;
  }

  /** このゲートの宣言または続行として選べる行動かを返す。 */
  public boolean contains(Action action) {
    return this != KYUSHU || action.type().group() != Action.Group.TSUMO;
  }
}
