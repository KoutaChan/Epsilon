package com.epsilon.engine;

import com.epsilon.calculate.scoring.HandScoreBuffer;
import com.epsilon.core.Action;
import java.util.List;
import java.util.Objects;

/**
 * エンジンが行動の選択を待つ判断時点。
 *
 * <p>状態と合法手は停止中のエンジンから読み取り専用で借用する。次に行動が反映されるまで有効で、呼び出し元は変更したり有効期間を超えて保持したりしてはならない。
 *
 * <dl>
 *   <dt>{@code id}
 *   <dd>行動を反映する際に照合する判断を一意に示すID
 *   <dt>{@code kind}
 *   <dd>ツモ番・打牌応答などの判断境界種別
 *   <dt>{@code player}
 *   <dd>選択を要求されている席番号（0-3）
 *   <dt>{@code legalActions}
 *   <dd>エンジンが管理する合法手の読み取り専用一覧。次の行動反映まで有効
 * </dl>
 */
public final class EngineDecisionPoint {

  private final EngineActionBuffer legalActions = new EngineActionBuffer();
  private final HandScoreBuffer immediateWinResult = new HandScoreBuffer();
  private boolean immediateWinAvailable;
  private long id;
  private EngineDecisionKind kind;
  private int player;

  EngineDecisionPoint() {}

  void bind(long id, EngineDecisionKind kind, int player) {
    this.id = id;
    this.kind = Objects.requireNonNull(kind, "kind");
    this.player = player;
    legalActions.clear();
    immediateWinAvailable = false;
  }

  void clear() {
    legalActions.clear();
    id = 0L;
    kind = null;
    player = -1;
    immediateWinAvailable = false;
  }

  EngineActionBuffer mutableLegalActions() {
    return legalActions;
  }

  HandScoreBuffer mutableImmediateWinResult() {
    return immediateWinResult;
  }

  void setImmediateWinAvailable(boolean available) {
    immediateWinAvailable = available;
  }

  HandScoreBuffer requireImmediateWinResult() {
    if (!immediateWinAvailable) {
      throw new IllegalStateException("decision has no cached win result: " + id);
    }
    return immediateWinResult;
  }

  void copyImmediateWinFrom(EngineDecisionPoint source) {
    immediateWinAvailable = source.immediateWinAvailable;
    if (immediateWinAvailable) immediateWinResult.copyFrom(source.immediateWinResult);
  }

  public long id() {
    return id;
  }

  public EngineDecisionKind kind() {
    return kind;
  }

  public int player() {
    return player;
  }

  public List<Action> legalActions() {
    return legalActions.borrowedView();
  }

  @Override
  public boolean equals(Object candidate) {
    if (this == candidate) {
      return true;
    }
    if (!(candidate instanceof EngineDecisionPoint other)) {
      return false;
    }
    return id == other.id
        && player == other.player
        && kind == other.kind
        && legalActions().equals(other.legalActions());
  }

  @Override
  public int hashCode() {
    return Objects.hash(id, kind, player, legalActions());
  }

  @Override
  public String toString() {
    return "EngineDecisionPoint[id="
        + id
        + ", kind="
        + kind
        + ", player="
        + player
        + ", legalActions="
        + legalActions()
        + ']';
  }
}
