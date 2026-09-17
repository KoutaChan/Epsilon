package com.epsilon.pico.ai.decision.input;

/**
 * 合法手とその後の遷移の可変個数を、あらかじめ定めた固定長の入力容量へ対応させる。
 *
 * @param legalActionCapacity 一行あたりの合法行動候補の位置数。実候補以降はパディング
 * @param actionTransitionCapacity 各行動が保持できる遷移候補の位置数。全行動が最低1件を持つ
 */
public record DecisionBucket(int legalActionCapacity, int actionTransitionCapacity) {
  /** 宣言値がスキーマで許可された容量区分かを検証する。 */
  public DecisionBucket {
    if (DecisionInputSchema.legalActionBucket(legalActionCapacity) != legalActionCapacity) {
      throw new IllegalArgumentException("not a legal-action bucket: " + legalActionCapacity);
    }
    if (DecisionInputSchema.actionTransitionBucket(actionTransitionCapacity)
        != actionTransitionCapacity) {
      throw new IllegalArgumentException(
          "not an action-transition bucket: " + actionTransitionCapacity);
    }
  }

  /**
   * 必要候補数を収容する最小容量区分を返す。
   *
   * @param legalActionCount 行に存在する合法行動数
   * @param maximumActionTransitionCount いずれか一行動が持つ最大遷移数
   * @return 両方をパディング込みで収容する最小スキーマ容量区分
   */
  public static DecisionBucket forRequiredCounts(
      int legalActionCount, int maximumActionTransitionCount) {
    return new DecisionBucket(
        DecisionInputSchema.legalActionBucket(legalActionCount),
        DecisionInputSchema.actionTransitionBucket(maximumActionTransitionCount));
  }

  /**
   * 推論バッチ間で形状を共有しやすい粗粒度容量区分を返す。
   *
   * <p>学習用の最小容量区分は変更しない。遷移が識別情報だけの通常判断は、単一候補と大候補集合を除いて16 行動へまとめる。
   * 鳴き後打牌を持つ判断は行動パディングと遷移パディングの積が大きいため、4・8・16 行動を維持する。
   *
   * @param legalActionCount 行に存在する合法行動数
   * @param maximumActionTransitionCount いずれか一行動が持つ最大遷移数
   * @return 推論バッチを合流しやすい粗粒度容量区分
   */
  public static DecisionBucket forInferenceCounts(
      int legalActionCount, int maximumActionTransitionCount) {
    DecisionInputSchema.legalActionBucket(legalActionCount);
    DecisionInputSchema.actionTransitionBucket(maximumActionTransitionCount);
    int actionTransitionCapacity = maximumActionTransitionCount == 1 ? 1 : 16;
    int legalActionCapacity;
    if (legalActionCount == 1) {
      legalActionCapacity = 1;
    } else if (legalActionCount > 16) {
      legalActionCapacity = 40;
    } else if (actionTransitionCapacity == 1) {
      legalActionCapacity = 16;
    } else if (legalActionCount <= 4) {
      legalActionCapacity = 4;
    } else if (legalActionCount <= 8) {
      legalActionCapacity = 8;
    } else {
      legalActionCapacity = 16;
    }
    return new DecisionBucket(legalActionCapacity, actionTransitionCapacity);
  }
}
