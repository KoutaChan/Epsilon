package com.epsilon.nano.ai.decision.data;

/** 牌譜の解析失敗として無視せず、呼び出し元へ通知する必要がある学習データの不正を表す例外。 */
public final class EpsilonDecisionDataException extends IllegalArgumentException {

  /**
   * 不正データの説明を持つ例外を作る。
   *
   * @param message 破損したフィールドや値を示す説明
   */
  public EpsilonDecisionDataException(String message) {
    super(message);
  }

  /**
   * 原因例外を保持する不正データ例外を作る。
   *
   * @param message 破損したフィールドや値を示す説明
   * @param cause 解析・復号失敗の原因
   */
  public EpsilonDecisionDataException(String message, Throwable cause) {
    super(message, cause);
  }
}
