package com.epsilon.pico.ai.decision.training;

import java.util.Objects;

/**
 * Decision 学習器の拒否理由を、制御用の型とログ用文字列に分けて保持する。
 *
 * @param code 機械判定に使う拒否コード
 * @param context パラメーター更新など呼び出し側が付与した文脈
 * @param diagnostic 観測値を含む人間向け診断
 */
public record EpsilonDecisionTrainingRejection(Code code, String context, String diagnostic) {

  public static final EpsilonDecisionTrainingRejection PASSED =
      new EpsilonDecisionTrainingRejection(Code.PASSED, "", "");

  /** 通過・拒否時の診断文字列契約を検証する。 */
  public EpsilonDecisionTrainingRejection {
    code = Objects.requireNonNull(code, "code");
    context = Objects.requireNonNull(context, "context");
    diagnostic = Objects.requireNonNull(diagnostic, "diagnostic");
    if (code == Code.PASSED) {
      if (!context.isEmpty() || !diagnostic.isEmpty()) {
        throw new IllegalArgumentException("passed rejection must not contain failure details");
      }
    } else if (diagnostic.isEmpty()) {
      throw new IllegalArgumentException("rejected result requires a diagnostic");
    }
  }

  /**
   * 文脈接頭部分を持たない拒否結果を作る。
   *
   * @param code 機械判定用の拒否コード
   * @param diagnostic 観測値を含む診断文字列
   * @return 検証済み拒否結果
   */
  public static EpsilonDecisionTrainingRejection rejected(Code code, String diagnostic) {
    return new EpsilonDecisionTrainingRejection(code, "", diagnostic);
  }

  /**
   * オプティマイザー確定を拒否すべき結果か判定する。
   *
   * @return コードが PASSED 以外なら {@code true}
   */
  public boolean rejected() {
    return code != Code.PASSED;
  }

  /**
   * 既存の reason・ログ形式を変えずに返す。
   *
   * @return コンテキストと診断結果を連結した文字列
   */
  public String reason() {
    return context + diagnostic;
  }

  /**
   * パラメーター更新などの呼出し文脈を診断結果の解析なしで前置する。
   *
   * @param prefix 既存コンテキストの前に加える文字列
   * @return 拒否時は接頭部分付きの新しい結果、通過時または空接頭部分ならこの結果
   */
  public EpsilonDecisionTrainingRejection withPrefix(String prefix) {
    String actual = Objects.requireNonNull(prefix, "prefix");
    if (!rejected() || actual.isEmpty()) {
      return this;
    }
    return new EpsilonDecisionTrainingRejection(code, actual + context, diagnostic);
  }

  /** 診断文字列を解析せずに制御分岐できる安定した拒否コード。 */
  public enum Code {
    /** すべての検証条件を通過した。 */
    PASSED,
    /** 平均 {@code KL(piRollout || piCurrent)} が上限を超えた。 */
    ROLLOUT_POLICY_KL_MEAN_EXCEEDED
  }
}
