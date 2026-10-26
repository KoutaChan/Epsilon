package com.epsilon.prism.rows;

import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.core.Kind;
import com.epsilon.prism.core.Stage;

/**
 * 収集した判断の 1 行（行の形式 版 1、{@link RowFormat}）。行は 3 種類（{@link Source}）。
 *
 * <ul>
 *   <li>{@link Source#MAIN}: 主経路のラベル行（M1、K = 1）。候補集合 C(o) の各手を同じ山・同じ鍵の分岐で局末まで打った値 Y_a。実際に打った手
 *       （played）は主経路の値をそのまま使う。
 *   <li>{@link Source#ROLLIN}: 写しの中の、逸脱した席のラベル行（M2）。値の付け方は MAIN と同じで、played は写しの中で打った手。
 *   <li>{@link Source#ROOT}: 写しの根の記録（S）。主経路の判断 o で、played（主経路の値 y_main）と逸脱 d（写しの値 y_br）の 2 枠。d が played と
 *       同じなら played の 1 枠（y_br = y_main）。傾向 p = β_c(o)·q(d|o) を持つ。
 * </ul>
 *
 * <p>値はどれも判断の席から見た局末の Φ（固定の GRP。半荘末は厳密な効用）。測っていない手は NaN。
 *
 * @param iteration 行を集めたコホート（行動方策 b = 版 iteration）
 * @param decisionId エンジンの判断 ID（主経路か写しの中の）
 * @param seat 判断の席
 * @param decisionClass 判断の類（λ_c と認証の集計の単位）
 * @param purpose 半荘の用途（TRAIN・HOLDOUT）
 * @param group 半荘の組（A・B。生徒 A は組 A の行で、生徒 B は組 B の行で学ぶ）
 * @param source 行の種類
 * @param played その経路で実際に打った手の位置（MAIN・ROOT は主経路の手、ROLLIN は写しの中の手）
 * @param actions 合法手の数
 * @param values 合法手ごとの値 Y（長さ actions。測った手のほかは NaN）
 * @param prior 合法手ごとの規則事前 p0（長さ actions、和 1。{@link com.epsilon.prism.policy.Prior}）
 * @param support 候補集合 C(o)（合法手の位置の昇順。双子をまとめた後、played の双子は played に置き換えたもの）。ラベル行で測った手は support ∪
 *     {played}、根の記録では q の台
 * @param deviation ROOT の逸脱 d の位置（ほかは -1）
 * @param deviationProbability ROOT の q(d|o)（ほかは NaN）
 * @param propensity ROOT の傾向 p = β_c(o)·q(d|o)（ほかは NaN）
 * @param rootDecisionId ROLLIN の行の写しを作った主経路の判断 ID（ほかは -1）
 * @param payload 判断の席の観測（合法手の並びを含む）
 */
public record PrismRow(
    int iteration,
    long gameId,
    long decisionId,
    int seat,
    Kind kind,
    Stage stage,
    DecisionClass decisionClass,
    Purpose purpose,
    Group group,
    Source source,
    int played,
    int actions,
    float[] values,
    float[] prior,
    byte[] support,
    int deviation,
    float deviationProbability,
    float propensity,
    long rootDecisionId,
    ObservationPayload payload) {
  /** 実際に打った手の値。 */
  public float playedValue() {
    return values[played];
  }

  /** 値を測った手（values が NaN でない手）の数。 */
  public int measuredCount() {
    return measuredCount(values);
  }

  /** values の NaN でない数。 */
  public static int measuredCount(float[] values) {
    int count = 0;
    for (float value : values) if (!Float.isNaN(value)) count++;
    return count;
  }

  /** ROOT の逸脱の値 y_br（d = played なら y_main）。 */
  public float deviationValue() {
    if (source != Source.ROOT) throw new IllegalStateException("Not a root record: " + source);
    return values[deviation];
  }

  /** 行の用途。番号は行の形式（フレームの索引と本体）に入るので、足すときは末尾に足す。 */
  public enum Purpose {
    /** 学習の半荘の行。 */
    TRAIN,
    /** 保留の半荘の行（保留の指標だけ。学習には使わない）。 */
    HOLDOUT
  }

  /** 半荘の組（交差した認証、論文 3.5）。番号は行の形式に入る。 */
  public enum Group {
    A,
    B;

    /** もう一方の組。 */
    public Group other() {
      return this == A ? B : A;
    }
  }

  /** 行の種類。番号は行の形式に入る。 */
  public enum Source {
    /** 主経路のラベル行。 */
    MAIN,
    /** 写しの中の逸脱した席のラベル行。 */
    ROLLIN,
    /** 写しの根の記録。 */
    ROOT;

    /** 学習（生徒の差分整合と蒸留）に使うラベル行か。 */
    public boolean label() {
      return this != ROOT;
    }
  }
}
