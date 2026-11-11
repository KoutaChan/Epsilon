package com.epsilon.prism.rows;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.core.DecisionClass;
import java.util.Objects;

/**
 * PRISM の 1 行で、収集した判断 1 つに当たる。直列化は {@link RowCodec}、コホートへの書き出しは {@link CohortWriter} が行う。行には
 * 3 種類ある（{@link Source}）。
 *
 * <ul>
 *   <li>{@link Source#MAIN}: 主経路のラベル行。候補集合 C(o) の各手を、同じ牌山・同じ乱数のキーで分岐させて局末まで打った値 Y_a を持つ。
 *       実際に打った手（played）には主経路の値をそのまま使う。
 *   <li>{@link Source#ROLLIN}: 写しの中で逸脱した席のラベル行。値の付け方は MAIN と同じで、played は写しの中で打った手。
 *   <li>{@link Source#ROOT}: 写しの根の記録。主経路の判断 o について、played（主経路の値 y_main）と逸脱 d（写しの値 y_br）の 2 手を持つ。
 *       d が played と同じなら played の 1 手だけ（y_br = y_main）。傾向 p = β_c(o)·q(d|o) を持つ。
 * </ul>
 *
 * <p>値はどれも、判断する席から見た局末の Φ（固定の GRP。半荘の終わりでは厳密な効用）。測っていない手は NaN。入力は記録 O で持つ。ほかに、
 * 補助のヘッドの教師（測った手ごとの局末の記録 3 ワードと、Oracle の記録）と、運の精算の帳簿を持つ。帳簿の列の数はコホートで 1 つに
 * 決まり、マニフェストに書く。
 *
 * <p>判断する席・合法手・判断の種類・局の段階は O が持つので、行には重ねて持たない（{@link Observation#observer} など）。O の合法手の数と
 * 配列の長さが合うことは、{@link RowCodec} が書くときと読むときに確かめる。配列は受け取ったものをそのまま持ち、コピーしない。
 *
 * @param gameId 半荘の ID
 * @param decisionId エンジンの判断 ID（主経路か、写しの中のもの）
 * @param decisionClass 判断のクラス（行を選ぶキー）
 * @param purpose 半荘の用途
 * @param source 行の種類
 * @param played その経路で実際に打った手の位置
 * @param values 合法手ごとの値 Y（測っていない手は NaN。長さは合法手の数）
 * @param prior 合法手ごとのルールベースの事前分布 p0
 * @param support 候補集合 C(o)（合法手の位置ごとのビット）。ラベル行で測った手は support ∪ {played}。根の記録では q の台
 * @param deviation ROOT の行では逸脱 d の位置（ほかの行は −1）
 * @param deviationProbability ROOT の行では q(d|o)（ほかの行は NaN）
 * @param propensity ROOT の行では傾向 β_c(o)·q(d|o)（ほかの行は NaN）
 * @param rootDecisionId ROLLIN の行では、その写しを作った主経路の判断 ID（ほかの行は −1）
 * @param ledger 合法手ごとの運の精算の帳簿。長さは合法手の数 × 列の数 F で、手 a の列 f は {@code ledger[a·F + f]}。測っていない手は NaN、
 *     測った手は有限の値
 * @param outcomes 合法手ごとの局末の記録。長さは合法手の数 × {@link OutcomeCodec#WORDS} で、手 a の記録は {@code a·WORDS} から始まる。
 *     記録のない手は 0。記録するのは、測った手のうち {@link TargetSchema#hasCandidateTargets} が真の手で、どの手かは収集が決める
 * @param oracle Oracle の記録（{@link OracleCodec}。付けない行は長さ 0）
 * @param observation 記録 O（{@link ObservationSchema#longs} の数の {@code long}）
 */
public record Row(
    long gameId,
    long decisionId,
    DecisionClass decisionClass,
    Purpose purpose,
    Source source,
    int played,
    float[] values,
    float[] prior,
    int support,
    int deviation,
    float deviationProbability,
    float propensity,
    long rootDecisionId,
    float[] ledger,
    long[] outcomes,
    long[] oracle,
    long[] observation) {

  /** 行の用途。番号は行の形式（フレームのインデックス）に入るので、足すときは末尾に足す。 */
  public enum Purpose {
    /** 学習の半荘の行。 */
    TRAIN,
    /** ホールドアウトの半荘の行（学習には使わない）。 */
    HOLDOUT
  }

  /** 行の種類。番号は行の形式に入る。 */
  public enum Source {
    /** 主経路のラベル行。 */
    MAIN,
    /** 写しの中で逸脱した席のラベル行。 */
    ROLLIN,
    /** 写しの根の記録。 */
    ROOT;

    /** 差分整合の教師にするラベル行か。 */
    public boolean label() {
      return this == MAIN || this == ROLLIN;
    }
  }

  /**
   * 行を作る。
   *
   * @throws IllegalArgumentException 配列の長さか手の位置が合わないか、測っていない手に帳簿か局末の記録があるとき
   */
  public Row {
    Objects.requireNonNull(decisionClass, "decisionClass");
    Objects.requireNonNull(purpose, "purpose");
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(oracle, "oracle");
    Objects.requireNonNull(observation, "observation");
    int actions = values.length;
    if (actions < 1 || actions > Decisions.MAX_LEGAL) {
      throw new IllegalArgumentException("legal action count out of range: " + actions);
    }
    if (prior.length != actions
        || ledger.length % actions != 0
        || outcomes.length != actions * OutcomeCodec.WORDS) {
      throw new IllegalArgumentException("row arrays must have one entry per legal action");
    }
    if (played < 0 || played >= actions || deviation < -1 || deviation >= actions) {
      throw new IllegalArgumentException(
          "action position out of range: played=" + played + " deviation=" + deviation);
    }
    if (actions < Integer.SIZE && (support >>> actions) != 0) {
      throw new IllegalArgumentException("support names actions beyond " + actions);
    }
    int features = ledger.length / actions;
    for (int action = 0; action < actions; action++) {
      boolean measured = !Float.isNaN(values[action]);
      for (int feature = 0; feature < features; feature++) {
        float term = ledger[action * features + feature];
        if (measured ? !Float.isFinite(term) : !Float.isNaN(term)) {
          throw new IllegalArgumentException(
              "ledger of action " + action + " does not match its measurement");
        }
      }
      for (int word = 0; word < OutcomeCodec.WORDS && !measured; word++) {
        if (outcomes[action * OutcomeCodec.WORDS + word] != 0L) {
          throw new IllegalArgumentException("outcome of an unmeasured action " + action);
        }
      }
    }
  }

  /** 合法手の数。 */
  public int actions() {
    return values.length;
  }

  /** 手 {@code action} を測ったか（値があるか）。 */
  public boolean measured(int action) {
    return !Float.isNaN(values[action]);
  }

  /** 測った手の数。 */
  public int measuredCount() {
    int count = 0;
    for (float value : values) {
      if (!Float.isNaN(value)) {
        count++;
      }
    }
    return count;
  }

  /** 帳簿を持つ行か。帳簿を付けない収集の行は、帳簿の列が 0。 */
  public boolean hasLedger() {
    return ledger.length > 0;
  }

  /** 帳簿の列の数（帳簿のない行は 0）。 */
  public int ledgerFeatures() {
    return ledger.length / values.length;
  }

  /** 手 {@code action} の、帳簿の列 {@code feature} の値（帳簿のない行は NaN）。 */
  public float ledger(int action, int feature) {
    return ledger.length == 0 ? Float.NaN : ledger[action * ledgerFeatures() + feature];
  }
}
