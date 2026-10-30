package com.epsilon.prism.rows;

import com.epsilon.prism.Limits;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.auxiliary.OutcomeCodec;
import com.epsilon.prism.auxiliary.TargetSchema;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.observe.ObservationSchema;
import java.util.Objects;

/**
 * PRISM の 1 行（収集した判断 1 つ）。直列化は {@link RowCodec}、コホートへの書き出しは {@link CohortWriter}。行は 3 種類（{@link Source}）。
 *
 * <ul>
 *   <li>{@link Source#MAIN}: 主経路のラベル行。候補集合 C(o) の各手を同じ山・同じ鍵の分岐で局末まで打った値 Y_a。実際に打った手
 *       （played）は主経路の値をそのまま使う。
 *   <li>{@link Source#ROLLIN}: 写しの中の、逸脱した席のラベル行。値の付け方は MAIN と同じで、played は写しの中で打った手。
 *   <li>{@link Source#ROOT}: 写しの根の記録。主経路の判断 o で、played（主経路の値 y_main）と逸脱 d（写しの値 y_br）の 2 枠。d が played と
 *       同じなら played の 1 枠（y_br = y_main）。傾向 p = β_c(o)·q(d|o) を持つ。
 * </ul>
 *
 * <p>値はどれも判断の席から見た局末の Φ（固定の GRP。半荘末は厳密な効用）。測っていない手は NaN。入力は記録 O で持ち、補助の頭の教師（測った手
 * ごとの局末の 3 語と神託）と運の精算の帳簿を持つ。帳簿の欄の数はコホートで 1 つ（目録）で、運の精算は {@link LedgerRow} の口で読む。
 *
 * <p>判断の席・合法手・判断の種類・局の段階は O が持つので、行には重ねない（{@link Observation#observer} など）。O の合法手の数と配列の長さ
 * が合うことは {@link RowCodec} が書くときと読むときに確かめる。配列は受け取ったものをそのまま持ち、写さない。
 *
 * @param gameId 半荘の ID
 * @param decisionId エンジンの判断 ID（主経路か写しの中の）
 * @param decisionClass 判断の類（行を選ぶ鍵）
 * @param purpose 半荘の用途
 * @param group 半荘の組
 * @param source 行の種類
 * @param played その経路で実際に打った手の位置
 * @param values 合法手ごとの値 Y（測っていない手は NaN。長さは合法手の数）
 * @param prior 合法手ごとの規則事前 p0
 * @param support 候補集合 C(o)（合法手の位置のビット。ラベル行で測った手は support ∪ {played}、根の記録では q の台）
 * @param deviation ROOT の逸脱 d の位置（ほかは −1）
 * @param deviationProbability ROOT の q(d|o)（ほかは NaN）
 * @param propensity ROOT の傾向 β_c(o)·q(d|o)、配分したラベル行の相対の包含確率 r(o)（ほかは NaN）
 * @param rootDecisionId ROLLIN の行の写しを作った主経路の判断 ID（ほかは −1）
 * @param ledger 手ごとの運の精算の帳簿（長さは合法手 × 欄の数 F で、手 a の欄 f は {@code ledger[a·F + f]}。測っていない手は NaN、測った手は
 *     有限の値）
 * @param outcomes 手ごとの局末の記録（長さは合法手 × {@link OutcomeCodec#WORDS} で、手 a の枠は {@code a·WORDS} から。記録の
 *     ない手は 0。記録するのは測った手のうち {@link TargetSchema#hasCandidateTargets} が真の手で、収集が決める）
 * @param oracle 神託（{@link OracleCodec}。神託を付けない行は長さ 0）
 * @param observation 記録 O（{@link ObservationSchema#longs} の数の {@code long}）
 */
public record Row(
    long gameId,
    long decisionId,
    DecisionClass decisionClass,
    Purpose purpose,
    Group group,
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
    long[] observation)
    implements LedgerRow {

  /** 行の用途。番号は行の形式（フレームの索引）に入るので、足すときは末尾に足す。 */
  public enum Purpose {
    /** 学習の半荘の行。 */
    TRAIN,
    /** 保留の半荘の行（学習には使わない）。 */
    HOLDOUT
  }

  /** 半荘の組（run の種と半荘 ID の硬貨で半々に分ける）。番号は行の形式に入る。 */
  public enum Group {
    A,
    B
  }

  /** 行の種類。番号は行の形式に入る。 */
  public enum Source {
    /** 主経路のラベル行。 */
    MAIN,
    /** 写しの中の逸脱した席のラベル行。 */
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
    Objects.requireNonNull(group, "group");
    Objects.requireNonNull(source, "source");
    Objects.requireNonNull(oracle, "oracle");
    Objects.requireNonNull(observation, "observation");
    int actions = values.length;
    if (actions < 1 || actions > Limits.MAX_LEGAL_ACTIONS) {
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

  /** 手 {@code action} を測った（値がある）。 */
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

  /** 帳簿を持つ行か（帳簿を付けない収集の行は欄 0）。 */
  @Override
  public boolean hasLedger() {
    return ledger.length > 0;
  }

  @Override
  public int ledgerFeatures() {
    return ledger.length / values.length;
  }

  /** 手 {@code action} の帳簿の欄 {@code feature}（帳簿のない行は NaN）。 */
  @Override
  public float ledger(int action, int feature) {
    return ledger.length == 0 ? Float.NaN : ledger[action * ledgerFeatures() + feature];
  }
}
