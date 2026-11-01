package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import com.epsilon.prism.auxiliary.TargetSchema.CandidateTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Loss;
import com.epsilon.prism.auxiliary.TargetSchema.Outcome;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingCondition;
import com.epsilon.prism.auxiliary.TargetSchema.ReadingTarget;
import com.epsilon.prism.auxiliary.TargetSchema.Reduction;
import java.util.Collections;
import java.util.EnumMap;
import java.util.Map;

/**
 * 補助の頭の損失。
 *
 * <pre>
 * L_aux = K′·[ α_o·Σ_{h∈読み} L̄_h + α_r·(1/8)·Σ_{h∈候補} L̄_h ]
 * L̄_h  = （頭 h の条件を満たす所の損失の和）/ n_h（微小バッチの中に条件を満たす所がなければ 0）
 * </pre>
 *
 * <p>頭ごとの損失の形・列のまとめ方・条件・正規化は {@link CandidateTarget}・{@link ReadingTarget} に従う。条件を満たさない所は損失にも
 * 分母にも入れない。列の平均（{@link Reduction#MEAN}）は条件を満たす列の平均、列の和（{@link Reduction#SUM}）は条件を満たす単位ごとの
 * 列の和の平均。softmax は単位ごとの交差エントロピーの平均。2 値の交差エントロピーと二乗誤差は FP32 のロジットで取り、点の大きな残差も
 * 切り詰めない。
 *
 * <p>分母 n_h（条件を満たす所の数）は {@link Denominators} で渡す。普通は微小バッチの教師から数えた数で、詰め方の 1 塊を分けた部分には
 * 元の塊の数を渡す。すると部分の項の和 Σ_部分（部分の和 / n_h(元の塊)）が元の塊の項になる（FP32 の和の順の違いだけ）。教師と条件の印は
 * {@link LossInputs} が順伝播の前に送り（{@link #stage}）、損失は送った配列を読む。
 *
 * @param oracleWeight 読みの頭の重み α_o
 * @param resultWeight 候補の結果の頭の重み α_r（1/頭の数 を掛けて使う）
 */
public record AuxiliaryLoss(float oracleWeight, float resultWeight) {

  private static final Outcome[] OUTCOMES = Outcome.values();
  private static final CandidateTarget[] CANDIDATE_TARGETS = CandidateTarget.values();
  private static final ReadingTarget[] READING_TARGETS = ReadingTarget.values();
  private static final int CANDIDATE_HEADS = CANDIDATE_TARGETS.length;
  private static final int OPPONENTS = AuxiliaryLabels.OPPONENTS;

  static {
    for (ReadingTarget target : READING_TARGETS) {
      if (target.condition() == ReadingCondition.RON_TILE
          && (target.width() != ReadingTarget.DANGER.width() || target.loss() == Loss.SOFTMAX)) {
        throw new IllegalStateException(target + " cannot use the ron-tile condition");
      }
    }
  }

  /**
   * 重みを確かめる。
   *
   * @throws IllegalArgumentException 重みが負か有限でないとき
   */
  public AuxiliaryLoss {
    if (!(oracleWeight >= 0) || !Float.isFinite(oracleWeight)) {
      throw new IllegalArgumentException("oracle weight must be non-negative: " + oracleWeight);
    }
    if (!(resultWeight >= 0) || !Float.isFinite(resultWeight)) {
      throw new IllegalArgumentException("result weight must be non-negative: " + resultWeight);
    }
  }

  /** 既定の重み（α_o = α_r = 0.1）。 */
  public static AuxiliaryLoss standard() {
    return new AuxiliaryLoss(0.1f, 0.1f);
  }

  /**
   * 頭ごとの損失と合計。
   *
   * @param candidate 候補の頭ごとの L̄_h（スカラー）
   * @param reading 読みの頭ごとの L̄_h（スカラー）
   * @param total L_aux（K′ と α を掛けたもの）
   */
  public record Terms(
      Map<CandidateTarget, NDArray> candidate, Map<ReadingTarget, NDArray> reading, NDArray total) {}

  /**
   * 頭ごとの平均の分母 n_h（条件を満たす所の数）。微小バッチの教師から数えるか（{@link #of}）、詰め方の 1 塊を分けた部分には元の塊の数
   * （部分の数の和。{@link #plus}）を渡す。
   *
   * @param candidate 候補の頭ごとの、条件を満たす候補の数（{@link CandidateTarget} の順）
   * @param reading 読みの頭ごとの、条件を満たす（単位, 列）の数（{@link ReadingTarget} の順。softmax の頭は単位の数 × 幅）
   */
  public record Denominators(int[] candidate, int[] reading) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException 頭の数と合わないか、数が負のとき
     */
    public Denominators {
      if (candidate.length != CANDIDATE_TARGETS.length || reading.length != READING_TARGETS.length) {
        throw new IllegalArgumentException("denominators do not match the heads");
      }
      for (int count : candidate) {
        requireCount(count);
      }
      for (int count : reading) {
        requireCount(count);
      }
    }

    /**
     * 教師から数える。
     *
     * @param labels 教師
     * @return 分母
     */
    public static Denominators of(AuxiliaryLabels labels) {
      int[] candidate = new int[CANDIDATE_TARGETS.length];
      for (CandidateTarget target : CANDIDATE_TARGETS) {
        candidate[target.ordinal()] = count(candidateMask(target, labels.outcomes()));
      }
      int units = labels.readingRows().length * OPPONENTS;
      int[] reading = new int[READING_TARGETS.length];
      for (ReadingTarget target : READING_TARGETS) {
        reading[target.ordinal()] = count(readingMask(target, labels.readingValues(), units));
      }
      return new Denominators(candidate, reading);
    }

    /**
     * 頭ごとに足す（同じ塊を分けた部分の数から元の塊の数を作る）。
     *
     * @param other 足す分母
     * @return 和
     */
    public Denominators plus(Denominators other) {
      int[] candidates = new int[candidate.length];
      for (int head = 0; head < candidates.length; head++) {
        candidates[head] = Math.addExact(candidate[head], other.candidate[head]);
      }
      int[] readings = new int[reading.length];
      for (int head = 0; head < readings.length; head++) {
        readings[head] = Math.addExact(reading[head], other.reading[head]);
      }
      return new Denominators(candidates, readings);
    }

    private static void requireCount(int count) {
      if (count < 0) {
        throw new IllegalArgumentException("denominators must be non-negative: " + count);
      }
    }
  }

  /**
   * 送った教師と条件の印（{@link #stage} で載せ、{@link LossInputs} が送る）。条件を満たす所がこの微小バッチにない頭は印を持たず、その頭の
   * 項は 0。
   *
   * @param candidateTargets 候補の教師 [N_c, {@link CandidateTarget#WIDTH}]（候補がなければ null）
   * @param candidateMasks 候補の頭ごとの条件の印 [N_c]
   * @param readingTargets 読みの教師 [N_r, 3, {@link ReadingTarget#WIDTH}]（読みの行がなければ null）
   * @param readingMasks 読みの頭ごとの条件の印（softmax の頭は単位ごとの [N_r, 3]、ほかの頭は列ごとの [N_r, 3, 幅]）
   * @param denominators 頭ごとの平均の分母
   */
  record Inputs(
      Staging.Section candidateTargets,
      Map<CandidateTarget, Staging.Section> candidateMasks,
      Staging.Section readingTargets,
      Map<ReadingTarget, Staging.Section> readingMasks,
      Denominators denominators) {}

  /**
   * 教師と、条件を満たす所がある頭の条件の印を送る区画に載せる（送るのは呼び手）。値と形は損失がその場で作っていたものと同じ。
   *
   * @param labels 教師
   * @param denominators 頭ごとの平均の分母
   * @param staging 送る区画
   * @return 送った後に読む教師と条件の印
   * @throws IllegalArgumentException 分母が教師の条件を満たす所の数より小さいとき
   */
  static Inputs stage(AuxiliaryLabels labels, Denominators denominators, Staging staging) {
    int candidates = labels.candidateActions().length;
    Map<CandidateTarget, Staging.Section> candidateMasks = new EnumMap<>(CandidateTarget.class);
    for (CandidateTarget target : CANDIDATE_TARGETS) {
      float[] mask = candidateMask(target, labels.outcomes());
      if (covered(target, mask, denominators.candidate()[target.ordinal()]) > 0) {
        candidateMasks.put(target, staging.add(mask, new Shape(candidates)));
      }
    }
    int rows = labels.readingRows().length;
    int units = rows * OPPONENTS;
    Map<ReadingTarget, Staging.Section> readingMasks = new EnumMap<>(ReadingTarget.class);
    for (ReadingTarget target : READING_TARGETS) {
      float[] mask = readingMask(target, labels.readingValues(), units);
      if (covered(target, mask, denominators.reading()[target.ordinal()]) > 0) {
        readingMasks.put(
            target,
            target.loss() == Loss.SOFTMAX
                ? staging.add(unitMask(mask, target.width(), units), new Shape(rows, OPPONENTS))
                : staging.add(mask, new Shape(rows, OPPONENTS, target.width())));
      }
    }
    return new Inputs(
        candidates == 0
            ? null
            : staging.add(labels.candidateValues(), new Shape(candidates, CandidateTarget.WIDTH)),
        candidateMasks,
        rows == 0
            ? null
            : staging.add(labels.readingValues(), new Shape(rows, OPPONENTS, ReadingTarget.WIDTH)),
        readingMasks,
        denominators);
  }

  /**
   * 損失を計算する（網の出力が既にあるときの口。試験と点検が使い、送り出しの置き場は scope に付けて作る）。教師を送ってから
   * {@link #terms(NDManager, Outputs.Auxiliary, Inputs, float)} と同じ演算を通し、分母は教師から数える。
   *
   * @param scope 作った配列と送り出しの置き場を付ける管理元
   * @param outputs 補助の頭の出力
   * @param labels 教師
   * @param gain ゲイン K′（replay では 1）
   * @return 頭ごとの損失と合計
   */
  public Terms terms(NDManager scope, Outputs.Auxiliary outputs, AuxiliaryLabels labels, float gain) {
    Staging staging = new Staging();
    Inputs inputs = stage(labels, Denominators.of(labels), staging);
    staging.send(Uploads.on(scope), scope);
    return terms(scope, outputs, inputs, gain);
  }

  /**
   * 送った教師で損失を計算する。頭の項は条件の印を掛けた損失の和を分母 n_h で割る。
   *
   * @param scope 作った配列を付ける管理元
   * @param outputs 補助の頭の出力
   * @param inputs 送った教師と条件の印
   * @param gain ゲイン K′（replay では 1）
   * @return 頭ごとの損失と合計
   */
  Terms terms(NDManager scope, Outputs.Auxiliary outputs, Inputs inputs, float gain) {
    Map<CandidateTarget, NDArray> candidate = new EnumMap<>(CandidateTarget.class);
    NDArray candidateSum = scope.zeros(new Shape());
    for (CandidateTarget target : CANDIDATE_TARGETS) {
      Staging.Section mask = inputs.candidateMasks().get(target);
      NDArray term;
      if (mask == null) {
        term = scope.zeros(new Shape());
      } else {
        NDArray logits = section(outputs.candidates(), target.offset(), target.width());
        NDArray truth = section(inputs.candidateTargets().array(), target.offset(), target.width());
        NDArray units = unitLoss(target.loss(), target.reduction(), logits, truth, target.scale());
        term = units.mul(mask.array()).sum().div(inputs.denominators().candidate()[target.ordinal()]);
      }
      candidate.put(target, term);
      candidateSum = candidateSum.add(term);
    }

    Map<ReadingTarget, NDArray> reading = new EnumMap<>(ReadingTarget.class);
    NDArray readingSum = scope.zeros(new Shape());
    for (ReadingTarget target : READING_TARGETS) {
      Staging.Section mask = inputs.readingMasks().get(target);
      NDArray term;
      if (mask == null) {
        term = scope.zeros(new Shape());
      } else {
        float used = inputs.denominators().reading()[target.ordinal()];
        NDArray logits = section(outputs.reading(), target.offset(), target.width());
        NDArray truth = section(inputs.readingTargets().array(), target.offset(), target.width());
        if (target.loss() == Loss.SOFTMAX) {
          NDArray entropy = crossEntropy(logits, truth);
          term = entropy.mul(mask.array()).sum().div(used / target.width());
        } else {
          NDArray columns = columnLoss(target.loss(), logits, truth, target.scale());
          float denominator = target.reduction() == Reduction.MEAN ? used : used / target.width();
          term = columns.mul(mask.array()).sum().div(denominator);
        }
      }
      reading.put(target, term);
      readingSum = readingSum.add(term);
    }

    NDArray total =
        readingSum
            .mul(oracleWeight)
            .add(candidateSum.mul(resultWeight / CANDIDATE_HEADS))
            .mul(gain);
    return new Terms(
        Collections.unmodifiableMap(candidate), Collections.unmodifiableMap(reading), total);
  }

  /** 最後の軸の [offset, offset + width)。 */
  private static NDArray section(NDArray array, int offset, int width) {
    return array.get(new NDIndex("..., {}:{}", offset, offset + width));
  }

  /** 単位（候補）ごとの損失 [N]。 */
  private static NDArray unitLoss(
      Loss loss, Reduction reduction, NDArray logits, NDArray truth, float scale) {
    if (loss == Loss.SOFTMAX) {
      return crossEntropy(logits, truth);
    }
    NDArray columns = columnLoss(loss, logits, truth, scale);
    int last = columns.getShape().dimension() - 1;
    return reduction == Reduction.MEAN ? columns.mean(new int[] {last}) : columns.sum(new int[] {last});
  }

  /** 列ごとの損失（2 値の交差エントロピーか二乗誤差）。 */
  private static NDArray columnLoss(Loss loss, NDArray logits, NDArray truth, float scale) {
    return switch (loss) {
      case SIGMOID ->
          logits
              .maximum(0)
              .sub(logits.mul(truth))
              .add(logits.abs().neg().exp().add(1).log());
      case SQUARED -> logits.sub(truth.div(scale)).square();
      case SOFTMAX -> throw new IllegalArgumentException("softmax has no column loss");
    };
  }

  /** 最後の軸の softmax の交差エントロピー。 */
  private static NDArray crossEntropy(NDArray logits, NDArray truth) {
    int last = logits.getShape().dimension() - 1;
    return truth.mul(logits.logSoftmax(last)).sum(new int[] {last}).neg();
  }

  /** 候補の頭の条件の印 [N_c]（条件を満たす候補で 1）。 */
  private static float[] candidateMask(CandidateTarget target, int[] outcomes) {
    float[] mask = new float[outcomes.length];
    for (int c = 0; c < outcomes.length; c++) {
      if (target.condition().holds(OUTCOMES[outcomes[c]])) {
        mask[c] = 1f;
      }
    }
    return mask;
  }

  /** 読みの頭の列ごとの条件の印（[単位 × 幅]）。 */
  private static float[] readingMask(ReadingTarget target, float[] values, int units) {
    int width = target.width();
    float[] mask = new float[units * width];
    for (int u = 0; u < units; u++) {
      int base = u * ReadingTarget.WIDTH;
      for (int k = 0; k < width; k++) {
        boolean used =
            switch (target.condition()) {
              case ALWAYS -> true;
              case OPPONENT_TENPAI -> values[base + ReadingTarget.TENPAI.offset()] == 1f;
              case RON_TILE -> values[base + ReadingTarget.DANGER.offset() + k] == 1f;
            };
        mask[u * width + k] = used ? 1f : 0f;
      }
    }
    return mask;
  }

  /** 列ごとの印を単位ごとの印 [単位] にする（softmax の頭の条件は単位の全ての列で同じ）。 */
  private static float[] unitMask(float[] mask, int width, int units) {
    float[] unit = new float[units];
    for (int u = 0; u < units; u++) {
      unit[u] = mask[u * width];
    }
    return unit;
  }

  /** 印の条件を満たす所の数を数え、分母を超えないことを確かめる。 */
  private static int covered(Enum<?> head, float[] mask, int denominator) {
    int count = count(mask);
    if (count > denominator) {
      throw new IllegalArgumentException(
          head + " has " + count + " places for the denominator " + denominator);
    }
    return count;
  }

  /** 印の条件を満たす所（値 1）の数。 */
  private static int count(float[] mask) {
    int count = 0;
    for (float value : mask) {
      if (value == 1f) {
        count++;
      }
    }
    return count;
  }
}
