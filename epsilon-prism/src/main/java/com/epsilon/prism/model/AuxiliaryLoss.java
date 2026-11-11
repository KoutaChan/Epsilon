package com.epsilon.prism.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.Shape;
import com.epsilon.agent.runtime.PinnedUploads;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.auxiliary.TargetSchema;
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
 * 補助のヘッドの損失。
 *
 * <pre>
 * L_aux = K′·[ α_o·Σ_{h∈読み} L̄_h + α_r·(1/8)·Σ_{h∈候補} L̄_h ]
 * L̄_h  = （ヘッド h の条件を満たす所の損失の和）/ n_h（マイクロバッチに条件を満たす所がなければ 0）
 * </pre>
 *
 * <p>ヘッドごとの損失の種類、列のまとめ方、条件、正規化は {@link CandidateTarget}・{@link ReadingTarget} に従う。条件を満たさない所は、
 * 損失にも分母にも入れない。列の平均（{@link Reduction#MEAN}）は、条件を満たす列の平均を取る。列の和（{@link Reduction#SUM}）は、
 * 条件を満たす単位ごとに列の和を取り、それを平均する。softmax は単位ごとの交差エントロピーの平均。2 値の交差エントロピーと二乗誤差は
 * FP32 のロジットで計算し、点数の大きな残差も切り詰めない。
 *
 * <p>分母 n_h（条件を満たす所の数）は {@link Denominators} で渡す。ふつうはマイクロバッチの教師から数えた数を渡す。1 つのバッチを
 * 分けて詰めた部分には、元のバッチの数を渡す。こうすると、部分ごとの項（部分の和 / 元のバッチの n_h）を足したものが元のバッチの項に
 * なる（違うのは FP32 の和の順だけ）。教師と条件のマスクは、順伝播の前に {@link LossInputs} が送る（{@link #stage}）。損失は送った
 * 配列を読む。
 *
 * @param oracleWeight 読みのヘッドの重み α_o
 * @param resultWeight 候補の結果のヘッドの重み α_r（候補のヘッドの数で割って使う）
 */
public record AuxiliaryLoss(float oracleWeight, float resultWeight) {

  private static final Outcome[] OUTCOMES = Outcome.values();
  private static final CandidateTarget[] CANDIDATE_TARGETS = CandidateTarget.values();
  private static final ReadingTarget[] READING_TARGETS = ReadingTarget.values();
  private static final int CANDIDATE_HEADS = CANDIDATE_TARGETS.length;

  static {
    for (Variant variant : Variant.values()) {
      TargetSchema heads = TargetSchema.of(variant);
      for (ReadingTarget target : READING_TARGETS) {
        if (target.condition() == ReadingCondition.RON_TILE
            && (heads.width(target) != heads.width(ReadingTarget.DANGER) || target.loss() == Loss.SOFTMAX)) {
          throw new IllegalStateException(target + " cannot use the ron-tile condition");
        }
      }
    }
  }

  /**
   * 重みを確かめる。
   *
   * @throws IllegalArgumentException 重みが負か、有限でないとき
   */
  public AuxiliaryLoss {
    if (!(oracleWeight >= 0) || !Float.isFinite(oracleWeight)) {
      throw new IllegalArgumentException("oracle weight must be non-negative: " + oracleWeight);
    }
    if (!(resultWeight >= 0) || !Float.isFinite(resultWeight)) {
      throw new IllegalArgumentException("result weight must be non-negative: " + resultWeight);
    }
  }

  /** 既定の重み（α_o = α_r = 0.1）で作る。 */
  public static AuxiliaryLoss standard() {
    return new AuxiliaryLoss(0.1f, 0.1f);
  }

  /**
   * ヘッドごとの損失と合計。
   *
   * @param candidate 候補のヘッドごとの L̄_h（スカラー）
   * @param reading 読みのヘッドごとの L̄_h（スカラー）
   * @param total L_aux（K′ と α を掛けた値）
   */
  public record Terms(
      Map<CandidateTarget, NDArray> candidate, Map<ReadingTarget, NDArray> reading, NDArray total) {}

  /**
   * ヘッドごとの平均の分母 n_h（条件を満たす所の数）。ふつうはマイクロバッチの教師から数える（{@link #of}）。1 つのバッチを分けて
   * 詰めた部分には、部分の数を足した元のバッチの数（{@link #plus}）を渡す。
   *
   * @param candidate 候補のヘッドごとに、条件を満たす候補の数（{@link CandidateTarget} の順）
   * @param reading 読みのヘッドごとに、条件を満たす（単位, 列）の数（{@link ReadingTarget} の順。softmax のヘッドは単位の数 × 幅）
   */
  public record Denominators(int[] candidate, int[] reading) {

    /**
     * 形を確かめる。
     *
     * @throws IllegalArgumentException ヘッドの数と合わないか、負の数があるとき
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
      TargetSchema heads = labels.heads();
      int units = labels.readingRows().length * heads.opponents();
      int[] reading = new int[READING_TARGETS.length];
      for (ReadingTarget target : READING_TARGETS) {
        reading[target.ordinal()] = count(readingMask(heads, target, labels.readingValues(), units));
      }
      return new Denominators(candidate, reading);
    }

    /**
     * ヘッドごとに足す。同じバッチを分けた部分の数から、元のバッチの数を作るのに使う。
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
   * 送った教師と条件のマスク。{@link #stage} で載せ、{@link LossInputs} が送る。このマイクロバッチに条件を満たす所がないヘッドは
   * マスクを持たず、そのヘッドの項は 0。
   *
   * @param heads 補助のヘッドの形
   * @param candidateTargets 候補の教師 [N_c, {@link TargetSchema#candidateWidth}]（候補がなければ null）
   * @param candidateMasks 候補のヘッドごとの条件のマスク [N_c]
   * @param readingTargets 読みの教師 [N_r, 相手の数, {@link TargetSchema#readingWidth}]（読みの行がなければ null）
   * @param readingMasks 読みのヘッドごとの条件のマスク（softmax のヘッドは単位ごとの [N_r, 相手の数]、ほかのヘッドは列ごとの
   *     [N_r, 相手の数, 幅]）
   * @param denominators ヘッドごとの平均の分母
   */
  record Inputs(
      TargetSchema heads,
      Staging.Section candidateTargets,
      Map<CandidateTarget, Staging.Section> candidateMasks,
      Staging.Section readingTargets,
      Map<ReadingTarget, Staging.Section> readingMasks,
      Denominators denominators) {}

  /**
   * 教師と、条件を満たす所があるヘッドの条件のマスクを、送る領域に載せる。送るのは呼び出し元。
   *
   * @param labels 教師
   * @param denominators ヘッドごとの平均の分母
   * @param staging 載せる先
   * @return 送った後に読む教師と条件のマスク
   * @throws IllegalArgumentException 分母が、教師の中で条件を満たす所の数より小さいとき
   */
  static Inputs stage(AuxiliaryLabels labels, Denominators denominators, Staging staging) {
    TargetSchema heads = labels.heads();
    int opponents = heads.opponents();
    int candidates = labels.candidateActions().length;
    Map<CandidateTarget, Staging.Section> candidateMasks = new EnumMap<>(CandidateTarget.class);
    for (CandidateTarget target : CANDIDATE_TARGETS) {
      float[] mask = candidateMask(target, labels.outcomes());
      if (covered(target, mask, denominators.candidate()[target.ordinal()]) > 0) {
        candidateMasks.put(target, staging.add(mask, new Shape(candidates)));
      }
    }
    int rows = labels.readingRows().length;
    int units = rows * opponents;
    Map<ReadingTarget, Staging.Section> readingMasks = new EnumMap<>(ReadingTarget.class);
    for (ReadingTarget target : READING_TARGETS) {
      float[] mask = readingMask(heads, target, labels.readingValues(), units);
      if (covered(target, mask, denominators.reading()[target.ordinal()]) > 0) {
        readingMasks.put(
            target,
            target.loss() == Loss.SOFTMAX
                ? staging.add(unitMask(mask, heads.width(target), units), new Shape(rows, opponents))
                : staging.add(mask, new Shape(rows, opponents, heads.width(target))));
      }
    }
    return new Inputs(
        heads,
        candidates == 0
            ? null
            : staging.add(labels.candidateValues(), new Shape(candidates, heads.candidateWidth())),
        candidateMasks,
        rows == 0
            ? null
            : staging.add(labels.readingValues(), new Shape(rows, opponents, heads.readingWidth())),
        readingMasks,
        denominators);
  }

  /**
   * ネットワークの出力が既にあるときに、損失を計算する。テストと点検で使う。送るためのバッファは scope に付けて作る。教師を送ってから
   * {@link #terms(NDManager, Outputs.Auxiliary, Inputs, float)} と同じ演算を通す。分母は教師から数える。
   *
   * @param scope 作った配列と、送るためのバッファを付けるマネージャー
   * @param outputs 補助のヘッドの出力
   * @param labels 教師
   * @param gain ゲイン K′（replay では 1）
   * @return ヘッドごとの損失と合計
   */
  public Terms terms(NDManager scope, Outputs.Auxiliary outputs, AuxiliaryLabels labels, float gain) {
    Staging staging = new Staging();
    Inputs inputs = stage(labels, Denominators.of(labels), staging);
    staging.send(PinnedUploads.on(scope), scope);
    return terms(scope, outputs, inputs, gain);
  }

  /**
   * 送った教師で損失を計算する。各ヘッドの項は、条件のマスクを掛けた損失の和を分母 n_h で割った値。
   *
   * @param scope 作った配列を付けるマネージャー
   * @param outputs 補助のヘッドの出力
   * @param inputs 送った教師と条件のマスク
   * @param gain ゲイン K′（replay では 1）
   * @return ヘッドごとの損失と合計
   */
  Terms terms(NDManager scope, Outputs.Auxiliary outputs, Inputs inputs, float gain) {
    TargetSchema heads = inputs.heads();
    Map<CandidateTarget, NDArray> candidate = new EnumMap<>(CandidateTarget.class);
    NDArray candidateSum = scope.zeros(new Shape());
    for (CandidateTarget target : CANDIDATE_TARGETS) {
      Staging.Section mask = inputs.candidateMasks().get(target);
      NDArray term;
      if (mask == null) {
        term = scope.zeros(new Shape());
      } else {
        NDArray logits = section(outputs.candidates(), heads.offset(target), heads.width(target));
        NDArray truth = section(inputs.candidateTargets().array(), heads.offset(target), heads.width(target));
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
        NDArray logits = section(outputs.reading(), heads.offset(target), heads.width(target));
        NDArray truth = section(inputs.readingTargets().array(), heads.offset(target), heads.width(target));
        if (target.loss() == Loss.SOFTMAX) {
          NDArray entropy = crossEntropy(logits, truth);
          term = entropy.mul(mask.array()).sum().div(used / heads.width(target));
        } else {
          NDArray columns = columnLoss(target.loss(), logits, truth, target.scale());
          float denominator = target.reduction() == Reduction.MEAN ? used : used / heads.width(target);
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

  /** 最後の軸の [offset, offset + width) を切り出す。 */
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

  /** 候補のヘッドの条件のマスク [N_c]（条件を満たす候補で 1）。 */
  private static float[] candidateMask(CandidateTarget target, int[] outcomes) {
    float[] mask = new float[outcomes.length];
    for (int c = 0; c < outcomes.length; c++) {
      if (target.condition().holds(OUTCOMES[outcomes[c]])) {
        mask[c] = 1f;
      }
    }
    return mask;
  }

  /** 読みのヘッドの、列ごとの条件のマスク（[単位 × 幅]）。 */
  private static float[] readingMask(TargetSchema heads, ReadingTarget target, float[] values, int units) {
    int width = heads.width(target);
    float[] mask = new float[units * width];
    for (int u = 0; u < units; u++) {
      int base = u * heads.readingWidth();
      for (int k = 0; k < width; k++) {
        boolean used =
            switch (target.condition()) {
              case ALWAYS -> true;
              case OPPONENT_TENPAI -> values[base + heads.offset(ReadingTarget.TENPAI)] == 1f;
              case RON_TILE -> values[base + heads.offset(ReadingTarget.DANGER) + k] == 1f;
            };
        mask[u * width + k] = used ? 1f : 0f;
      }
    }
    return mask;
  }

  /** 列ごとのマスクを、単位ごとのマスク [単位] にする（softmax のヘッドの条件は、単位の全ての列で同じ）。 */
  private static float[] unitMask(float[] mask, int width, int units) {
    float[] unit = new float[units];
    for (int u = 0; u < units; u++) {
      unit[u] = mask[u * width];
    }
    return unit;
  }

  /** マスクで条件を満たす所を数え、分母を超えないことを確かめる。 */
  private static int covered(Enum<?> head, float[] mask, int denominator) {
    int count = count(mask);
    if (count > denominator) {
      throw new IllegalArgumentException(
          head + " has " + count + " places for the denominator " + denominator);
    }
    return count;
  }

  /** マスクで条件を満たす所（値 1）の数。 */
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
