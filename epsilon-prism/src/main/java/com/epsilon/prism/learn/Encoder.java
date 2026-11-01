package com.epsilon.prism.learn;

import com.epsilon.prism.input.FrontierFeatures;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputWriter;
import com.epsilon.prism.input.Tokenizer;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.observe.Observation;
import com.epsilon.prism.rows.Row;
import java.util.ArrayList;
import java.util.List;

/**
 * 行の並びを 1 つの微小バッチに符号化する（符号化のスレッドに 1 つ）。行ごとに網の入力（{@link Inputs}）、差分整合の教師（{@link Teacher#values}
 * の値・λ_c・行の重み）、補助の頭の教師（{@link AuxiliaryTargets}）を作る。錨の s は行の貯蔵の番号で引いて教師に混ぜる。
 */
final class Encoder {

  private final Inputs inputs;
  private final FitSettings learner;
  private final AuxiliaryTargets targets;

  /**
   * 符号化の作業域を作る。
   *
   * @param plan 当てはめの計画（網の形の前線の腕の有無で、前線を書くかが決まる）
   */
  Encoder(FitPlan plan) {
    inputs = new Inputs(plan.shape().frontier());
    learner = plan.learner();
    targets =
        AuxiliaryTargets.needed(plan.auxiliary()) ? new AuxiliaryTargets(plan.auxiliary()) : null;
  }

  /**
   * 行を塊に書く。塊の計画は、この順の行で足して封じたもの。
   *
   * @param batch 封じた塊（前線の腕なら前線の塊つき）
   * @param rows 塊の行の順の行
   * @param ids 行の貯蔵の番号（rows と同じ順。錨の s を引く）
   * @param teacher 教師の作り方
   * @return 微小バッチ
   * @throws IllegalArgumentException 行の数が塊の計画か番号の数と合わないか、塊の前線の有無が計画と合わないとき
   */
  Microbatch encode(HostInputs batch, List<Row> rows, int[] ids, Teacher teacher) {
    if (rows.size() != batch.plan().rows() || rows.size() != ids.length) {
      throw new IllegalArgumentException(
          rows.size() + " rows with " + ids.length + " ids for a batch planned with "
              + batch.plan().rows());
    }
    if (targets != null) {
      targets.clear();
    }
    List<ValueLabel> labels = new ArrayList<>(rows.size());
    for (int row = 0; row < rows.size(); row++) {
      Row source = rows.get(row);
      Observation o = inputs.write(batch, row, source);
      labels.add(
          new ValueLabel(
              teacher.values(source, ids[row]),
              learner.lambda(source.decisionClass()),
              inclusionWeight(source)));
      if (targets != null) {
        targets.add(row, batch.plan().firstAction(row), source, o, inputs.tokenizer);
      }
    }
    return Microbatch.of(batch, labels, targets == null ? null : targets.labels());
  }

  /**
   * 収集の測定の配分の重み: 配分して集めた主経路のラベル行（propensity が相対の包含確率 r(o)）は 1/r（Horvitz–Thompson の重みで、重みつきの
   * 勾配の期待値は一様な率 ρ_c で集めたときと同じ）、ほかの行は 1。
   */
  static float inclusionWeight(Row row) {
    float relative = row.propensity();
    return row.source() == Row.Source.MAIN && relative > 0 && Float.isFinite(relative)
        ? 1f / relative
        : 1f;
  }

  /**
   * 行の網の入力の書き手（スレッドに 1 つ）: O → {@link Tokenizer} → 基準の入力、前線の腕なら {@link FrontierFeatures} → 前線の塊。前線は
   * O の純関数なので、行を集め直さずに学習のときに作る。生徒の微小バッチと錨の s（{@link AnchorScores}）は同じ書き手で書くので、同じ行の
   * 入力のバイトは同じ。
   */
  static final class Inputs {

    private final Tokenizer tokenizer = new Tokenizer();
    private final InputWriter writer = new InputWriter();
    private final Observation observation = new Observation();

    /** 前線の書き手（前線のない腕は null）。 */
    private final FrontierFeatures frontier;

    /**
     * 書き手を作る。
     *
     * @param frontier 前線の腕なら true
     */
    Inputs(boolean frontier) {
      this.frontier = frontier ? new FrontierFeatures() : null;
    }

    /**
     * 行の O を束ねる（塊の計画に足す）。
     *
     * @param source 行
     * @return 行に束ねた O（次に束ねるまで有効）
     */
    Observation bind(Row source) {
      return observation.bind(source.observation(), 0);
    }

    /**
     * 行の入力を塊の row 番目に書く（塊の計画は、この行を row 番目に足したもの）。
     *
     * @param batch 塊
     * @param row 塊の中の行の位置
     * @param source 行
     * @return 行に束ねた O（次に書くまで有効）
     * @throws IllegalArgumentException 塊の前線の有無が書き手と合わないとき
     */
    Observation write(HostInputs batch, int row, Row source) {
      if ((frontier != null) != (batch.frontier() != null)) {
        throw new IllegalArgumentException(
            "the batch and the fit plan disagree on the frontier arm");
      }
      Observation o = bind(source);
      tokenizer.write(o, writer.bind(batch.input(), row));
      if (frontier != null) {
        frontier.write(o, batch.frontier(), row);
      }
      return o;
    }
  }
}
