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
 * 行の並びを 1 つの微小バッチに符号化する（符号化のスレッドに 1 つ）。行ごとに O → {@link Tokenizer} → 基準の入力、前線の腕なら
 * {@link FrontierFeatures} → 前線の塊、差分整合の教師（{@link Teacher#values} の値・λ_c・行の重み）、補助の頭の教師（{@link AuxiliaryTargets}）
 * を作る。前線は O の純関数なので、行を集め直さずに学習のときに作る。錨は入力を書いた後に混ぜる（{@link Teacher#mix}）。
 */
final class Encoder {

  private final Tokenizer tokenizer = new Tokenizer();
  private final InputWriter writer = new InputWriter();
  private final Observation observation = new Observation();
  private final FitSettings learner;
  private final Teacher teacher;
  private final AuxiliaryTargets targets;

  /** 前線の書き手（前線のない腕は null）。 */
  private final FrontierFeatures frontier;

  /**
   * 符号化の作業域を作る。
   *
   * @param plan 当てはめの計画（網の形の前線の腕の有無で、前線を書くかが決まる）
   * @param teacher 教師の作り方
   */
  Encoder(FitPlan plan, Teacher teacher) {
    learner = plan.learner();
    this.teacher = teacher;
    targets =
        AuxiliaryTargets.needed(plan.auxiliary()) ? new AuxiliaryTargets(plan.auxiliary()) : null;
    frontier = plan.shape().frontier() ? new FrontierFeatures() : null;
  }

  /**
   * 行を塊に書く。塊の計画は、この順の行で足して封じたもの。
   *
   * @param batch 封じた塊（前線の腕なら前線の塊つき）
   * @param rows 塊の行の順の行
   * @return 微小バッチ
   * @throws IllegalArgumentException 行の数が塊の計画と合わないか、塊の前線の有無が計画と合わないとき
   */
  Microbatch encode(HostInputs batch, List<Row> rows) {
    if (rows.size() != batch.plan().rows()) {
      throw new IllegalArgumentException(
          rows.size() + " rows for a batch planned with " + batch.plan().rows());
    }
    if ((frontier != null) != (batch.frontier() != null)) {
      throw new IllegalArgumentException("the batch and the fit plan disagree on the frontier arm");
    }
    if (targets != null) {
      targets.clear();
    }
    List<ValueLabel> labels = new ArrayList<>(rows.size());
    for (int row = 0; row < rows.size(); row++) {
      Row source = rows.get(row);
      Observation o = observation.bind(source.observation(), 0);
      tokenizer.write(o, writer.bind(batch.input(), row));
      if (frontier != null) {
        frontier.write(o, batch.frontier(), row);
      }
      labels.add(
          new ValueLabel(
              teacher.values(source),
              learner.lambda(source.decisionClass()),
              inclusionWeight(source)));
      if (targets != null) {
        targets.add(row, batch.plan().firstAction(row), source, o, tokenizer);
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
}
