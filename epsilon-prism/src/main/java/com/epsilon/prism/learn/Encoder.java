package com.epsilon.prism.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.input.HostInputs;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.ValueLabel;
import com.epsilon.prism.rows.Row;
import com.epsilon.prism.variant.PrismVariant;
import java.util.ArrayList;
import java.util.List;

/**
 * 行の並びを 1 つのマイクロバッチにエンコードする。エンコードのスレッドごとに 1 つ使う。行ごとに、ネットワークの入力（{@link Inputs}）、
 * 差分整合の教師（{@link Teacher#values} の値・λ_c・行の重み）、補助のヘッドの教師（ルールごとの {@link TargetWriter}）を作る。
 * アンカーの s は、行のストアの番号で引いて教師に混ぜる。
 */
final class Encoder {

  private final Inputs inputs;
  private final FitSettings learner;
  private final TargetWriter targets;

  /**
   * エンコードの作業領域を作る。
   *
   * @param plan 学習の計画
   * @param variant 行のルール（行のストアのルール）
   */
  Encoder(FitPlan plan, Variant variant) {
    inputs = new Inputs(variant);
    learner = plan.learner();
    targets =
        TargetWriter.needed(plan.auxiliary())
            ? PrismVariant.of(variant).targets(plan.auxiliary())
            : null;
  }

  /**
   * 行をバッチに書く。バッチの計画は、この順に行を足して確定したもの。
   *
   * @param batch 確定したバッチ
   * @param rows 行（バッチの中の順）
   * @param ids 行のストアの番号（行と同じ順。アンカーの s を引くのに使う）
   * @param teacher 教師の作り方
   * @return マイクロバッチ
   * @throws IllegalArgumentException 行の数がバッチの計画か番号の数と合わないとき
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
      labels.add(new ValueLabel(teacher.values(source, ids[row]), learner.lambda(source.decisionClass()), 1f));
      if (targets != null) {
        targets.add(row, batch.plan().firstAction(row), source, o, inputs.writer);
      }
    }
    return Microbatch.of(batch, labels, targets == null ? null : targets.labels());
  }

  /**
   * 行からネットワークの入力を書くライター。スレッドごとに 1 つ使う。O を入力形式の 1 行のライターに渡して書く。生徒のマイクロバッチと
   * アンカーの s（{@link AnchorScores}）は同じライターで書くので、同じ行なら入力のバイト列も同じになる。
   */
  static final class Inputs {

    private final InputFormat.RowWriter writer;
    private final Observation observation;

    /**
     * ライターを作る。
     *
     * @param variant 行のルール
     */
    Inputs(Variant variant) {
      writer = InputFormat.of(variant).writer();
      observation = new Observation(variant);
    }

    /**
     * 行の O を束ねる。バッチの計画に行を足すときに使う。
     *
     * @param source 行
     * @return 行に束ねた O（次に束ねるまで有効）
     */
    Observation bind(Row source) {
      return observation.bind(source.observation(), 0);
    }

    /**
     * 行の入力をバッチの {@code row} 番目に書く。バッチの計画は、この行を {@code row} 番目に足したもの。
     *
     * @param batch バッチ
     * @param row バッチの中の行の位置
     * @param source 行
     * @return 行に束ねた O（次に書くまで有効）
     */
    Observation write(HostInputs batch, int row, Row source) {
      Observation o = bind(source);
      writer.write(o, batch, row);
      return o;
    }
  }
}
