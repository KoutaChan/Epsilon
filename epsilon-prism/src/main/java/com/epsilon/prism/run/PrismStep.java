package com.epsilon.prism.run;

import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepKind;
import java.nio.file.Path;

/** PRISM の系譜の段。宣言の順に回し、各段の前と後がセーフポイントになる。 */
public enum PrismStep implements StepKind<PrismIteration> {
  /** 反復の Φ を決める（自分の GRP を学ぶ run なら、バックグラウンドの作業が前の反復の区切りを終えるのを待つ）。同じファイルからは同じ答えを出すので、jar に依らない。 */
  PHI(new PhiStep(), PhiReport.class, false),
  /** 全席がバージョン start で打って行を集め、コホートを確定する。 */
  COLLECT(new CollectStep(), CollectReport.class, true),
  /** 帳簿を当てはめ、生徒を学習する。 */
  LEARN(new LearnStep(), LearnReport.class, true),
  /** 次のバージョンを公開する（方策の記憶を使うなら平均する）。 */
  PUBLISH(new PublishStep(), PublishReport.class, true);

  /** 書きかけの段の出力に付ける、名前の末尾。 */
  static final String PART = ".part";

  /** 段の出力に入る、段の報告のファイル。 */
  static final String REPORT = "report.json";

  private final Step<PrismIteration> step;
  private final Class<? extends Record> metrics;
  private final boolean jarBound;

  PrismStep(Step<PrismIteration> step, Class<? extends Record> metrics, boolean jarBound) {
    this.step = step;
    this.metrics = metrics;
    this.jarBound = jarBound;
  }

  @Override
  public Step<PrismIteration> create(PrismIteration iteration) {
    return step;
  }

  @Override
  public Class<? extends Record> metrics() {
    return metrics;
  }

  @Override
  public boolean jarBound() {
    return jarBound;
  }

  /** 完成したときの出力の名前から、書きかけの出力のパス（{@code <名前>.part}）を返す。 */
  static Path part(Path done) {
    return done.resolveSibling(done.getFileName() + PART);
  }
}
