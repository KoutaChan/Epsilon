package com.epsilon.classic.campaign;

import com.epsilon.agent.run.Step;
import com.epsilon.agent.run.StepKind;
import java.nio.file.Path;

/** classic の campaign の段。macro ごとに宣言の順に回し、段の前と後がセーフポイント。 */
public enum CampaignStep implements StepKind<CampaignIteration> {
  /** 学習器のバージョンで自己対局を打ち、学習器の席の軌跡と迷いの分布をコホートに確定する。 */
  COLLECT(new CollectStep(), CollectReport.class),
  /** コホートで学習器を学ばせ、学んだ重みと途中の状態を確定して、次のバージョンを公開する。 */
  LEARN(new LearnStep(), LearnReport.class),
  /** 公開したバージョンを候補にした王者との決闘。王者を替える卓をこの macro で打つときだけ行う。 */
  DUEL(new DuelStep<>(), DuelReport.class),
  /** 決闘の判定を、アラートの行・王者の保存先・一時停止の要求に当てる。決闘の段と同じ macro だけで行う。 */
  DECIDE(new DecideStep<>(), Decision.class);

  /** 段の出力の書きかけの名前の末尾。 */
  static final String PART = ".part";

  /** 段の出力の中の、段の報告のファイル。 */
  static final String REPORT = "report.json";

  private final Step<CampaignIteration> step;
  private final Class<? extends Record> metrics;

  CampaignStep(Step<CampaignIteration> step, Class<? extends Record> metrics) {
    this.step = step;
    this.metrics = metrics;
  }

  @Override
  public Step<CampaignIteration> create(CampaignIteration iteration) {
    return step;
  }

  @Override
  public Class<? extends Record> metrics() {
    return metrics;
  }

  /** 完了した名前の出力の書きかけ（{@code <名前>.part}）。 */
  static Path part(Path done) {
    return done.resolveSibling(done.getFileName() + PART);
  }
}
