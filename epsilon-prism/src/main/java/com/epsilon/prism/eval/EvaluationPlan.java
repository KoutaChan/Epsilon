package com.epsilon.prism.eval;

import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.prism.core.TwinDiscard;
import java.util.Objects;

/**
 * 評価のジョブの計画（job.json の plan）。ジョブを初めて始めるときに決め、後からは変えない。続きもこの計画で打つので、同じ牌山なら
 * 同じ判定になる。
 *
 * @param candidate 候補の打ち手（決闘の計画の候補と同じ）
 * @param duel 相手と打つ決闘の計画。王者の保存先が空のときの王者の卓では null で、ジョブは打たずに候補を初めの王者にする
 * @param jar 計画を決めた JVM の jar の ID（記録だけ）
 * @param twinDiscard PRISM の打ち手が、同値な打牌のうち切る方。切る方の違うジョブの結果とは、牌山ごとに対にしない
 */
public record EvaluationPlan(
    DuelSetup.Entrant candidate,
    @Key(doc = "相手と打つ決闘の計画", presence = Presence.ABSENT) DuelSetup duel,
    String jar,
    TwinDiscard twinDiscard) {

  public EvaluationPlan {
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(jar, "jar");
    Objects.requireNonNull(twinDiscard, "twinDiscard");
    if (duel != null && !duel.candidate().equals(candidate)) {
      throw new IllegalArgumentException(
          "The duel of an evaluation plan seats " + duel.candidate().label() + ", not the candidate " + candidate.label());
    }
  }

  /** 相手と打たずに、候補を初めの王者にする計画なら true。 */
  public boolean crowns() {
    return duel == null;
  }
}
