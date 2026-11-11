package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.eval.Verdict;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.util.Objects;

/**
 * 決闘の段の報告。
 *
 * @param job 決闘のジョブの名前
 * @param candidate 候補（この macro の学習器のネットワーク）
 * @param champion 相手にした王者。王者の保存先が空で、候補を初めの王者にするときは無い
 * @param verdict 判定（候補が良い向き）。FAILED になった決闘は未決。王者がいなければ無い
 * @param failed 決闘のジョブが、落ちた回数の上限で FAILED になったか
 * @param better 判定に使った量の、候補が良い向きの平均と区間。打ち終えた決闘だけにある
 */
public record DuelReport(
    String job,
    Champion candidate,
    @Key(doc = "相手にした王者", presence = Presence.ABSENT) Champion champion,
    @Key(doc = "判定", presence = Presence.ABSENT) Verdict verdict,
    boolean failed,
    @Key(doc = "判定に使った量の、候補が良い向きの平均と区間", presence = Presence.ABSENT) Estimate better) {

  public DuelReport {
    Objects.requireNonNull(job, "job");
    Objects.requireNonNull(candidate, "candidate");
    if ((champion == null) != (verdict == null)) {
      throw new IllegalArgumentException("A duel has a verdict exactly when it has a champion: " + job);
    }
    if (failed && (verdict != Verdict.UNDECIDED || better != null)) {
      throw new IllegalArgumentException("A failed duel is undecided and has no estimate: " + job);
    }
    if ((better == null) != (champion == null || failed)) {
      throw new IllegalArgumentException("A played duel has an estimate, and only a played duel: " + job);
    }
  }

  /** 王者がいないので打たずに、候補を初めの王者にする報告。 */
  static DuelReport uncontested(String job, Champion candidate) {
    return new DuelReport(job, candidate, null, null, false, null);
  }

  /** 打ち終えた決闘の報告。 */
  static DuelReport played(String job, Champion candidate, Champion champion, DuelJobs.Played played) {
    return new DuelReport(
        job, candidate, champion, played.result().verdict(), false, DuelJobs.better(played.result()));
  }

  /** ジョブが FAILED になった決闘の報告（判定は未決）。 */
  static DuelReport fell(String job, Champion candidate, Champion champion) {
    return new DuelReport(job, candidate, champion, Verdict.UNDECIDED, true, null);
  }

  /** @return 王者の保存先が空で、候補を初めの王者にするなら true */
  public boolean crowns() {
    return champion == null;
  }
}
