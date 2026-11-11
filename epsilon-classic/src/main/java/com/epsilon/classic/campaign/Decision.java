package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.Alarm;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.Estimate;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import java.util.List;
import java.util.Objects;

/**
 * 昇格の段の判定。王者の保存先と一時停止の要求に当てる前に work に書き、段を当て直すときは書いた判定を読んで同じものを当てる。
 *
 * @param action 行うこと
 * @param duel 決闘のジョブの名前
 * @param candidate 候補
 * @param check 確かめの卓で候補を測るジョブの名前（確かめの卓が無ければ無い）
 * @param confirm 確かめの卓の、候補 − 王者の牌山ごとの対の差（判定に使う量の、候補が良い向き）の平均と固定の 99% の区間。確かめの卓で
 *     どちらも測れたときだけにある
 * @param alarms この決闘で出したアラート（王者に悪いと、悪いが続く）
 */
public record Decision(
    Action action,
    String duel,
    Champion candidate,
    @Key(doc = "確かめの卓で候補を測るジョブの名前", presence = Presence.ABSENT) String check,
    @Key(doc = "確かめの卓の、候補 − 王者の対の差の平均と区間", presence = Presence.ABSENT) Estimate confirm,
    List<Alarm> alarms) {

  /** 昇格の段が行うこと。 */
  public enum Action {
    /** 王者の保存先が空なので、候補を初めの王者にする。確かめの卓があれば、その卓で王者を測っておく。 */
    CROWN,
    /** 決闘で良く、確かめの卓も通ったので、候補を王者にする。前の王者は過去の王者になる。 */
    PROMOTE,
    /** 決闘では良いが、確かめの卓で外れたので、王者を据え置く。 */
    NOT_CONFIRMED,
    /** 王者に悪いが続いたので、王者を据え置いて run を一時停止し、持ち主の再開を待つ。 */
    HOLD,
    /** 王者を据え置いて続ける（互角・未決・FAILED の決闘と、報告だけの悪い）。 */
    KEEP
  }

  public Decision {
    Objects.requireNonNull(action, "action");
    Objects.requireNonNull(duel, "duel");
    Objects.requireNonNull(candidate, "candidate");
    alarms = List.copyOf(alarms);
    if (action == Action.NOT_CONFIRMED && check == null) {
      throw new IllegalArgumentException("A candidate is not confirmed only by a confirm table: " + duel);
    }
  }
}
