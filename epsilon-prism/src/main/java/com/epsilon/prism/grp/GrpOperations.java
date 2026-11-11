package com.epsilon.prism.grp;

import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.grp.train.GrpInit;
import java.util.Objects;

/**
 * 自分の GRP の運用の設定。主経路の半荘を記録し、確定したコホートで学び直し、関門に合格した候補を使い始める。
 *
 * @param monitor 主経路の半荘を記録するだけか、裏で採点もするか（記録しない設定は選べない）
 * @param refit いつ学び直すか
 * @param switching 関門に合格した候補をどう使い始めるか
 * @param init 学び直しの始め方（使用中の Φ が同じ形のネットワークでなければ、ゼロから学ぶ）
 * @param halfLife 学び直しでコホートに掛ける重みの半減期（コホートの数）
 * @param window 学び直しに使うコホートの数の上限
 * @param trustGames 点数だけの Φ を初めて置き換えるのに、学び直しの範囲に要る半荘の数
 * @param shadow 影として採点する Φ
 * @param threads 学び直しの間に torch が使う演算スレッドの数
 * @param training 学び直しの学び方
 */
public record GrpOperations(
    @Key(doc = "主経路の半荘を記録するだけか、裏で採点もするか", value = "SCORE") PhiMonitor monitor,
    @Key(doc = "学び直すタイミング", value = "SCHEDULE") GrpRefit refit,
    @Key(doc = "関門に合格した候補の使い始め方", value = "AUTO") GrpSwitch switching,
    @Key(doc = "学び直しの始め方", value = "WARM") GrpInit init,
    @Key(doc = "学び直しでコホートに掛ける重みの半減期（コホートの数）", value = "2.0") @Range(min = 0, minInclusive = false)
        double halfLife,
    @Key(doc = "学び直しに使うコホートの数の上限", value = "64") @Range(min = 1) int window,
    @Key(doc = "点数だけの Φ を初めて置き換えるのに、学び直しの範囲に要る半荘の数", value = "16384") @Range(min = 1) int trustGames,
    @Key(doc = "影として採点する Φ", value = "SELF_ONLY") PhiShadowReference shadow,
    @Key(doc = "学び直しの間に torch が使う演算スレッドの数", value = "4", change = Change.MACHINE) @Range(min = 1)
        int threads,
    @Key(doc = "学び直しの学び方") GrpTraining training) {

  public GrpOperations {
    Objects.requireNonNull(monitor, "monitor");
    Objects.requireNonNull(refit, "refit");
    Objects.requireNonNull(switching, "switching");
    Objects.requireNonNull(init, "init");
    Objects.requireNonNull(shadow, "shadow");
    Objects.requireNonNull(training, "training");
    if (!monitor.records()) {
      throw new InvalidSetting("monitor", "the self-play GRP learns from the recorded games, so monitor cannot be OFF");
    }
  }
}
