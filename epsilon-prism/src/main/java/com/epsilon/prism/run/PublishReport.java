package com.epsilon.prism.run;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;

/**
 * 公開の段の報告。
 *
 * @param version 公開したバージョン
 * @param average 方策の記憶の記録（記憶を使わなければ無い）
 */
public record PublishReport(
    int version, @Key(doc = "方策の記憶の記録", presence = Presence.ABSENT) Snapshot.AverageRecord average) {}
