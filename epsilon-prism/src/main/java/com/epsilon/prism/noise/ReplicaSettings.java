package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Case;
import com.epsilon.agent.settings.Choice;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import java.util.List;

/** 複製の行のセクション。主経路のラベル行を決まった割合で選び、その行の全ての候補を、並べ直した牌山でもう 1 度打つ。 */
@Choice(key = "enabled", doc = "複製の行を作るか")
public sealed interface ReplicaSettings {

  /** 複製を作らない。 */
  @Case("false")
  record Off() implements ReplicaSettings {}

  /**
   * 複製を作る。
   *
   * @param schedule 反復ごとに選ぶ割合の段（始まる反復の昇順に並べ、最初の段は反復 1 から始める）
   * @param newCoins 新しい乱数で打つ複製の割合
   * @param handSeed 1 つの局の中で、並べ直しの乱数の種を共有する複製の割合
   * @param identity 牌山を並べ直さず、元の牌山のまま打ち直す複製の割合
   */
  @Case("true")
  record On(
      @Key(doc = "反復ごとに選ぶ割合の段（反復 1 から、始まる反復の昇順）", value = "[{from = 1, value = 0.02}, {from = 9, value = 0.005}]")
          List<Step> schedule,
      @Key(doc = "新しい乱数で打つ複製の割合", value = "0.5") @Range(min = 0, max = 1) double newCoins,
      @Key(doc = "1 つの局の中で、並べ直しの乱数の種を共有する複製の割合", value = "0.5") @Range(min = 0, max = 1) double handSeed,
      @Key(doc = "牌山を並べ直さず、元の牌山のまま打ち直す複製の割合", value = "0.002") @Range(min = 0, minInclusive = false, max = 1)
          double identity)
      implements ReplicaSettings {

    public On {
      schedule = List.copyOf(schedule);
      if (schedule.isEmpty() || schedule.getFirst().from() != 1) {
        throw new InvalidSetting("schedule", "the schedule starts at iteration 1");
      }
      for (int index = 1; index < schedule.size(); index++) {
        if (schedule.get(index).from() <= schedule.get(index - 1).from()) {
          throw new InvalidSetting("schedule", "the schedule's iterations must increase: " + schedule);
        }
      }
    }
  }

  /**
   * 割合の段の 1 つ。次の段が始まるまで、この割合で選ぶ。
   *
   * @param from この段が始まる反復
   * @param value 選ぶ割合
   */
  record Step(
      @Key(doc = "この段が始まる反復", presence = Presence.REQUIRED) @Range(min = 1) int from,
      @Key(doc = "選ぶ割合", presence = Presence.REQUIRED) @Range(min = 0, minInclusive = false, max = 1) double value) {}
}
