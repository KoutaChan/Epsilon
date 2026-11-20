package com.epsilon.agent.runtime;

import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.Settings;
import java.util.List;
import java.util.Objects;

/** 後続のイベントで値が変わらないモデルの、推論用 GPU ページの設定。 */
public record EventCacheSettings(
    @Key(doc = "因果的なイベントの履歴を再利用するか（CAUSAL は対応する形の GPU 推論だけで使う）", value = "CAUSAL") Mode mode,
    @Key(doc = "レーンに保存するイベント数。0 なら GPU の空き容量をレーンで分ける", value = "0") @Range(min = 0)
        int eventsPerLane,
    @Key(doc = "自動で容量を決めるとき、ほかの処理に残す総 VRAM の割合", value = "0.0625") @Range(min = 0, max = 1)
        double reserveFraction) {
  public enum Mode {
    CAUSAL,
    OFF
  }

  public static final EventCacheSettings DEFAULT =
      Settings.section(EventCacheSettings.class, "eventCache", List.of()).value();

  public EventCacheSettings {
    Objects.requireNonNull(mode, "mode");
    if (eventsPerLane < 0)
      throw new InvalidSetting("eventsPerLane", "Event capacity must be nonnegative");
    if (!(reserveFraction >= 0 && reserveFraction <= 1)) {
      throw new InvalidSetting(
          "reserveFraction", "Reserved VRAM fraction must be between zero and one");
    }
  }
}
