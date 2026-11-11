package com.epsilon.prism.settings;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.mahjong.rules.Variant;
import java.util.Objects;

/**
 * run のセクション。どちらのキーも、run の途中では変えられない。
 *
 * @param variant ルール（四人打ちか三人打ちか）。そのルールの PRISM のモジュールが jar に無ければ、設定を読み込めない
 * @param seed 牌山、抽選、ネットワークの初期値に使う乱数の種
 */
public record RunSettings(
    @Key(doc = "ルール（四人打ちか三人打ちか）", presence = Presence.REQUIRED) Variant variant,
    @Key(doc = "牌山、抽選、ネットワークの初期値に使う乱数の種", presence = Presence.REQUIRED) long seed) {

  public RunSettings {
    Objects.requireNonNull(variant, "variant");
  }
}
