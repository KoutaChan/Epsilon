package com.epsilon.prism.noise;

import com.epsilon.agent.settings.Key;
import com.epsilon.mahjong.engine.DrawOrder;
import java.util.Objects;

/**
 * 共通乱数のセクション。分岐の続きの分布は変えず、兄弟の分岐どうしの相関だけを変える。
 *
 * @param wall 分岐の牌山の作り方
 * @param seatStreams 席ごとのツモ列で打つか
 * @param actions 判断で手を引く乱数を、親と分岐でどう共有するか
 */
public record CrnSettings(
    @Key(doc = "分岐の牌山の作り方", value = "SHARED") WallSharing wall,
    @Key(doc = "席ごとのツモ列で打つか（false なら牌山の順にツモる）", value = "true") boolean seatStreams,
    @Key(doc = "判断で手を引く乱数を、親と分岐でどう共有するか", value = "TWO_STAGE_FLOOR") ActionSharing actions) {

  public CrnSettings {
    Objects.requireNonNull(wall, "wall");
    Objects.requireNonNull(actions, "actions");
  }

  /** 収集で打つ局で、ツモをどの順に引くかを返す。 */
  public DrawOrder draws() {
    return seatStreams ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER;
  }
}
