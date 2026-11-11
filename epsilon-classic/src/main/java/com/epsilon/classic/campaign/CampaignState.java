package com.epsilon.classic.campaign;

import com.epsilon.agent.run.Progress;

/**
 * classic の campaign の進み（state.json）。反復を閉じたときに書く。
 *
 * @param next 次に回す macro（macro next − 1 までを閉じ、バージョン next まで公開した）
 * @param games これまでに集めた半荘の数
 * @param decisions これまでに集めた学習器の席の判断の数
 */
public record CampaignState(int next, long games, long decisions) implements Progress {

  public CampaignState {
    if (next < 1 || games < 0 || decisions < 0) {
      throw new IllegalArgumentException(
          "Invalid campaign state: next " + next + ", games " + games + ", decisions " + decisions);
    }
  }

  /** 始めの state（次は macro 1、バージョン 1 を公開した）。 */
  public static CampaignState initial() {
    return new CampaignState(1, 0, 0);
  }
}
