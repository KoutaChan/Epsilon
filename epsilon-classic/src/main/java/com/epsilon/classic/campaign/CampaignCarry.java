package com.epsilon.classic.campaign;

/**
 * 反復の報告の持ち越し。
 *
 * @param actorRate 次の macro の方策の学習率
 */
public record CampaignCarry(double actorRate) {

  public CampaignCarry {
    if (!(actorRate > 0) || Double.isInfinite(actorRate)) {
      throw new IllegalArgumentException("The actor learning rate must be finite and positive: " + actorRate);
    }
  }
}
