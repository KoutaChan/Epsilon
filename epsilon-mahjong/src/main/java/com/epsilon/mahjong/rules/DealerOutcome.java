package com.epsilon.mahjong.rules;

public enum DealerOutcome {
  PASSES,
  CONTINUES,
  /** 途中流局で親が続く。和了止め・聴牌止めにはならない。 */
  ABORTIVE
}
