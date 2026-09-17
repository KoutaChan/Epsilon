package com.epsilon.major.ai.model;

/** Epsilon の Decision モデルについて、候補ごとの計算幅と共有表現の幅を定義する。 */
public final class EpsilonDecisionArchitecture {
  public static final int HIDDEN_SIZE = 384;
  public static final int POLICY_CONTEXT_WIDTH = 64;
  public static final int GATE_WIDTH = 256;
  public static final int TILE_ATTENTION_WIDTH = 128;
  public static final int TILE_FEED_FORWARD_WIDTH = 256;
  public static final int STRATEGIC_BLOCK_COUNT = 3;
  public static final int STRATEGIC_FEED_FORWARD_WIDTH = 960;
  public static final int PUBLIC_HISTORY_FEED_FORWARD_WIDTH = 256;

  private EpsilonDecisionArchitecture() {}
}
