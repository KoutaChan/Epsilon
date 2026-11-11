package com.epsilon.classic.model;

import com.epsilon.classic.Architecture;

/**
 * Nano と Major で違う、ネットワークの幅と段の数。どの値も、変換した重みの形と合っていなければならない。Attention の幅はヘッドの数
 * （{@link #HEADS}）で割り切れること。Nano と Major で同じ幅は定数にする。
 */
enum Widths {
  /** 牌の関係の Attention 64・FFN 128、戦略の文脈 2 段・FFN 512、集約の Attention 64、方策の文脈 64。 */
  NANO(Architecture.NANO, 64, 128, 2, 512, 64, 64, 0),
  /** 牌の関係の Attention 128・FFN 256、戦略の文脈 3 段・FFN 960、集約の Attention 128、方策の文脈 128、公開の履歴の FFN 256。 */
  MAJOR(Architecture.MAJOR, 128, 256, 3, 960, 128, 128, 256);

  /** すべての Attention で使うヘッドの数。 */
  static final int HEADS = 4;

  /** 席の記憶の Attention の幅。 */
  static final int MEMORY_ATTENTION = 64;

  /** 席の記憶の FFN の幅。 */
  static final int MEMORY_FEED_FORWARD = 128;

  /** 戦略の文脈で、自分の手のトークンを作る Attention の幅。 */
  static final int HAND_ATTENTION = 64;

  /** 戦略の文脈で、6 トークンが互いを読む Attention の幅。 */
  static final int STRATEGIC_ATTENTION = 128;

  /** 方策の集約の FFN の幅。 */
  static final int POLICY_READOUT_FEED_FORWARD = 384;

  /** 価値の集約の FFN の幅。 */
  static final int VALUE_READOUT_FEED_FORWARD = 256;

  /** 二択の分岐の隠れ層の幅。 */
  static final int GATE = 256;

  /** 整数のフィールド 1 つの埋め込みの幅。 */
  static final int CATEGORY = 8;

  private final Architecture architecture;
  private final int tileAttention;
  private final int tileFeedForward;
  private final int strategicBlocks;
  private final int strategicFeedForward;
  private final int readoutAttention;
  private final int context;
  private final int historyFeedForward;

  Widths(
      Architecture architecture,
      int tileAttention,
      int tileFeedForward,
      int strategicBlocks,
      int strategicFeedForward,
      int readoutAttention,
      int context,
      int historyFeedForward) {
    this.architecture = architecture;
    this.tileAttention = tileAttention;
    this.tileFeedForward = tileFeedForward;
    this.strategicBlocks = strategicBlocks;
    this.strategicFeedForward = strategicFeedForward;
    this.readoutAttention = readoutAttention;
    this.context = context;
    this.historyFeedForward = historyFeedForward;
  }

  static Widths of(Architecture architecture) {
    return architecture == Architecture.NANO ? NANO : MAJOR;
  }

  Architecture architecture() {
    return architecture;
  }

  /** 状態・合法手・遷移の表現の幅。 */
  int hidden() {
    return architecture.hidden();
  }

  /** 牌の関係の Attention の幅。 */
  int tileAttention() {
    return tileAttention;
  }

  /** 牌の関係の FFN の幅。 */
  int tileFeedForward() {
    return tileFeedForward;
  }

  /** 戦略の文脈の段の数。 */
  int strategicBlocks() {
    return strategicBlocks;
  }

  /** 戦略の文脈の FFN の幅。 */
  int strategicFeedForward() {
    return strategicFeedForward;
  }

  /** 方策と価値の集約の Attention の幅。 */
  int readoutAttention() {
    return readoutAttention;
  }

  /** 方策ヘッドが、牌・席の記憶・待ちから読む文脈の幅。Major では公開の履歴の Attention の幅も兼ねる。 */
  int context() {
    return context;
  }

  /** 公開の履歴の FFN の幅（Major だけ。Nano は 0）。 */
  int historyFeedForward() {
    return historyFeedForward;
  }
}
