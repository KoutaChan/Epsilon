package com.epsilon.prism.hip;

import java.util.List;

/** バックボーンの 1 層の FFN の幅。幅は {@link #CHUNK} の倍数にする（K3・K5 は F をこの幅ずつ処理する）。 */
public sealed interface HipFfn permits HipFfn.Shared, HipFfn.Typed {

  /** K3・K5・K6 が F を処理する単位の幅。隠れ層 h のこの幅の列を LDS に置く。 */
  int CHUNK = 128;

  /**
   * トークンの組が通る FFN の幅を返す。
   *
   * @param group トークンの組
   * @return 幅
   * @throws IllegalArgumentException この FFN を通らない組のとき
   */
  int width(TokenGroup group);

  /**
   * FFN を通る組を返す。
   *
   * @return 組（トークンの軸の順）
   */
  List<TokenGroup> groups();

  /**
   * 全てのトークンが 1 つの FFN を通る。
   *
   * @param width 幅
   */
  record Shared(int width) implements HipFfn {

    /**
     * 幅を確かめる。
     *
     * @throws IllegalArgumentException 幅が {@link #CHUNK} の正の倍数でないとき
     */
    public Shared {
      requireWidth("ffn", width);
    }

    @Override
    public int width(TokenGroup group) {
      if (group != TokenGroup.ALL) {
        throw new IllegalArgumentException("a shared FFN has no " + group + " group");
      }
      return width;
    }

    @Override
    public List<TokenGroup> groups() {
      return List.of(TokenGroup.ALL);
    }
  }

  /**
   * 局・席と牌が別々の FFN を通る。
   *
   * @param context 局・席の FFN の幅
   * @param tile 牌の FFN の幅
   */
  record Typed(int context, int tile) implements HipFfn {

    /**
     * 幅を確かめる。
     *
     * @throws IllegalArgumentException 幅が {@link #CHUNK} の正の倍数でないとき
     */
    public Typed {
      requireWidth("context ffn", context);
      requireWidth("tile ffn", tile);
    }

    @Override
    public int width(TokenGroup group) {
      return switch (group) {
        case CONTEXT -> context;
        case TILE -> tile;
        case ALL -> throw new IllegalArgumentException("a typed FFN has no shared group");
      };
    }

    @Override
    public List<TokenGroup> groups() {
      return List.of(TokenGroup.CONTEXT, TokenGroup.TILE);
    }
  }

  private static void requireWidth(String name, int width) {
    if (width <= 0 || width % CHUNK != 0) {
      throw new IllegalArgumentException(name + " width must be a positive multiple of " + CHUNK + ", got " + width);
    }
  }
}
