package com.epsilon.prism.input;

import com.epsilon.mahjong.tile.Tiles;

/**
 * ルールベースの事前分布 p0 が使う特徴（打牌の後の最小の向聴と、受け入れの残り枚数）を、p0 が読む float の値に直す。
 *
 * <p>値は、Nano の遷移の特徴の正規化（{@code DecisionTransitionEncoder}。向聴は (s + 1) / 9、受け入れは枚数 / 136）と、元の尺度への
 * 戻し（9·x − 1、136·x）を、同じ式・同じ順に float で計算したもの。正規化と戻しで生じる float の丸めまで、Nano とビット単位で同じに
 * なる。p0 はこの値から決まる。向聴と受け入れの枚数は、エンジンと同じ手牌の形の解析（{@link ShapeAnalysis}）で求めた値。打牌でない手の
 * 特徴は {@code RulePrior} が読まないので使わない。
 */
public final class PriorFeatures {

  private static final float SHANTEN_SCALE = 9.0f;
  private static final float UKEIRE_SCALE = Tiles.KINDS * (float) Tiles.COPIES;

  private PriorFeatures() {}

  /**
   * 打牌の後の向聴を、p0 が読む値にして返す。
   *
   * @param minimumShanten 打牌の後の手の最小の向聴
   * @return p0 が読む値
   */
  public static float shanten(int minimumShanten) {
    float normalized = (minimumShanten + 1.0f) / SHANTEN_SCALE;
    return SHANTEN_SCALE * normalized - 1f;
  }

  /**
   * 打牌の後の受け入れの残り枚数を、p0 が読む値にして返す。
   *
   * @param liveUkeire 打牌の後の手の受け入れの残り枚数
   * @return p0 が読む値
   */
  public static float ukeire(int liveUkeire) {
    float normalized = liveUkeire / UKEIRE_SCALE;
    return UKEIRE_SCALE * normalized;
  }
}
