package com.epsilon.prism.input;

import com.epsilon.core.Tile;

/**
 * 規則事前 p0 の特徴（打牌の後の最小の向聴と、受け入れの生きた枚数）を、p0 が読む float の値にする。
 *
 * <p>値は Nano の遷移特徴の正規化（{@code DecisionTransitionEncoder}: 向聴は (s + 1) / 9、受け入れは枚数 / 136）を元の尺度に戻した値
 * （9·x − 1、136·x）と同じ式を同じ順に通したもの（正規化と戻しの float の丸めまでビットで同じ。p0 はこの値で定まる）。向聴と受け入れの枚数は、
 * エンジンの解析と同じ形の解析（{@link ShapeAnalysis}）の値。打牌でない手の特徴は使わない（{@code RulePrior} が読まない）。
 */
final class PriorFeatures {

  private static final float SHANTEN_SCALE = 9.0f;
  private static final float UKEIRE_SCALE = Tile.NUM_TILE_TYPES * (float) Tile.TILES_PER_TYPE;

  private PriorFeatures() {}

  /**
   * 打牌の後の向聴を返す。
   *
   * @param minimumShanten 後の手の最小の向聴
   * @return p0 が読む値
   */
  static float shanten(int minimumShanten) {
    float normalized = (minimumShanten + 1.0f) / SHANTEN_SCALE;
    return SHANTEN_SCALE * normalized - 1f;
  }

  /**
   * 打牌の後の受け入れの生きた枚数を返す。
   *
   * @param liveUkeire 後の手の受け入れの生きた枚数
   * @return p0 が読む値
   */
  static float ukeire(int liveUkeire) {
    float normalized = liveUkeire / UKEIRE_SCALE;
    return UKEIRE_SCALE * normalized;
  }
}
