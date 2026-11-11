package com.epsilon.prism.learn;

/**
 * 学習のマイクロバッチ 1 つにかかるデバイスの時間の見積もり（ns）。マイクロバッチをデバイスに割り当てるときと、最後のバッチを
 * 釣り合うように分けるときに使う。
 *
 * <p>cost = A·行 + B·行·S + C·行·S² + D。S はマイクロバッチで最も長い行のトークン数で、行·S はパディングを含むトークン数、行·S² は
 * パディングを含むトークンの組の数になる。ヘッド（候補・値・補助のヘッド、損失とヘッドの逆伝播）は行数に比例し、バックボーン（埋め込み・
 * 正規化・射影・MLP とその逆伝播）はトークン数に比例し、Attention とその逆伝播は組の数に比例するとみなす。転送・インデックス・関係のコード・
 * 結果の返送はマイクロバッチごとに一定の時間がかかる。係数は、{@link FitProfile} で同期して測ったステップの段ごとの時間を、段の組ごとに
 * レーンで比例する量（行数・トークン数・組の数）とマイクロバッチ数へ最小二乗で当てはめた値である。ヘッドとバックボーンのマイクロバッチ
 * ごとの固定の分は D に含める。
 */
final class StepCost {

  /** 1 行あたりの ns（ヘッドの順伝播と逆伝播）。 */
  static final double NANOS_PER_ROW = 9_400;

  /** パディングを含むトークン 1 つあたりの ns（バックボーンの順伝播と逆伝播）。 */
  static final double NANOS_PER_TOKEN = 255;

  /** パディングを含むトークンの組 1 つあたりの ns（Attention の順伝播と逆伝播）。 */
  static final double NANOS_PER_PAIR = 0.37;

  /** マイクロバッチ 1 つあたりの ns（転送・インデックス・関係のコード・返送と、ヘッドとバックボーンの固定の分）。 */
  static final double NANOS_PER_MICROBATCH = 11_000_000;

  private StepCost() {}

  /**
   * マイクロバッチ 1 つの時間の見積もりを返す。
   *
   * @param rows 行数（1 以上）
   * @param width 最も長い行のトークン数 S
   * @return 見積もりの ns（小数は切り捨て）。行数と S について単調に増える整数なので、和や差を比べても丸めで結果が変わらない
   */
  static long nanos(int rows, int width) {
    double tokens = (double) rows * width;
    return (long)
        (NANOS_PER_ROW * rows
            + NANOS_PER_TOKEN * tokens
            + NANOS_PER_PAIR * tokens * width
            + NANOS_PER_MICROBATCH);
  }
}
