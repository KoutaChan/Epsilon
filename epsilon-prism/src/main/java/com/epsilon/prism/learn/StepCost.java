package com.epsilon.prism.learn;

/**
 * 学習の微小バッチ 1 つのデバイスの時間の見積もり（ns）。デバイスへの割り当てと、最後の塊の分け方の釣り合いに使う。
 *
 * <p>cost = A·行 + B·行·S + C·行·S² + D（S は微小バッチで最も長い行のトークンの数。行·S は詰めたトークン、行·S² は詰めた組）。頭（候補・値・
 * 補助の頭・損失と頭の逆伝播）は行に、背骨（埋め込み・正規化・射影・MLP とその逆伝播）は詰めたトークンに、注意とその逆伝播は詰めた組に
 * 比例するとみなし、送り・索引・関係の符号・返しは微小バッチごとにかかる。係数は {@link FitProfile} の同期する歩の段の時間を、段の組ごとに
 * レーンの比例の量と微小バッチの数へ最小二乗で当てた値（頭と背骨の組の微小バッチごとの分は D に入れた）。
 */
final class StepCost {

  /** 行あたりの ns（頭の順と逆）。 */
  static final double NANOS_PER_ROW = 9_400;

  /** 詰めたトークンあたりの ns（背骨の順と逆）。 */
  static final double NANOS_PER_TOKEN = 255;

  /** 詰めた組あたりの ns（注意の順と逆）。 */
  static final double NANOS_PER_PAIR = 0.37;

  /** 微小バッチあたりの ns（送り・索引・関係の符号・返しと、頭と背骨の固定の分）。 */
  static final double NANOS_PER_MICROBATCH = 11_000_000;

  private StepCost() {}

  /**
   * 微小バッチの見積もりを返す。
   *
   * @param rows 行の数（1 以上）
   * @param width 最も長い行のトークンの数 S
   * @return ns（切り捨て。行と S について単調に増える整数なので、和と差の比べが丸めに依らない）
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
