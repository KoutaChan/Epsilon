package com.epsilon.util;

import java.util.Locale;

/** ログやメトリクス向けのロケール固定フォーマット。 */
public final class FormatUtils {

  private FormatUtils() {}

  /**
   * 小数点以下を表示せずに数値を整形する。
   *
   * @param value 整形する値
   * @return {@link Locale#ROOT} で丸めた文字列
   */
  public static String fixed0(double value) {
    return fixed(value, 0);
  }

  /**
   * 小数点以下5桁で数値を整形する。
   *
   * @param value 整形する値
   * @return {@link Locale#ROOT} の固定小数文字列
   */
  public static String fixed5(double value) {
    return fixed(value, 5);
  }

  private static String fixed(double value, int decimals) {
    return String.format(Locale.ROOT, "%." + decimals + "f", value);
  }

  /**
   * 整数を指定幅まで0埋めする。
   *
   * @param value 整形する整数
   * @param width 最小表示幅
   * @return {@link Locale#ROOT} の0埋め文字列
   */
  public static String zeroPad(int value, int width) {
    return String.format(Locale.ROOT, "%0" + width + "d", value);
  }
}
