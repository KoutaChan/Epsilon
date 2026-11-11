package com.epsilon.prism.settings;

/**
 * 追加の診断の出力を指定する JVM のシステムプロパティ。設定の診断のセクションが起動のときに書き（{@link DiagnosticsSettings#apply}）、
 * 使う側がネットワークを開くときや学習を始めるときに読む。
 */
public enum DiagnosticProperty {
  /** 学習の計測（{@code sync:K:W}: K ステップに 1 ステップを同期し、ウォームアップは W ステップ）。 */
  FIT_TIMING("prism.fitProfile");

  private final String property;

  DiagnosticProperty(String property) {
    this.property = property;
  }

  /** システムプロパティの名前。 */
  public String property() {
    return property;
  }

  /** @return 今の値。無ければ null */
  public String value() {
    return System.getProperty(property);
  }

  /** @param value 書く値。null なら消す */
  public void set(String value) {
    if (value == null) {
      System.clearProperty(property);
    } else {
      System.setProperty(property, value);
    }
  }
}
