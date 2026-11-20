package com.epsilon.agent.settings;

/** 設定のキーが無いときの扱い。 */
public enum Presence {
  /** 無ければ既定値を入れる。既定値の無い入れ子のセクション・テーブル・配列は空から作る（中のキーはそれぞれの既定値になる）。 */
  DEFAULT(null),
  /** 無ければエラー。 */
  REQUIRED("必須"),
  /** 無ければ null。 */
  ABSENT("省略できる");

  private final String label;

  Presence(String label) {
    this.label = label;
  }

  String label() {
    return label;
  }
}
