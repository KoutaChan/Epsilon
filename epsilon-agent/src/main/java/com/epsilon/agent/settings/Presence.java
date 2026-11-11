package com.epsilon.agent.settings;

/** 設定のキーが無いときの扱い。 */
public enum Presence {
  /** 無ければ既定値を入れる。既定値の無い入れ子のセクション・テーブル・配列は空から作る（中のキーはそれぞれの既定値になる）。 */
  DEFAULT,
  /** 無ければエラー。 */
  REQUIRED,
  /** 無ければ null。 */
  ABSENT
}
