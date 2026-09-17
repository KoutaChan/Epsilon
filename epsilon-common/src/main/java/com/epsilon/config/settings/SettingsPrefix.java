package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.TYPE;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** 設定レコードまたは入れ子の設定を読み込む際の、設定キーの接頭辞を指定する。 */
@Retention(RUNTIME)
@Target(TYPE)
public @interface SettingsPrefix {
  /**
   * 設定レコードが読み取る設定キーの接頭辞を返す。
   *
   * @return 例: {@code epsilon.decision.train.selectedPg}
   */
  String value();
}
