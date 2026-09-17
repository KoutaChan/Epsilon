package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.RECORD_COMPONENT;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** レコードの各構成要素に対応する設定キーの末尾名を指定する。 */
@Retention(RUNTIME)
@Target(RECORD_COMPONENT)
public @interface Setting {
  /**
   * 設定キーの接頭辞に続く相対名を返す。
   *
   * @return レコード構成要素に対応する相対設定キー
   */
  String value();
}
