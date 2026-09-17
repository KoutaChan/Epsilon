package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.RECORD_COMPONENT;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** 設定ファイルとシステムプロパティのどちらにも値がない場合に使う既定値。 */
@Retention(RUNTIME)
@Target(RECORD_COMPONENT)
public @interface Default {
  /**
   * 既定値を返す。真偽値・数値・列挙型も文字列として指定する。
   *
   * @return 構成要素型へ変換する前の設定文字列
   */
  String value();
}
