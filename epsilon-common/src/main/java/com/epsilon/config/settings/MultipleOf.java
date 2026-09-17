package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.RECORD_COMPONENT;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** 整数設定が指定値の倍数であることを要求します。 */
@Retention(RUNTIME)
@Target(RECORD_COMPONENT)
public @interface MultipleOf {
  /**
   * 正の除数を返します。
   *
   * @return 設定値を割り切る必要がある正の整数
   */
  long value();
}
