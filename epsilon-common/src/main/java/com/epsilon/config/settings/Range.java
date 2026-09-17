package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.RECORD_COMPONENT;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** 数値設定が有限かつ指定範囲に収まることを要求します。 */
@Retention(RUNTIME)
@Target(RECORD_COMPONENT)
public @interface Range {
  /**
   * 許容する最小値を返します。
   *
   * @return 境界を指定しない場合は {@code -Double.MAX_VALUE}
   */
  double min() default -Double.MAX_VALUE;

  /**
   * 下限値を許容するかを返します。
   *
   * @return 閉区間なら {@code true}、開区間なら {@code false}
   */
  boolean minInclusive() default true;

  /**
   * 許容する最大値を返します。
   *
   * @return 境界を指定しない場合は {@code Double.MAX_VALUE}
   */
  double max() default Double.MAX_VALUE;

  /**
   * 上限値を許容するかを返します。
   *
   * @return 閉区間なら {@code true}、開区間なら {@code false}
   */
  boolean maxInclusive() default true;
}
