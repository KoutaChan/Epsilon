package com.epsilon.agent.settings;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 数の範囲。数のコンポーネントの型か、配列・テーブルの要素の型に付ける（{@code Map<DecisionClass, @Range(min = 0, max = 1) Double>}）。
 * 書かない側の端は制限しない。
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE_USE)
public @interface Range {

  double min() default Double.NEGATIVE_INFINITY;

  boolean minInclusive() default true;

  double max() default Double.POSITIVE_INFINITY;

  boolean maxInclusive() default true;
}
