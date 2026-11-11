package com.epsilon.agent.settings;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** 種類を選ぶセクションの、種類の 1 つ。値は種類のキーの値（enabled のセクションなら true か false）。 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Case {

  String value();
}
