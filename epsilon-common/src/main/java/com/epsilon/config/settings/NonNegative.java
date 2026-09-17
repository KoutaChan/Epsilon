package com.epsilon.config.settings;

import static java.lang.annotation.ElementType.RECORD_COMPONENT;
import static java.lang.annotation.RetentionPolicy.RUNTIME;

import java.lang.annotation.Retention;
import java.lang.annotation.Target;

/** 数値設定が有限かつ 0 以上であることを要求します。 */
@Retention(RUNTIME)
@Target(RECORD_COMPONENT)
public @interface NonNegative {}
