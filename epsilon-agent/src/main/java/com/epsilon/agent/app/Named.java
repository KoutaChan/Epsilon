package com.epsilon.agent.app;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/** コマンドの名前と、使い方の表示に出す 1 文。コマンドの record に付ける。 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Named {

  /** 小文字とハイフンから成る語を 1 つか 2 つ、空白でつないだ名前（{@code train} や {@code settings change}）。 */
  String value();

  String doc();

  Reads reads() default Reads.LAYERS;
}
