package com.epsilon.agent.settings;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 種類を選ぶセクション。種類のキーの値で、許した record の 1 つを選ぶ。sealed の interface に付け、許すクラスは全て {@link Case} を
 * 付けた record にする。
 *
 * <p>選んだ種類に無いキーはエラー。コンポーネントの無い種類は、値だけで書ける（{@code stop = "FIXED"} は
 * {@code stop = {kind = "FIXED"}} と同じ）。上の層が種類のキーを違う値にしたら、下の層のそのセクションは捨てる。種類のキーを変えてよい
 * 時期は、このセクションを持つコンポーネントの注釈に従う。
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.TYPE)
public @interface Choice {

  /** 種類のキーの名前（kind・mode・scheme・source・enabled など）。 */
  String key();

  /** 種類のキーの説明（何を選ぶか）。ひな形と効いている設定の表示が、種類のキーの上に書く。設定でない記録の型では空のまま。 */
  String doc() default "";
}
