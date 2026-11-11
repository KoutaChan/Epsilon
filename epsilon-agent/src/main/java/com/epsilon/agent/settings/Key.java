package com.epsilon.agent.settings;

import java.lang.annotation.ElementType;
import java.lang.annotation.Retention;
import java.lang.annotation.RetentionPolicy;
import java.lang.annotation.Target;

/**
 * 設定のキー。record のコンポーネントに付け、説明・既定値・無いときの扱い・変えてよい時期・秘密かどうかを、ここだけに書く。
 *
 * <p>プログラムが書くファイルの record では、フィールドは全て必要。無くてよいのは、この注釈で既定値を書いたコンポーネント（既定値を
 * 初期値として入れる）と、無ければ null としたコンポーネントだけ（{@link RecordFile}）。
 */
@Retention(RetentionPolicy.RUNTIME)
@Target(ElementType.RECORD_COMPONENT)
public @interface Key {

  /** 説明とひな形に出す 1 文。 */
  String doc();

  /** 既定値。コンポーネントの型で読む（enum と文字列は引用符なし、配列とテーブルは TOML の 1 行の書き方）。空なら既定値は無い。 */
  String value() default "";

  Presence presence() default Presence.DEFAULT;

  Change change() default Change.INHERIT;

  /** 書き出し・ログ・記録では伏せ、設定ファイルからだけ受け付ける。 */
  boolean secret() default false;
}
