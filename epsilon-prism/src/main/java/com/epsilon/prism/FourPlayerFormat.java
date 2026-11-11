package com.epsilon.prism;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;

/**
 * 四人打ちにしか対応していない形式（入力、補助のヘッドの教師）が、フィールドの範囲と語彙を決めるのに使うルール。これらの形式の
 * 配置とフィンガープリントは、このルールで作る。ルールに依らない enum の語彙は全ての定数を持つので、フィンガープリントの記述は enum の
 * 定数を並べたものと同じになる。
 */
public final class FourPlayerFormat {

  public static final Variant VARIANT = Variant.TENHOU_4P;

  /** 四人打ちの席の数。 */
  public static final int SEATS = VARIANT.seats();

  private FourPlayerFormat() {}

  /** 四人打ちでの enum の語彙を返す。 */
  @SuppressWarnings({"unchecked", "rawtypes"})
  public static Vocabulary<?> values(Class<? extends Enum<?>> type) {
    return Vocabulary.of((Class) type, VARIANT);
  }
}
