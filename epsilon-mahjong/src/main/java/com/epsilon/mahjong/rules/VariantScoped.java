package com.epsilon.mahjong.rules;

/** ルールごとに有無が分かれる enum の定数。{@link Vocabulary} は、そのルールにある定数だけを宣言の順に並べる。 */
public interface VariantScoped {

  /** この定数があるのは、ルールがこの項目を持つときだけ。null なら、どのルールにもある。 */
  RuleFlag requires();

  /** このルールにこの定数があるなら true。どのルールにもある定数か、ルールが必要な項目を持つときに true。 */
  default boolean in(Variant variant) {
    RuleFlag flag = requires();
    return flag == null || variant.has(flag);
  }
}
