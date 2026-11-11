package com.epsilon.prism.core;

import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import java.util.Collections;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.Map;
import java.util.Set;

/** 判断のクラスをキーにした表と集合を扱う。設定にあるクラスごとの表は、そのルールの判断のクラスをちょうど 1 回ずつ持つ。 */
public final class ClassTables {

  private static final Set<DecisionClass> WITH_NUKI = Collections.unmodifiableSet(EnumSet.allOf(DecisionClass.class));

  private static final Set<DecisionClass> WITHOUT_NUKI =
      Collections.unmodifiableSet(EnumSet.complementOf(EnumSet.of(DecisionClass.NUKI, DecisionClass.NUKI_RESPONSE)));

  private ClassTables() {}

  /** そのルールで使う判断のクラスを返す。北抜きと抜いた北への応答のクラスは、北抜きのある三人打ちだけが持つ。 */
  public static Set<DecisionClass> classes(Variant variant) {
    return variant.has(RuleFlag.NUKI) ? WITH_NUKI : WITHOUT_NUKI;
  }

  /** 変更できないコピーを返す。並びは判断のクラスの順。 */
  public static <V> Map<DecisionClass, V> copy(Map<DecisionClass, V> table) {
    Map<DecisionClass, V> copy = new EnumMap<>(DecisionClass.class);
    copy.putAll(table);
    return Collections.unmodifiableMap(copy);
  }

  /** 変更できないコピーを返す。並びは判断のクラスの順。 */
  public static Set<DecisionClass> copy(Set<DecisionClass> classes) {
    Set<DecisionClass> copy = EnumSet.noneOf(DecisionClass.class);
    copy.addAll(classes);
    return Collections.unmodifiableSet(copy);
  }

  /**
   * 表のキーが、そのルールで使う判断のクラスと一致しなければ、違いを英語の 1 文で返す。
   *
   * @return 違いを書いた文。同じなら null
   */
  public static String mismatch(Set<DecisionClass> keys, Variant variant) {
    Set<DecisionClass> expected = classes(variant);
    if (keys.equals(expected)) {
      return null;
    }
    Set<DecisionClass> missing = EnumSet.noneOf(DecisionClass.class);
    missing.addAll(expected);
    missing.removeAll(keys);
    Set<DecisionClass> extra = EnumSet.noneOf(DecisionClass.class);
    extra.addAll(keys);
    extra.removeAll(expected);
    return "the table must hold each class of " + variant + " once (missing " + missing + ", not of the variant "
        + extra + ")";
  }
}
