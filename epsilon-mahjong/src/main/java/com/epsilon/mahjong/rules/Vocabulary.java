package com.epsilon.mahjong.rules;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * そのルールにある enum の定数を、宣言の順に並べた列。定数の番号はこの列の中の位置で、ordinal と同じとは限らない（三人打ちでは
 * チーが抜ける）。{@link VariantScoped} でない enum では、全ての定数が入る。
 *
 * <p>文字列表現は {@code Arrays.toString(type.getEnumConstants())} と同じ形式で、全ての定数を持つ列ならそれと一致する。
 */
public final class Vocabulary<E extends Enum<E>> {

  private final Class<E> type;
  private final Variant variant;
  private final List<E> constants;

  /** 位置 → 定数。List を通さずに配列から引く。 */
  private final Object[] byIndex;

  /** ordinal → 列の中の位置。ルールに無い定数は −1。 */
  private final int[] indexes;

  private Vocabulary(Class<E> type, Variant variant, List<E> constants) {
    this.type = type;
    this.variant = variant;
    this.constants = List.copyOf(constants);
    byIndex = this.constants.toArray();
    indexes = new int[type.getEnumConstants().length];
    Arrays.fill(indexes, -1);
    for (int index = 0; index < this.constants.size(); index++) {
      indexes[this.constants.get(index).ordinal()] = index;
    }
  }

  public static <E extends Enum<E>> Vocabulary<E> of(Class<E> type, Variant variant) {
    List<E> constants = new ArrayList<>();
    for (E constant : type.getEnumConstants()) {
      if (!(constant instanceof VariantScoped scoped) || scoped.in(variant)) {
        constants.add(constant);
      }
    }
    return new Vocabulary<>(type, variant, constants);
  }

  public Class<E> type() {
    return type;
  }

  public Variant variant() {
    return variant;
  }

  public int size() {
    return constants.size();
  }

  @SuppressWarnings("unchecked")
  public E get(int index) {
    return (E) byIndex[index];
  }

  /** 列の中での定数の位置。そのルールに無い定数なら IllegalArgumentException。 */
  public int index(E constant) {
    int index = indexes[constant.ordinal()];
    if (index < 0) {
      throw new IllegalArgumentException(
          constant + " is not in the " + variant + " vocabulary of " + type.getSimpleName());
    }
    return index;
  }

  public boolean contains(E constant) {
    return indexes[constant.ordinal()] >= 0;
  }

  /** 定数の列（変更できない）。 */
  public List<E> constants() {
    return constants;
  }

  @Override
  public String toString() {
    return constants.toString();
  }
}
