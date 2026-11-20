package com.epsilon.prism.input;

import java.util.Objects;
import java.util.StringJoiner;
import java.util.function.ToIntFunction;

/** enum の宣言順に上位から桁を並べる混合基数のコード。 */
public final class CodeSpace<D extends Enum<D>> {
  private final int[] weights;
  private final int[] radices;
  private final D[] order;
  private final int size;
  private final String description;

  public CodeSpace(Class<D> type, ToIntFunction<D> radix) {
    this(type, type.getEnumConstants(), radix);
  }

  CodeSpace(Class<D> type, D[] digits, ToIntFunction<D> radix) {
    order = digits.clone();
    if (order.length == 0) throw new IllegalArgumentException("Empty code space");
    weights = new int[type.getEnumConstants().length];
    radices = new int[weights.length];
    int product = 1;
    for (int i = digits.length - 1; i >= 0; i--) {
      int base = radix.applyAsInt(digits[i]);
      if (base < 1)
        throw new IllegalArgumentException("Invalid radix for " + digits[i] + ": " + base);
      int field = digits[i].ordinal();
      if (weights[field] != 0) throw new IllegalArgumentException("Repeated digit: " + digits[i]);
      weights[field] = product;
      radices[field] = base;
      product = Math.multiplyExact(product, base);
    }
    size = product;
    var text = new StringJoiner(",");
    for (D digit : digits) text.add(digit.name() + "=" + radix(digit));
    description = text.toString();
  }

  public int size() {
    return size;
  }

  public int weight(D digit) {
    int weight = weights[digit.ordinal()];
    if (weight == 0) throw new IllegalArgumentException("Digit outside this code space: " + digit);
    return weight;
  }

  public int radix(D digit) {
    return radices[digit.ordinal()];
  }

  /** 宣言順に渡した桁の値をコードにする。 */
  public int encode(int... values) {
    if (values.length != order.length)
      throw new IllegalArgumentException("Wrong number of digits: " + values.length);
    int code = 0;
    for (int i = 0; i < order.length; i++) code += part(order[i], values[i]);
    return code;
  }

  /** その桁の値がコードに足す分を返す。 */
  public int part(D digit, int value) {
    return Objects.checkIndex(value, radix(digit)) * weight(digit);
  }

  /** コードからその桁の値を取り出す。 */
  public int digit(int code, D digit) {
    return Objects.checkIndex(code, size) / weight(digit) % radix(digit);
  }

  /** 桁の名前と基数を上位から並べた記述。 */
  public String describe() {
    return description;
  }
}
