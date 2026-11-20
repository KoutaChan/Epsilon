package com.epsilon.prism.input;

import java.util.Objects;

/** トークンの順に並ぶ 1 列の書き込み先。 */
final class InputColumn {
  @FunctionalInterface
  interface Source {
    void write(RowFacts row, InputColumn column);
  }

  interface Field extends FieldSpec.BoundedField {
    Source source();

    default Source source(InputFormat format) {
      return source();
    }
  }

  private final Source source;
  private final byte[] draft;
  private final int first;
  private final int stride;
  private final FieldSpec spec;

  InputColumn(Source source, byte[] draft, int first, int stride, FieldSpec spec) {
    this.source = source;
    this.draft = draft;
    this.first = first;
    this.stride = stride;
    this.spec = spec;
  }

  void write(RowFacts row) {
    source.write(row, this);
  }

  void count(int token, int value) {
    draft[first + token * stride] = (byte) Objects.checkIndex(value, spec.vocabulary());
  }

  void flag(int token, boolean value) {
    draft[first + token * stride] = (byte) (value ? 1 : 0);
  }

  void id(int token, int value) {
    draft[first + token * stride] = (byte) Ids.index(value, spec.vocabulary() - 1);
  }

  void value(int token, Enum<?> value) {
    value(token, 0, value);
  }

  void value(int token, int repeat, Enum<?> value) {
    draft[first + token * stride + Objects.checkIndex(repeat, spec.repeats())] = encode(value);
  }

  void values(RowFacts row, int start, int repeat, Enum<?> value, long tiles) {
    byte encoded = encode(value);
    int offset = first + start * stride + Objects.checkIndex(repeat, spec.repeats());
    for (long rest = tiles; rest != 0; rest &= rest - 1) {
      draft[offset + row.token(Long.numberOfTrailingZeros(rest)) * stride] = encoded;
    }
  }

  private byte encode(Enum<?> value) {
    if (value != null && value.getDeclaringClass() != spec.values().type()) {
      throw new IllegalArgumentException("Unexpected value type: " + value.getDeclaringClass());
    }
    return (byte) Ids.of(spec.values(), value);
  }

  void count(int token, int repeat, int value) {
    draft[first + token * stride + Objects.checkIndex(repeat, spec.repeats())] =
        (byte) Objects.checkIndex(value, spec.vocabulary());
  }

  void flags(RowFacts row, long mask) {
    for (long rest = mask; rest != 0; rest &= rest - 1)
      flag(row.token(Long.numberOfTrailingZeros(rest)), true);
  }
}
