package com.epsilon.agent.settings;

import java.math.BigDecimal;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * 設定の木のノード。TOML と JSON は、いったんこの木に直してから重ね、record に結び付ける。
 *
 * <p>どのノードも、それが書かれた場所（{@code run.toml:12}、{@code preset production:3}、{@code --set}、{@code default}）を持つ。
 * 等しいかどうかは値だけで決め、書かれた場所は比べない。数はリテラルのまま持ち、値で比べる（1 と 1.0 は等しい）。
 */
public sealed interface Node permits Node.Table, Node.Array, Node.Text, Node.Numeral, Node.Bool {

  /** 値が書かれた場所。 */
  String origin();

  /** 全てのノードの書かれた場所を、引数の場所に替えた木。 */
  Node relabel(String origin);

  /** キーからノードへのテーブル。キーは書かれた順に並ぶ。 */
  record Table(Map<String, Node> entries, String origin) implements Node {

    public Table {
      entries = Collections.unmodifiableMap(new LinkedHashMap<>(entries));
      Objects.requireNonNull(origin, "origin");
    }

    /** @return キーのノード。無ければ null */
    public Node get(String key) {
      return entries.get(key);
    }

    /** キーを 1 つ除いたテーブル。 */
    public Table without(String key) {
      if (!entries.containsKey(key)) {
        return this;
      }
      Map<String, Node> kept = new LinkedHashMap<>(entries);
      kept.remove(key);
      return new Table(kept, origin);
    }

    @Override
    public Table relabel(String origin) {
      Map<String, Node> relabelled = new LinkedHashMap<>();
      entries.forEach((key, node) -> relabelled.put(key, node.relabel(origin)));
      return new Table(relabelled, origin);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Table table && entries.equals(table.entries);
    }

    @Override
    public int hashCode() {
      return entries.hashCode();
    }
  }

  /** 配列。 */
  record Array(List<Node> items, String origin) implements Node {

    public Array {
      items = List.copyOf(items);
      Objects.requireNonNull(origin, "origin");
    }

    @Override
    public Array relabel(String origin) {
      return new Array(items.stream().map(item -> item.relabel(origin)).toList(), origin);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Array array && items.equals(array.items);
    }

    @Override
    public int hashCode() {
      return items.hashCode();
    }
  }

  /** 文字列。 */
  record Text(String value, String origin) implements Node {

    public Text {
      Objects.requireNonNull(value, "value");
      Objects.requireNonNull(origin, "origin");
    }

    @Override
    public Text relabel(String origin) {
      return new Text(value, origin);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Text text && value.equals(text.value);
    }

    @Override
    public int hashCode() {
      return value.hashCode();
    }
  }

  /**
   * 数。
   *
   * @param literal 書いた形から正の符号 + と区切りの _ を除いた形。JSON の数としても読める
   * @param written 書いたままの形（{@code +1_000}）。TOML の数として読める
   */
  record Numeral(String literal, String written, String origin) implements Node {

    public Numeral {
      Objects.requireNonNull(written, "written");
      Objects.requireNonNull(origin, "origin");
      try {
        new BigDecimal(literal);
      } catch (NumberFormatException invalid) {
        throw new IllegalArgumentException("Not a number: " + literal, invalid);
      }
    }

    /** 符号 + も区切りの _ も無い形で書いた数。 */
    public Numeral(String literal, String origin) {
      this(literal, literal, origin);
    }

    /** 整数のリテラルか（小数点も指数も無い）。 */
    public boolean integer() {
      return literal.indexOf('.') < 0 && literal.indexOf('e') < 0 && literal.indexOf('E') < 0;
    }

    public BigDecimal decimal() {
      return new BigDecimal(literal);
    }

    @Override
    public Numeral relabel(String origin) {
      return new Numeral(literal, written, origin);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Numeral numeral && decimal().compareTo(numeral.decimal()) == 0;
    }

    @Override
    public int hashCode() {
      return decimal().stripTrailingZeros().hashCode();
    }
  }

  /** 真偽値。 */
  record Bool(boolean value, String origin) implements Node {

    public Bool {
      Objects.requireNonNull(origin, "origin");
    }

    @Override
    public Bool relabel(String origin) {
      return new Bool(value, origin);
    }

    @Override
    public boolean equals(Object other) {
      return other instanceof Bool bool && value == bool.value;
    }

    @Override
    public int hashCode() {
      return Boolean.hashCode(value);
    }
  }
}
