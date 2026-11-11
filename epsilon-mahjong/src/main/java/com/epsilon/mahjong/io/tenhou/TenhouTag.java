package com.epsilon.mahjong.io.tenhou;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** 天鳳のタグ 1 つ（開始タグか空要素）の名前と属性。牌譜の XML も、ライブのメッセージも、同じ形で読む。 */
public record TenhouTag(String name, Map<String, String> attributes) {

  /**
   * 開始タグか空要素のタグに一致する正規表現。名前と属性の並びを取り出す。属性の並びは後戻りしない繰り返し（*+）で読むので、
   * 属性がいくつあっても照合のスタックは深くならない。
   */
  private static final Pattern TAG =
      Pattern.compile("<([A-Za-z][A-Za-z0-9]*)((?:\\s+[A-Za-z][A-Za-z0-9]*\\s*=\\s*\"[^\"]*\")*+)\\s*/?>");

  private static final Pattern ATTRIBUTE = Pattern.compile("([A-Za-z][A-Za-z0-9]*)\\s*=\\s*\"([^\"]*)\"");

  /** 数の列の区切り。 */
  private static final String SEPARATOR = ",";

  public TenhouTag {
    Objects.requireNonNull(name, "name");
    attributes = Map.copyOf(attributes);
  }

  /**
   * XML の開始タグと空要素のタグを、現れた順に読む。終了タグと処理命令は読み飛ばす。形の崩れたタグがあれば
   * IllegalArgumentException を投げる。
   */
  public static List<TenhouTag> parse(String xml) {
    List<TenhouTag> tags = new ArrayList<>();
    Matcher tag = TAG.matcher(xml);
    int at = xml.indexOf('<');
    while (at >= 0) {
      char next = at + 1 < xml.length() ? xml.charAt(at + 1) : ' ';
      int end;
      if (next == '/' || next == '?') {
        end = xml.indexOf('>', at);
        if (end < 0) {
          throw new IllegalArgumentException("Unclosed tag at " + at);
        }
      } else {
        if (!tag.region(at, xml.length()).lookingAt()) {
          throw new IllegalArgumentException("Malformed Tenhou tag at " + at);
        }
        Map<String, String> attributes = new HashMap<>();
        Matcher attribute = ATTRIBUTE.matcher(tag.group(2));
        while (attribute.find()) {
          attributes.put(attribute.group(1), attribute.group(2));
        }
        tags.add(new TenhouTag(tag.group(1), attributes));
        end = tag.end() - 1;
      }
      at = xml.indexOf('<', end + 1);
    }
    return tags;
  }

  /** 属性の値を返す。無ければ null。 */
  public String get(String key) {
    return attributes.get(key);
  }

  /** 属性の値を返す。無ければ IllegalArgumentException を投げる。 */
  public String required(String key) {
    String value = attributes.get(key);
    if (value == null) {
      throw new IllegalArgumentException("Tenhou " + name + " has no " + key);
    }
    return value;
  }

  /** 属性の値を整数として読む。無いか、整数でなければ IllegalArgumentException を投げる。 */
  public int integer(String key) {
    return parseInteger(required(key));
  }

  /** 属性の値を、カンマで区切った整数の列として読む。空の文字列なら長さ 0 の配列を返す。 */
  public int[] integers(String key) {
    String list = required(key);
    if (list.isEmpty()) {
      return new int[0];
    }
    String[] values = list.split(SEPARATOR, -1);
    int[] integers = new int[values.length];
    for (int index = 0; index < values.length; index++) {
      integers[index] = parseInteger(values[index]);
    }
    return integers;
  }

  private int parseInteger(String text) {
    try {
      return Integer.parseInt(text.trim());
    } catch (NumberFormatException failure) {
      throw new IllegalArgumentException("Tenhou " + name + " has a value that is not an integer: " + text, failure);
    }
  }
}
