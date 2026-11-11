package com.epsilon.agent.app;

import com.epsilon.agent.settings.InvalidSetting;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 両端を含む番号の範囲（コホートやバージョン）。引数では {@code 400..403} と書き、1 つだけなら {@code 400..400} と書く。
 * コマンドの record は文字列のコンポーネントで受け、コンパクトコンストラクタで {@link #of} を呼んで書き方を確かめる。
 *
 * @param first 最初の番号（1 以上）
 * @param last 最後の番号（first 以上）
 */
public record Span(int first, int last) {

  private static final Pattern FORM = Pattern.compile("([0-9]+)\\.\\.([0-9]+)");

  public Span {
    if (first < 1 || last < first) {
      throw new IllegalArgumentException("Invalid range " + first + ".." + last);
    }
  }

  /**
   * 範囲を読む。
   *
   * @param component 範囲を書いたコンポーネントの名前。エラーメッセージに使う
   * @param text {@code a..b} の形の文字列
   * @throws InvalidSetting 書き方が違うか、{@code 1 <= a <= b} でない
   */
  public static Span of(String component, String text) {
    Matcher matcher = FORM.matcher(text);
    if (matcher.matches()) {
      try {
        int first = Integer.parseInt(matcher.group(1));
        int last = Integer.parseInt(matcher.group(2));
        if (first >= 1 && last >= first) {
          return new Span(first, last);
        }
      } catch (NumberFormatException overflow) {
        // 桁あふれも、下で書き方の誤りとして扱う
      }
    }
    throw new InvalidSetting(component, "expected a range a..b with 1 <= a <= b, not " + text);
  }

  /** 範囲の番号の数。 */
  public int size() {
    return last - first + 1;
  }

  @Override
  public String toString() {
    return first + ".." + last;
  }
}
