package com.epsilon.reviewer.store;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.UUID;
import java.util.regex.Pattern;

/** 解析結果の id（共有リンクの id も同じ値）。小文字の UUID。 */
public record ResultId(String value) {
  private static final Pattern FORM =
      Pattern.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");

  public ResultId {
    if (!isValid(value)) throw new IllegalArgumentException("Invalid result ID.");
  }

  /** @param notFound 形が違う時の符号（持ち主の道は result_not_found、共有の道は share_not_found） */
  public static ResultId parse(String text, ErrorCode notFound) {
    if (!isValid(text)) throw new ReviewerException(notFound);
    return new ResultId(text);
  }

  public static ResultId random() {
    return new ResultId(UUID.randomUUID().toString());
  }

  static boolean isValid(String text) {
    return text != null && FORM.matcher(text).matches();
  }
}
