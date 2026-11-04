package com.epsilon.reviewer.record;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.UUID;
import java.util.regex.Pattern;

/** 読み込んだ牌譜の id。小文字の UUID。 */
public record RecordId(String value) {
  private static final Pattern FORM =
      Pattern.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");

  /** @throws ReviewerException record_expired（形が違う） */
  public static RecordId parse(String text) {
    if (text == null || !FORM.matcher(text).matches())
      throw new ReviewerException(ErrorCode.RECORD_EXPIRED);
    return new RecordId(text);
  }

  public static RecordId random() {
    return new RecordId(UUID.randomUUID().toString());
  }
}
