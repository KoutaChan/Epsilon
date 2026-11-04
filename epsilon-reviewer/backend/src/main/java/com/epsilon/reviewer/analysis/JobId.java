package com.epsilon.reviewer.analysis;

import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import java.util.UUID;
import java.util.regex.Pattern;

/** 解析の仕事の id。小文字の UUID。 */
public record JobId(String value) {
  private static final Pattern FORM =
      Pattern.compile("[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}");

  /** @throws ReviewerException job_not_found（形が違う） */
  public static JobId parse(String text) {
    if (text == null || !FORM.matcher(text).matches())
      throw new ReviewerException(ErrorCode.JOB_NOT_FOUND);
    return new JobId(text);
  }

  static JobId random() {
    return new JobId(UUID.randomUUID().toString());
  }
}
