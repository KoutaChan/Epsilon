package com.epsilon.reviewer.store;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.regex.Pattern;

/** 結果の持ち主。Web は Cookie の鍵の SHA-256、デスクトップは {@link #LOCAL}。小文字の 16 進 64 字。 */
public record OwnerId(String value) {
  private static final Pattern FORM = Pattern.compile("[0-9a-f]{64}");

  /** デスクトップの持ち主（0 が 64 個）。 */
  public static final OwnerId LOCAL = new OwnerId("0".repeat(64));

  public OwnerId {
    if (!isValid(value)) throw new IllegalArgumentException("Invalid owner ID.");
  }

  /** Cookie の鍵の US-ASCII の SHA-256 の小文字の 16 進。 */
  public static OwnerId fromCookieKey(String key) {
    try {
      return new OwnerId(
          HexFormat.of()
              .formatHex(
                  MessageDigest.getInstance("SHA-256")
                      .digest(key.getBytes(StandardCharsets.US_ASCII))));
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException(failure);
    }
  }

  static boolean isValid(String value) {
    return value != null && FORM.matcher(value).matches();
  }
}
