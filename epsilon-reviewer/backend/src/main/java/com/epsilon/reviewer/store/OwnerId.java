package com.epsilon.reviewer.store;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.regex.Pattern;

/** 結果のオーナー。Web では Cookie のキーの SHA-256、デスクトップでは {@link #LOCAL}。値は小文字の 16 進で 64 文字。 */
public record OwnerId(String value) {
  private static final Pattern FORM = Pattern.compile("[0-9a-f]{64}");

  /** デスクトップのオーナー（0 を 64 個並べた値）。 */
  public static final OwnerId LOCAL = new OwnerId("0".repeat(64));

  public OwnerId {
    if (!isValid(value)) throw new IllegalArgumentException("Invalid owner ID.");
  }

  /** Cookie のキーを US-ASCII でバイト列にして SHA-256 を取り、小文字の 16 進で書いた値をオーナーにする。 */
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
