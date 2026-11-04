package com.epsilon.reviewer.model;

import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.DigestInputStream;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/**
 * チェックポイントのフォルダーの SHA-256（小文字の 16 進）。普通のファイルを並べ替えた順に、'/' で区切る相対のパスの UTF-8、0x00、
 * 大きさの 8 バイトのビッグエンディアン、中身を足す。models.json の sha256 があるモデルを初めて開く時に 1 回だけ作る。
 */
final class CheckpointDigest {
  private CheckpointDigest() {}

  static String of(Path directory) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException(failure);
    }
    for (Path file : CheckpointFingerprint.files(directory)) {
      String name = CheckpointFingerprint.relative(directory, file);
      digest.update(name.getBytes(StandardCharsets.UTF_8));
      digest.update((byte) 0);
      digest.update(ByteBuffer.allocate(Long.BYTES).putLong(Files.size(file)).array());
      try (InputStream in = Files.newInputStream(file);
          var stream = new DigestInputStream(in, digest)) {
        stream.transferTo(OutputStream.nullOutputStream());
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
