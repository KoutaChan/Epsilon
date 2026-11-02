package com.epsilon.prism.grp;

import com.epsilon.prism.rows.CohortWriter;
import com.epsilon.prism.rows.RowCodec;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.HexFormat;
import java.util.List;

/**
 * 封印したコホートの行の指紋: 行のフレームのバイトごとの sha256 を並べ替えてまとめた sha256。シャードへの振り分けと書いた順（書き手の
 * スレッドの順）に依らず、行の中身が 1 ビットでも違えば変わる。
 */
final class RowHash {
  private RowHash() {}

  /**
   * コホートのディレクトリの行の指紋と行の数。
   *
   * @param rows 行の数
   * @param sha256 指紋
   */
  record Digest(long rows, String sha256) {}

  static Digest of(Path cohort) throws IOException {
    CohortWriter.Manifest manifest = CohortWriter.readManifest(cohort);
    RowCodec codec = new RowCodec(manifest.ledgerFeatures());
    List<String> frames = new ArrayList<>();
    for (CohortWriter.Shard shard : manifest.shards()) {
      ByteBuffer bytes = ByteBuffer.wrap(Files.readAllBytes(cohort.resolve(shard.file())));
      while (bytes.hasRemaining()) {
        int start = bytes.position();
        codec.decode(bytes);
        frames.add(HexFormat.of().formatHex(sha256(bytes.array(), start, bytes.position() - start)));
      }
    }
    if (frames.size() != manifest.rows())
      throw new IOException("Manifest says " + manifest.rows() + " rows, the shards hold " + frames.size());
    frames.sort(null);
    byte[] joined = String.join("\n", frames).getBytes(StandardCharsets.US_ASCII);
    return new Digest(frames.size(), HexFormat.of().formatHex(sha256(joined, 0, joined.length)));
  }

  private static byte[] sha256(byte[] bytes, int offset, int length) {
    try {
      MessageDigest digest = MessageDigest.getInstance("SHA-256");
      digest.update(bytes, offset, length);
      return digest.digest();
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }
}
