package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.prism.codec.SchemaDigest;
import java.nio.ByteBuffer;
import java.util.Objects;

/**
 * 標本の shard のバイト列の形。{@link LuckSampleWriter} が書き、{@link LuckSampleStore} が読む。値は全てリトルエンディアン。
 *
 * <pre>
 * shard  = frame*
 * frame  = [i32 圧縮後の長さ][i32 展開後の長さ][i32 CRC32C(展開後)][i32 標本の数][zstd level 1 の本体]
 * 展開後 = sample × 標本の数
 * sample = [i64 半荘 ID][u16 O° の long の数 n][i64 × n の O°][f32 × 4 の Y（行為者から見た相対席の順）][u16 層の鍵][f32 重み]
 * </pre>
 *
 * <p>枠は約 {@link #BLOCK} バイトの標本をまとめて圧縮する（1 件ずつでは縮まない）。学習は標本を順に読むので、枠の中を任意に引く索引は
 * 持たない。形を変えれば {@link #fingerprint()} が変わり、目録の指紋が合わない shard は読まない。
 */
final class LuckSampleFormat {

  /** zstd の段。 */
  static final int LEVEL = 1;

  /** 枠の頭のバイト数。 */
  static final int FRAME_HEADER = 4 * Integer.BYTES;

  /** 枠にまとめる展開後のバイト数の目安（展開後の長さの上限でもある）。 */
  static final int BLOCK = 1 << 20;

  /** 標本 1 件の固定部（半荘 ID・O° の長さ・Y・層・重み）のバイト数。 */
  static final int SAMPLE_FIXED =
      Long.BYTES + Short.BYTES + GameState.NUM_PLAYERS * Float.BYTES + Short.BYTES + Float.BYTES;

  /** 標本 1 件の最大のバイト数。 */
  static final int MAX_SAMPLE = SAMPLE_FIXED + LuckSchema.MAX_LONGS * Long.BYTES;

  private LuckSampleFormat() {}

  /**
   * 形の指紋を返す（O° の指紋・枠・標本の並び・層の定義を含む）。目録に書き、読むときに照らす。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  static String fingerprint() {
    return Fingerprint.VALUE;
  }

  /**
   * 標本を 1 件書く。
   *
   * @param out 書く先（{@link #MAX_SAMPLE} バイト以上の空き）
   * @param game 半荘 ID
   * @param words O° を置いた配列
   * @param base O° の先頭
   * @param y 局末の Φ（相対席の順に 4 席）
   * @param stratum 層の鍵
   * @param weight 重み
   * @throws IllegalArgumentException Y・層・重みが範囲の外のとき
   * @throws IndexOutOfBoundsException O° が配列に収まらないとき
   */
  static void put(
      ByteBuffer out, long game, long[] words, int base, float[] y, int stratum, float weight) {
    LuckSample.requireValues(y, stratum, weight);
    int longs = LuckSchema.longs(words, base);
    Objects.checkFromIndexSize(base, longs, words.length);
    out.putLong(game);
    out.putShort((short) longs);
    for (int index = 0; index < longs; index++) {
      out.putLong(words[base + index]);
    }
    for (float value : y) {
      out.putFloat(value);
    }
    out.putShort((short) stratum);
    out.putFloat(weight);
  }

  /** 遅延で作る（初期化の順の罠を避ける）。 */
  private static final class Fingerprint {

    private static final String VALUE =
        SchemaDigest.named("epsilon-prism-v2-luck-sample")
            .with("observation", LuckSchema.fingerprint())
            .with("frame", "i32 compressed,i32 raw,i32 crc32c,i32 samples,zstd")
            .with("level", LEVEL)
            .with(
                "sample",
                "i64 game,u16 longs,i64 observation,f32 y relative x4,u16 stratum,f32 weight")
            .with("stratum", LuckStratum.DEFINITION)
            .with("strata", LuckStratum.COUNT)
            .hex();
  }
}
