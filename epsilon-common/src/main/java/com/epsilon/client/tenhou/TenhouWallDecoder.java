package com.epsilon.client.tenhou;

import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Base64;

/**
 * 天鳳牌譜の SHUFFLE シードから牌山（136枚の実牌ID配列）を復元する。
 *
 * <p>アルゴリズム (天鳳公式ブログ 2009-07-13 準拠):
 *
 * <ol>
 *   <li>Base64 デコード → 624 個の 32bit LE 整数
 *   <li>MT19937 を init_by_array で初期化
 *   <li>各局ごとに MT から 288 個生成し、9 ブロック SHA512 → rnd[144]
 *   <li>前方向 Fisher-Yates で 136 枚シャッフル
 *   <li>通常ツモが先頭から並ぶよう反転し、嶺上牌だけ共通実装の牌山配置へ変換
 * </ol>
 *
 * @see <a href="https://tenhou.net/stat/rand/">天鳳乱数ハッシュ公開</a>
 */
public final class TenhouWallDecoder {

  private static final int WALL_SIZE = 136;
  private static final int SHA512_BYTES = 64;
  private static final int SHA512_DWORDS = SHA512_BYTES / 4; // 16
  private static final int NUM_SHA_BLOCKS = 9;
  private static final int RND_SIZE = SHA512_DWORDS * NUM_SHA_BLOCKS; // 144
  private static final int SRC_SIZE = RND_SIZE * 2; // 288

  private TenhouWallDecoder() {}

  /**
   * SHUFFLE シードから指定局の牌山を復元する。
   *
   * @param shuffleSeed SHUFFLE タグのシード属性値
   * @param kyokuIndex 局番号 (0 = 最初の局)
   * @return 136 要素の実牌ID配列
   */
  public static int[] decodeWall(String shuffleSeed, int kyokuIndex) {
    int[] seedArray = parseSeed(shuffleSeed);

    MT19937 mt = new MT19937();
    mt.initByArray(seedArray);

    // 目的の局まで MT を進める (各局で SRC_SIZE 個消費)
    for (int k = 0; k < kyokuIndex; k++) {
      for (int i = 0; i < SRC_SIZE; i++) {
        mt.nextInt();
      }
    }

    return generateWall(mt);
  }

  /**
   * SHUFFLE シードから最初の局の牌山を復元する。
   *
   * @param shuffleSeed SHUFFLE タグのシード属性値
   * @return 最初の局について、配牌から消費する順に並んだ136要素の実牌 ID
   */
  public static int[] decodeWall(String shuffleSeed) {
    return decodeWall(shuffleSeed, 0);
  }

  /**
   * SHUFFLE シードから先頭の複数局の牌山を、MT 状態を戻さず順番に復元する。
   *
   * @param shuffleSeed SHUFFLE タグのシード属性値
   * @param count 復元する先頭からの局数
   * @return 局インデックスごとに136要素の実牌 ID を保持する二次元配列
   */
  public static int[][] decodeWalls(String shuffleSeed, int count) {
    int[][] walls = new int[count][];
    MT19937 mt = new MT19937();
    mt.initByArray(parseSeed(shuffleSeed));
    for (int kyoku = 0; kyoku < count; kyoku++) {
      walls[kyoku] = generateWall(mt);
    }
    return walls;
  }

  /** 現在の MT 状態から 1 局分の牌山を生成する。 */
  private static int[] generateWall(MT19937 mt) {
    // MT から 288 個の乱数を生成
    int[] src = new int[SRC_SIZE];
    for (int i = 0; i < SRC_SIZE; i++) {
      src[i] = mt.nextInt();
    }

    // SHA512 ハッシュ: 9 ブロック × (1024bit → 512bit)
    int[] rnd = hashToRnd(src);

    // 前方向 Fisher-Yates シャッフル
    int[] wall = new int[WALL_SIZE];
    for (int i = 0; i < WALL_SIZE; i++) {
      wall[i] = i;
    }
    for (int i = 0; i < WALL_SIZE - 1; i++) {
      int j = i + (int) (Integer.toUnsignedLong(rnd[i]) % (WALL_SIZE - i));
      int tmp = wall[i];
      wall[i] = wall[j];
      wall[j] = tmp;
    }

    // 天鳳では牌山を末尾から消費する → 反転して [0]=最初の配牌, [52]=最初のツモの順にする
    for (int l = 0, r = WALL_SIZE - 1; l < r; l++, r--) {
      int tmp = wall[l];
      wall[l] = wall[r];
      wall[r] = tmp;
    }

    // 天鳳の嶺上順 134,135,132,133 を、共通実装の牌山が末尾から読む順序へ合わせる
    for (int l = WALL_SIZE - 4; l < WALL_SIZE; l += 2) {
      int tmp = wall[l];
      wall[l] = wall[l + 1];
      wall[l + 1] = tmp;
    }

    return wall;
  }

  /**
   * src[288] を 9 ブロックの SHA512 でハッシュして rnd[144] を返す。
   *
   * <p>各ブロック: src[i*32 .. i*32+31] (1024bit) → SHA512 → rnd[i*16 .. i*16+15] (512bit)
   */
  private static int[] hashToRnd(int[] src) {
    int[] rnd = new int[RND_SIZE];
    try {
      MessageDigest sha512 = MessageDigest.getInstance("SHA-512");
      for (int block = 0; block < NUM_SHA_BLOCKS; block++) {
        // 32 DWORD → 128 バイト(リトルエンディアン)。
        byte[] input = new byte[SHA512_BYTES * 2];
        for (int j = 0; j < 32; j++) {
          int val = src[block * 32 + j];
          input[j * 4] = (byte) val;
          input[j * 4 + 1] = (byte) (val >>> 8);
          input[j * 4 + 2] = (byte) (val >>> 16);
          input[j * 4 + 3] = (byte) (val >>> 24);
        }
        byte[] hash = sha512.digest(input);
        // 64 バイト → 16 DWORD(リトルエンディアン)。
        for (int j = 0; j < SHA512_DWORDS; j++) {
          rnd[block * SHA512_DWORDS + j] =
              (hash[j * 4] & 0xFF)
                  | ((hash[j * 4 + 1] & 0xFF) << 8)
                  | ((hash[j * 4 + 2] & 0xFF) << 16)
                  | ((hash[j * 4 + 3] & 0xFF) << 24);
        }
      }
    } catch (NoSuchAlgorithmException e) {
      throw new AssertionError("SHA-512 not available", e);
    }
    return rnd;
  }

  /** SHUFFLE シード文字列を解析し、int 配列を返す。 */
  static int[] parseSeed(String shuffleSeed) {
    int commaIdx = shuffleSeed.indexOf(',');
    if (commaIdx < 0) {
      throw new IllegalArgumentException("Invalid SHUFFLE seed format: no comma");
    }
    String base64Part = shuffleSeed.substring(commaIdx + 1);
    byte[] raw = Base64.getDecoder().decode(base64Part);
    if (raw.length % 4 != 0) {
      throw new IllegalArgumentException(
          "Seed data length " + raw.length + " is not a multiple of 4");
    }

    // リトルエンディアンの 32ビット整数列。
    int[] result = new int[raw.length / 4];
    for (int i = 0; i < result.length; i++) {
      result[i] =
          (raw[i * 4] & 0xFF)
              | ((raw[i * 4 + 1] & 0xFF) << 8)
              | ((raw[i * 4 + 2] & 0xFF) << 16)
              | ((raw[i * 4 + 3] & 0xFF) << 24);
    }
    return result;
  }

  /** Mersenne Twister 19937 (MT19937ar) の Java 実装。 */
  static final class MT19937 {

    private static final int N = 624;
    private static final int M = 397;
    private static final int MATRIX_A = 0x9908b0df;
    private static final int UPPER_MASK = 0x80000000;
    private static final int LOWER_MASK = 0x7fffffff;

    private final int[] mt = new int[N];
    private int mti = N + 1;

    void initGenrand(int seed) {
      mt[0] = seed;
      for (mti = 1; mti < N; mti++) {
        mt[mti] = 1812433253 * (mt[mti - 1] ^ (mt[mti - 1] >>> 30)) + mti;
      }
    }

    void initByArray(int[] initKey) {
      initGenrand(19650218);
      int i = 1, j = 0;
      int k = Math.max(N, initKey.length);
      for (; k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >>> 30)) * 1664525)) + initKey[j] + j;
        i++;
        j++;
        if (i >= N) {
          mt[0] = mt[N - 1];
          i = 1;
        }
        if (j >= initKey.length) {
          j = 0;
        }
      }
      for (k = N - 1; k > 0; k--) {
        mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >>> 30)) * 1566083941)) - i;
        i++;
        if (i >= N) {
          mt[0] = mt[N - 1];
          i = 1;
        }
      }
      mt[0] = UPPER_MASK;
    }

    int nextInt() {
      int y;
      if (mti >= N) {
        int kk;
        for (kk = 0; kk < N - M; kk++) {
          y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
          mt[kk] = mt[kk + M] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        }
        for (; kk < N - 1; kk++) {
          y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
          mt[kk] = mt[kk + (M - N)] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        }
        y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
        mt[N - 1] = mt[M - 1] ^ (y >>> 1) ^ ((y & 1) != 0 ? MATRIX_A : 0);
        mti = 0;
      }
      y = mt[mti++];
      y ^= (y >>> 11);
      y ^= (y << 7) & 0x9d2c5680;
      y ^= (y << 15) & 0xefc60000;
      y ^= (y >>> 18);
      return y;
    }
  }
}
