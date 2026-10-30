package com.epsilon.prism.luck;

import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.util.SeedMixer;

/**
 * 運の網 Ω の標本を引く硬貨。出来事ごとに鍵 (コホートの種, 半荘, 経路, 根の判断, 根からの序数) のハッシュの一様値を引き、率
 * {@code rate} より小さい出来事を標本にする。重みは 1/rate。
 *
 * <p>鍵は出来事の結果を含まず、序数は根からその出来事までの出来事の数なので、硬貨は出来事の前の状態だけで決まる（結果と独立）。塩は
 * {@link CrnSeeds} のどのドメインとも違うので、牌山・手の抽選・ほかの硬貨とも独立。
 *
 * <p>兄弟の分岐（同じ判断から別の手で分かれた経路）は同じ根を置くので、同じ序数の出来事は一緒に引かれる（行の中で中心化した c の
 * ばらつきを測れる）。根は {@code GameRunner} が置く（{@link LuckCapture#root}）:
 *
 * <ul>
 *   <li>主経路: 局の始めに (半荘, {@link #MAIN_PATH}, {@link #handRoot}(局の序数))、写しか分岐を作った判断 D の後は
 *       (半荘, MAIN_PATH, D)。打った手の続きが D の分岐・写しと同じ鍵になる。
 *   <li>主経路の判断 D の分岐・写し: (半荘, MAIN_PATH, D)。
 *   <li>根 R の写しの中で分岐を作った判断 D′ の後とその分岐: (半荘, R, D′)。写しの判断 ID は主経路と同じ数を使い回すので、経路 R で分ける。
 * </ul>
 *
 * @param cohortSeed コホートの種
 * @param rate 標本の率（(0, 1]）
 */
public record LuckSampling(long cohortSeed, double rate) {

  /** 主経路の経路の番号（写しは根の判断 ID）。 */
  public static final long MAIN_PATH = -1L;

  /** 塩（"LUCK"）。 */
  private static final long SALT = 0x4c55434bL;

  /**
   * 硬貨を作る。
   *
   * @throws IllegalArgumentException 率が (0, 1] の外のとき
   */
  public LuckSampling {
    if (!(rate > 0 && rate <= 1)) {
      throw new IllegalArgumentException("sampling rate must be in (0, 1]: " + rate);
    }
  }

  /**
   * 主経路の局の初めの根（ラベル行の前の出来事の根）を返す。判断 ID（0 以上）と重ならないよう負にする。
   *
   * @param handOrdinal 半荘の中の局の序数（0 始まり）
   * @return 根
   */
  public static long handRoot(int handOrdinal) {
    return -1L - handOrdinal;
  }

  /**
   * 出来事を標本にするかを返す。
   *
   * @param gameId 半荘 ID
   * @param path 経路（主経路は {@link #MAIN_PATH}、写しの中は写しの根の判断 ID）
   * @param root 根の判断 ID（主経路の局の初めは {@link #handRoot}）
   * @param ordinal 根からの出来事の序数（0 始まり）
   * @return 標本にするなら true
   */
  public boolean sampled(long gameId, long path, long root, int ordinal) {
    long bits = SeedMixer.mix64(SeedMixer.mix64(cohortSeed ^ SALT) ^ SeedMixer.mix64(gameId));
    bits = SeedMixer.mix64(bits ^ SeedMixer.mix64(path));
    bits = SeedMixer.mix64(bits ^ SeedMixer.mix64(root));
    bits = SeedMixer.mix64(bits ^ SeedMixer.mix64(ordinal));
    return CrnSeeds.unit(bits) < rate;
  }

  /** 標本の重み 1/rate。 */
  public float weight() {
    return (float) (1.0 / rate);
  }
}
