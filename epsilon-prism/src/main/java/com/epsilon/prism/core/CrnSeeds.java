package com.epsilon.prism.core;

import com.epsilon.util.SeedMixer;

/**
 * 乱数の唯一の素材。用途ごとのドメインと座標から決定的に導出し、実行順・バッチ構成・スレッド数に依存させない。
 *
 * <p>熱い経路で使うので、{@link SeedMixer#derive} の可変長引数による確保を避けた固定長の版を持つ。
 *
 * <p>WALL・HAND_KEY・HOLDOUT の塩を変えると、同じ run の種と半荘 ID でも牌山・局のキー（行動の共通乱数）・保留の半荘が変わる。
 *
 * <p>判断ごとの硬貨は、進行ごとの硬貨の鍵（{@link #mainKey}・{@link #copyKey}）から {@code unit(derive(鍵, ドメイン, 判断 ID[, 座標]))} で引く。
 * 座標の判断 ID はエンジンの通し番号で、応答できる席の数で枝ごとにずれる（見送った鳴きの申し出など、他家の伏せた手牌に依る）。したがって硬貨は席の
 * 観測だけの関数ではない。それでも鍵は牌山（{@link Domain#WALL}）とも行動の抽選（{@link Domain#HAND_KEY}）とも別の塩のハッシュなので、隠れた状態 H
 * を固定しても硬貨は [0, 1) の一様値で、行の採否・候補の 1 手・逸脱の確率は H に依らない（論文の命題 3 (d)・命題 4 の A3 はこの意味で成り立つ）。
 * 硬貨の鍵を牌山の種と共有する変更はこれを壊すので入れない。
 */
public final class CrnSeeds {
  /** 用途の分離。値は行の再現性に効くので変えない。 */
  public enum Domain {
    /** 牌山。 */
    WALL(0x57414c4cL),
    /** 局のキーとコホートの種。 */
    HAND_KEY(0x48414e44L),
    /** 半荘の用途（TRAIN・HOLDOUT）の抽選。 */
    HOLDOUT(0x484f4c44L),
    /** 学習の行の並べ替え。 */
    LEARN(0x4c45524eL),
    /** 半荘の組（A・B）の抽選。 */
    GROUP(0x47525550L),
    /** 主経路の硬貨の鍵（半荘ごと）。 */
    MAIN(0x4d41494eL),
    /** 写しの硬貨の鍵（半荘と写しの根の判断 ID ごと）。 */
    COPY(0x434f5059L),
    /** ラベル行の予約の硬貨（率 ρ_c）。 */
    LABEL(0x4c41424cL),
    /** 中立ロールインの硬貨（率 β_c）。 */
    ROLLIN(0x524f4c4cL),
    /** 逸脱 d ~ q(·|o) の抽選。 */
    DEVIATION(0x44455654L),
    /** 候補集合の一様な 1 手（座標 1: 率 ε_C の硬貨、座標 2: どの手か）。 */
    EXPLORE(0x45585052L),
    /** 写しの中の逸脱した席のラベル行の硬貨（率 ρ_br）。 */
    COPY_LABEL(0x43504c42L),
    /** データの記憶で古いコホートの行を間引く硬貨（座標 1: コホート、座標 2: コホートの中の行の番号）。 */
    MEMORY(0x4d454d4fL);

    final long salt;

    Domain(long salt) {
      this.salt = salt;
    }
  }

  private static final double UNIT = 0x1.0p-53;

  private CrnSeeds() {}

  public static long derive(long base, Domain domain, long a) {
    return SeedMixer.mix64(SeedMixer.mix64(base ^ domain.salt) ^ SeedMixer.mix64(a));
  }

  public static long derive(long base, Domain domain, long a, long b) {
    return SeedMixer.mix64(derive(base, domain, a) ^ SeedMixer.mix64(b));
  }

  public static long derive(long base, Domain domain, long a, long b, long c) {
    return SeedMixer.mix64(derive(base, domain, a, b) ^ SeedMixer.mix64(c));
  }

  /** [0, 1) の一様値。 */
  public static double unit(long bits) {
    return (bits >>> 11) * UNIT;
  }

  /** (0, 1) の一様値。対数を取る用途向けに 0 を含めない。 */
  public static double openUnit(long bits) {
    return ((bits >>> 11) + 0.5) * UNIT;
  }

  /** 主経路の硬貨の鍵。コホートの種と半荘 ID だけで決まる。 */
  public static long mainKey(long iterationSeed, long gameId) {
    return derive(iterationSeed, Domain.MAIN, gameId);
  }

  /** 写しの硬貨の鍵。コホートの種、半荘 ID、写しを作った主経路の判断 ID だけで決まる。 */
  public static long copyKey(long iterationSeed, long gameId, long rootDecisionId) {
    return derive(iterationSeed, Domain.COPY, gameId, rootDecisionId);
  }

  /** 鍵 key の判断 decisionId のドメイン domain の硬貨（[0, 1)）。 */
  public static double coin(long key, Domain domain, long decisionId) {
    return unit(derive(key, domain, decisionId));
  }

  /** 鍵 key の判断 decisionId のドメイン domain の座標 coordinate の硬貨（[0, 1)）。 */
  public static double coin(long key, Domain domain, long decisionId, long coordinate) {
    return unit(derive(key, domain, decisionId, coordinate));
  }
}
