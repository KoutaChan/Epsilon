package com.epsilon.prism.core;

import com.epsilon.agent.play.TableSeeds;
import com.epsilon.mahjong.util.SeedMixer;

/**
 * PRISM の乱数は全てここから作る。用途ごとのドメインと座標から決定的に導き、実行の順・バッチの組み方・スレッド数には依らない。牌山の種と
 * 局のキーは卓が作る（{@link TableSeeds}）。
 *
 * <p>頻繁に呼ばれる所で使うので、{@link SeedMixer#derive} の可変長引数による配列の確保を避け、引数の数ごとのメソッドを持つ。
 *
 * <p>{@link Domain#HAND_KEY}・{@link Domain#HOLDOUT} のソルトを変えると、同じ run の種でもコホートの種とホールドアウトの半荘が変わる。
 *
 * <p>判断ごとの乱数は、経路ごとの乱数のキー（{@link #mainKey}・{@link #copyKey}）から {@code unit(derive(キー, ドメイン, 判断 ID[, 座標]))} で
 * 引く。判断 ID はエンジンの通し番号で、応答できる席の数によって枝ごとにずれる。見送った鳴きのように、応答できる席があるかは他家の
 * 伏せた手牌で決まる。したがって乱数は、席の観測だけで決まる値ではない。それでもキーは牌山とも行動の抽選（卓の局のキー）とも別の
 * ソルトのハッシュなので、隠れた状態 H を固定しても乱数は [0, 1) の一様な値になる。行の採否・候補の 1 手・逸脱の確率は H に依らない。
 * 乱数のキーを牌山の種と共有してはならない。共有するとこの性質が壊れる。
 */
public final class CrnSeeds {
  /** 乱数の用途。値を変えると行を再現できなくなるので変えない。 */
  public enum Domain {
    /** コホートの種。 */
    HAND_KEY(0x48414e44L),
    /** 半荘を学習用とホールドアウトのどちらに使うかの抽選。 */
    HOLDOUT(0x484f4c44L),
    /** 学習の行の並べ替え。 */
    LEARN(0x4c45524eL),
    /** 主経路の乱数のキー（半荘ごと）。 */
    MAIN(0x4d41494eL),
    /** 写しの乱数のキー（半荘と、写しの根の判断 ID ごと）。 */
    COPY(0x434f5059L),
    /** ラベル行を予約するかの乱数（率 ρ_c）。 */
    LABEL(0x4c41424cL),
    /** 中立ロールインにするかの乱数（率 β_c）。 */
    ROLLIN(0x524f4c4cL),
    /** 逸脱 d ~ q(·|o) の抽選。 */
    DEVIATION(0x44455654L),
    /** 候補集合から一様に選ぶ 1 手（座標 1: 率 ε_C で選ぶかの乱数、座標 2: どの手か）。 */
    EXPLORE(0x45585052L),
    /** 写しの中で逸脱した席の行を、ラベル行にするかの乱数（率 ρ_br）。 */
    COPY_LABEL(0x43504c42L),
    /** 運の精算の帳簿を行に書くときの確率的な丸めの乱数（座標 1: 半荘、座標 2: 判断 ID、座標 3: 手 × フィールドの数 + フィールド）。 */
    LEDGER(0x4c454447L),
    /** 裏ドラの期待値も求めて監査する局の抽選（座標 1: 半荘、座標 2: 半荘の中の局の序数）。同じ局の主経路・写し・分岐で同じ値になる。 */
    URA_AUDIT(0x55524141L);

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
    return extend(derive(base, domain, a), b);
  }

  public static long derive(long base, Domain domain, long a, long b, long c) {
    return extend(derive(base, domain, a, b), c);
  }

  /**
   * 導いた値に座標を 1 つ足す。座標の多い {@link #derive} は座標をこの形で 1 つずつ足すので、前の座標までの値を求めておけば、後の座標を
   * 足すだけで同じ値になる（{@code derive(base, domain, a, b, c) == extend(derive(base, domain, a, b), c)}）。
   */
  public static long extend(long derived, long coordinate) {
    return SeedMixer.mix64(derived ^ SeedMixer.mix64(coordinate));
  }

  /** ビットから [0, 1) の一様な値を作る。 */
  public static double unit(long bits) {
    return (bits >>> 11) * UNIT;
  }

  /** 主経路の乱数のキーを返す。コホートの種と半荘 ID だけで決まる。 */
  public static long mainKey(long iterationSeed, long gameId) {
    return derive(iterationSeed, Domain.MAIN, gameId);
  }

  /** 写しの乱数のキーを返す。コホートの種、半荘 ID、写しを作った主経路の判断 ID だけで決まる。 */
  public static long copyKey(long iterationSeed, long gameId, long rootDecisionId) {
    return derive(iterationSeed, Domain.COPY, gameId, rootDecisionId);
  }

  /** キー・ドメイン・判断 ID から、[0, 1) の一様な乱数を引く。 */
  public static double coin(long key, Domain domain, long decisionId) {
    return unit(derive(key, domain, decisionId));
  }

  /** キー・ドメイン・判断 ID・座標から、[0, 1) の一様な乱数を引く。 */
  public static double coin(long key, Domain domain, long decisionId, long coordinate) {
    return unit(derive(key, domain, decisionId, coordinate));
  }
}
