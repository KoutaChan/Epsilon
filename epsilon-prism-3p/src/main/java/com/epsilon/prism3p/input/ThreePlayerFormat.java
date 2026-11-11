package com.epsilon.prism3p.input;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.FieldSpec;
import com.epsilon.prism.input.TileTokens;

/**
 * 三人打ちの入力のフィールドが、値の範囲と語彙を決めるのに使う定数とメソッド。フィールドの enum の初期化から読むので、入力の形式のクラス
 * （{@link InputSchema}）とは別に置く。同じクラスに置くと、フィールドの enum を先に使ったときに初期化が循環する。
 */
final class ThreePlayerFormat {

  static final Variant VARIANT = Variant.TENHOU_3P;

  /** 席の数。 */
  static final int SEATS = VARIANT.seats();

  /** 牌のトークンの数（2〜8 萬を除く 27 種類）。 */
  static final int TILES = TileTokens.of(VARIANT).count();

  /** 1 行に入るイベントのトークンの上限（記録の公開のイベント 87 ＋ 槓ドラの表示 4）。 */
  static final int MAX_EVENTS = VARIANT.maxPublicEvents() + Limits.MAX_INDICATORS - 1;

  private ThreePlayerFormat() {}

  /**
   * 三人打ちの語彙で値を書く enum のフィールドを返す。三人打ちの語彙は、三人打ちで起こりうる定数だけを宣言の順に並べたもの。
   *
   * @param type 値の enum
   * @return フィールドの仕様
   */
  static FieldSpec oneOf(Class<? extends Enum<?>> type) {
    return FieldSpec.oneOf(type, VARIANT);
  }
}
