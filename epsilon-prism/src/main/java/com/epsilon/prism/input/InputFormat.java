package com.epsilon.prism.input;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputBlob.Composition;
import com.epsilon.prism.input.InputBlob.Tensor;
import com.epsilon.prism.variant.PrismVariant;
import java.util.List;

/**
 * ルールごとのネットワークの入力形式。ネットワーク・HIP とのつなぎ・方策・学習は、入力のフィールドの宣言を直接読まない。
 * このインターフェースから並び・幅・キーの範囲・手の扱いの表を読み、1 行分のライターを作る。
 */
public interface InputFormat {

  /**
   * ルールの入力形式を返す。
   *
   * @param variant ルール
   * @return 入力形式
   * @throws IllegalStateException ルールのモジュールが classpath に無いとき
   */
  static InputFormat of(Variant variant) {
    return PrismVariant.of(variant).input();
  }

  /** この入力形式のルール。 */
  Variant variant();

  /** 終局の効用と待ちの射影の両方に使う効用。系譜の設定の効用と同じでなければならない。 */
  RankUtility utility();

  /**
   * 入力のフィンガープリントを返す。model.json に書き、モデルを開くときに照合する。
   *
   * @return SHA-256 の先頭 8 B の 16 進
   */
  String fingerprint();

  /** 局のトークンのフィールドの配置。 */
  TokenLayout<?> round();

  /** 席のトークンのフィールドの配置。 */
  TokenLayout<?> seat();

  /** 牌のトークンのフィールドの配置。 */
  TokenLayout<?> tile();

  /** イベントのトークンのフィールドの配置。 */
  TokenLayout<?> event();

  /** 遷移のフィールドの配置。 */
  TokenLayout<?> transition();

  /** 関係の属性の配置。 */
  TokenLayout<?> relation();

  /**
   * 軸の 1 つあたりの要素の数を返す。バッチを準備するときに呼び、行を書くたびには呼ばない。
   *
   * @param tensor 領域
   * @return 要素の数
   */
  int width(Tensor tensor);

  /** バッチの 1 行が持てるイベントのトークンの数。 */
  int eventCapacity();

  /** 遷移ごとのキーの範囲。 */
  Keys keys();

  /** 行ごとの席のトークンの数（席の数）。 */
  default int seats() {
    return keys().seat();
  }

  /** 行ごとの牌のトークンの数。 */
  default int tiles() {
    return keys().tile();
  }

  /** 行ごとに必ずあるトークン（局・席・牌）の数。 */
  default int fixedTokens() {
    return keys().count();
  }

  /** 行ごとのイベントのトークンの上限（公開イベントの上限 ＋ 槓ドラの表示牌の数）。 */
  default int maxEvents() {
    return variant().maxPublicEvents() + Limits.MAX_INDICATORS - 1;
  }

  /** 牌のトークンの並び。 */
  default TileTokens tileTokens() {
    return TileTokens.of(variant());
  }

  /** 手の扱いの表（番号の順）。 */
  List<? extends Role> roles();

  /**
   * 手の種類ごとの扱いを返す。
   *
   * @param type 手の種類
   * @return 手の扱い
   * @throws IllegalArgumentException このルールの入力に無い種類のとき
   */
  Role role(Action.Type type);

  /** 打牌の扱い。 */
  Role discard();

  /** 見送りの扱い。 */
  Role pass();

  /**
   * 関係の属性の列を返す。
   *
   * @param attribute 属性の種類
   * @param repeat 繰り返しの番号（0 始まり）
   * @return 列
   */
  int relationColumn(RelationRole attribute, int repeat);

  /**
   * 関係の属性の繰り返しの数を返す。
   *
   * @param attribute 属性の種類
   * @return 繰り返しの数
   */
  int relationRepeats(RelationRole attribute);

  /**
   * 遷移のフィールドの列を返す。
   *
   * @param field フィールドの種類
   * @return 列
   */
  int transitionColumn(TransitionRole field);

  /**
   * 1 行分のライターを作る。ライターはスレッドごとに 1 つ持つ。
   *
   * @return ライター
   */
  RowWriter writer();

  /**
   * 遷移ごとのキーの範囲。キーは局・席・牌の順に並ぶ。コードの番号は、局・席・牌の範囲をまたいで続けて振る。
   *
   * @param round 局のキーの数
   * @param seat 席のキーの数
   * @param tile 牌のキーの数
   * @param codes コードの総数（Ek・Ev の行の数）
   */
  record Keys(int round, int seat, int tile, int codes) {

    /** キーの総数（遷移のコードの領域の列の数）。 */
    public int count() {
      return round + seat + tile;
    }
  }

  /** 局・席・牌のどれか 1 つのキーの範囲。 */
  interface KeyRange {

    /** 範囲のキーの数。 */
    int keys();

    /** 範囲のコードの数。 */
    int codes();

    /** 範囲の最初のキーの列。 */
    int firstKey();

    /** 範囲の最初のコード。 */
    int firstCode();
  }

  /** BF16 で送る数値のフィールド。値は {@link #scale} で割ってから丸める。列は領域の中の {@link #ordinal} 番目。 */
  interface Scalar {

    /** フィールドの領域（{@link Tensor#ROUND_SCALAR}・{@link Tensor#SEAT_SCALAR}・{@link Tensor#EVENT_SCALAR}・{@link Tensor#TRANSITION_SCALAR}）。 */
    Tensor tensor();

    /** 領域の中の列。 */
    int ordinal();

    /** 値を割る数。 */
    float scale();
  }

  /** FP32 のままヘッドへ渡す厳密なフィールド。値は {@link #scale} で割ってから送る。列は領域の中の {@link #ordinal} 番目。 */
  interface Exact {

    /** フィールドの領域（{@link Tensor#DISCARD_EXACT} か {@link Tensor#ACTION_EXACT}）。 */
    Tensor tensor();

    /** 領域の中の列。 */
    int ordinal();

    /** 値を割る数。 */
    float scale();
  }

  /** 価値グラフでの手の扱い。番号は {@link InputFormat#roles()} の表の順。 */
  interface Role {

    /** 表の中の番号。 */
    int ordinal();

    /** 扱いの名前。 */
    String name();

    /** ゲートのヘッドが値を出す手なら true。 */
    boolean gate();

    /** s の作り方。 */
    Composition composition();

    /** 局の終わり方が手で決まる（和了・九種九牌）なら true。 */
    boolean endsHand();
  }

  /** ネットワークが読む関係の属性の種類。 */
  enum RelationRole {
    TILE,
    OWNER,
    EVENT,
    OWNER_RIICHI_EVENT,
    OWNER_LAST_DISCARD_EVENT,
    LINK,
    SAFETY
  }

  /** ネットワークが読む遷移のフィールドの種類。 */
  enum TransitionRole {
    /** 主牌（打牌・鳴いた牌・槓の牌）。 */
    MAIN_TILE,
    /** 鳴いた後に捨てる牌。 */
    CALL_DISCARD
  }

  /** 1 行分の入力を書くライター。スレッドセーフではない。 */
  interface RowWriter {

    /**
     * 1 行分の入力を、バッチの {@code row} 番目の行に書く。バッチの計画には、この行を {@code row} 番目に足してあること。
     *
     * @param o 行の判断の記録
     * @param batch 書き先のバッチ
     * @param row バッチの中の行の位置
     */
    void write(Observation o, HostInputs batch, int row);

    /**
     * 物理の牌の種類を、色を正規化した順の牌のトークンに直す。色の置き換えは、直前に書いた行のものを使う。
     *
     * @param physicalTile 牌の種類
     * @return 色を正規化した牌のトークンの番号（{@link TileTokens}）
     */
    int canonicalTile(int physicalTile);
  }
}
