package com.epsilon.ai.grp;

import com.epsilon.core.GameState;

/** Epsilon GRP 入力。半荘進行度と公開された点棒状態を持つ。 */
public final class EpsilonGrpFeature {

  /** 1局境界あたりの特徴量数。 */
  public static final int FEATURE_SIZE = 7;

  /**
   * 正規化局番号。親席は独立特徴量にしない。
   *
   * <p>通常の半荘では東1の親が席 0 で、連荘時は局番号と親が共に据え置かれ、親流れ時は共に1進む。正規化値に {@link #KYOKU_SCALE} を掛けて丸めれば局番号が戻り、親席は
   * {@code floorMod(decodedKyokuIndex, 4)} で復元できる。
   */
  public static final int KYOKU_INDEX = 0;

  /** 正規化した本場数のインデックス。 */
  public static final int HONBA = 1;

  /** 正規化した供託リーチ棒数のインデックス。 */
  public static final int KYOTAKU = 2;

  /** 席 0 の正規化得点が始まるインデックス。 */
  public static final int SCORE_OFFSET = 3;

  /** 局番号を得点特徴と近い尺度へ揃える正規化係数。 */
  public static final float KYOKU_SCALE = 8.0f;

  /** 本場数を得点特徴と近い尺度へ揃える正規化係数。 */
  public static final float HONBA_SCALE = 8.0f;

  /** 供託数を得点特徴と近い尺度へ揃える正規化係数。 */
  public static final float KYOTAKU_SCALE = 4.0f;

  private static final float SCORE_SCALE = 10_000.0f;

  private EpsilonGrpFeature() {}

  /**
   * 現在局の進行度と全員の得点を固定幅特徴量に変換する。
   *
   * @param state 特徴量を読む対局状態
   * @return 長さ {@link #FEATURE_SIZE} の正規化特徴量
   */
  public static float[] fromState(GameState state) {
    float[] feature = new float[FEATURE_SIZE];
    feature[KYOKU_INDEX] = state.getKyokuIndex() / KYOKU_SCALE;
    feature[HONBA] = state.getHonba() / HONBA_SCALE;
    feature[KYOTAKU] = state.getKyotakuCount() / KYOTAKU_SCALE;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      feature[SCORE_OFFSET + player] = state.getScore(player) / SCORE_SCALE;
    }
    return feature;
  }

  /**
   * 牌譜から復元した局進行情報を固定幅特徴量に変換する。
   *
   * @param kyokuIndex 0始まりの通算局番号
   * @param honba 本場数
   * @param kyotakuCount 供託リーチ棒数
   * @param scores 席ごとの得点
   * @return 長さ {@link #FEATURE_SIZE} の正規化特徴量
   */
  public static float[] fromProgress(int kyokuIndex, int honba, int kyotakuCount, int[] scores) {
    float[] feature = new float[FEATURE_SIZE];
    feature[KYOKU_INDEX] = kyokuIndex / KYOKU_SCALE;
    feature[HONBA] = honba / HONBA_SCALE;
    feature[KYOTAKU] = kyotakuCount / KYOTAKU_SCALE;
    for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
      feature[SCORE_OFFSET + player] = scores[player] / SCORE_SCALE;
    }
    return feature;
  }

  /**
   * 一次元化済み GRP 系列に含まれる局境界数を返す。
   *
   * @param flattened {@code [steps, FEATURE_SIZE]} を一次元化した配列
   * @return 系列のステップ数
   * @throws IllegalArgumentException 配列長が特徴量幅で割り切れない場合
   */
  public static int steps(float[] flattened) {
    if (flattened.length == 0) {
      return 0;
    }
    if (flattened.length % FEATURE_SIZE != 0) {
      throw new IllegalArgumentException("GRP sequence length is not divisible by feature size");
    }
    return flattened.length / FEATURE_SIZE;
  }

  /**
   * 一次元化済み GRP 系列の形状と有限性を検証して複製する。
   *
   * @param flattened 検証する特徴量系列
   * @return 呼び出し元と共有しない検証済み配列
   * @throws IllegalArgumentException 形状が不正か非有限値を含む場合
   */
  public static float[] validatedCopy(float[] flattened) {
    if (flattened.length % FEATURE_SIZE != 0) {
      throw new IllegalArgumentException("GRP sequence length is not divisible by feature size");
    }
    float[] out = flattened.clone();
    for (float value : out) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("GRP feature contains non-finite value");
      }
    }
    return out;
  }
}
