package com.epsilon.ai.grp;

import com.epsilon.core.GameState;
import java.util.Arrays;

/**
 * 半荘の開始から現在の局までのGRP特徴量系列を保持する。
 *
 * <p>同じ局では同じ読み取り専用配列を返す。局番号または本場数が変わると、新しい局の特徴量を加えた配列を作る。呼び出し元は{@link #current()}の戻り値を変更してはならない。
 */
public final class EpsilonGrpSequence {

  private float[] flattened = new float[0];

  /** 空の半荘特徴量系列を作る。 */
  public EpsilonGrpSequence() {}

  /**
   * 現在局が未収録なら状態の局境界特徴量を追加する。
   *
   * @param state 現在の対局状態
   * @return 追加後の読み取り専用の特徴量系列
   */
  public float[] include(GameState state) {
    if (!contains(state.getKyokuIndex(), state.getHonba())) {
      append(EpsilonGrpFeature.fromState(state));
    }
    return flattened;
  }

  /**
   * 現在局が未収録なら牌譜から復元した進行情報を特徴量系列へ追加する。
   *
   * @param kyokuIndex 0始まりの通算局番号
   * @param honba 本場数
   * @param kyotakuCount 供託リーチ棒数
   * @param scores 席ごとの得点
   * @return 追加後の読み取り専用の特徴量系列
   */
  public float[] include(int kyokuIndex, int honba, int kyotakuCount, int[] scores) {
    if (!contains(kyokuIndex, honba)) {
      append(EpsilonGrpFeature.fromProgress(kyokuIndex, honba, kyotakuCount, scores));
    }
    return flattened;
  }

  /**
   * 現在の読み取り専用の特徴量系列を返す。
   *
   * @return 一次元化済み GRP 特徴量系列
   */
  public float[] current() {
    return flattened;
  }

  /** 保持している半荘特徴量系列を空に戻す。 */
  public void clear() {
    flattened = new float[0];
  }

  private boolean contains(int kyokuIndex, int honba) {
    if (flattened.length == 0) {
      return false;
    }
    int last = flattened.length - EpsilonGrpFeature.FEATURE_SIZE;
    return flattened[last + EpsilonGrpFeature.KYOKU_INDEX]
            == kyokuIndex / EpsilonGrpFeature.KYOKU_SCALE
        && flattened[last + EpsilonGrpFeature.HONBA] == honba / EpsilonGrpFeature.HONBA_SCALE;
  }

  private void append(float[] feature) {
    int offset = flattened.length;
    flattened = Arrays.copyOf(flattened, offset + EpsilonGrpFeature.FEATURE_SIZE);
    System.arraycopy(feature, 0, flattened, offset, EpsilonGrpFeature.FEATURE_SIZE);
  }
}
