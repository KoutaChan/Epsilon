package com.epsilon.pico.ai.decision.policy;

import com.epsilon.core.Action;
import java.util.HashSet;
import java.util.Set;

/**
 * 和了・鳴き・槓などの条件付き分岐で用いるスコアの種類と出力位置を定義する。
 *
 * <p>終端二択の判定、CALL/KAN の二択の判定、MELD/KAN の条件付き型選択だけをこの座標系に置く。 PASS、CONTINUE、DECLINE は対応する二値
 * 二択の判定の基準側であり、独立した学習スコアを持たない。具体的な pattern、赤牌種類、打牌の識別情報は行動候補スコア で正規化する。
 *
 * <p>{@link #networkIndex()} はネットワークテンソルとチェックポイントに保存される明示的な列インデックスである。 列挙型 の宣言順を暗黙の永続形式として使わない。
 */
public enum DecisionAlternative {
  /** RON対DECLINE_RON 二択の判定のRON側。 */
  RON(0, Action.Type.RON_AGARI),

  /** TSUMO対DECLINE 二択の判定のTSUMO側。 */
  TSUMO(1, Action.Type.TSUMO_AGARI),

  /** 九種九牌対CONTINUE 二択の判定の九種九牌側。 */
  KYUSHU(2, Action.Type.KYUSHU_KYUHAI),

  /** MELD対PASS 二択の判定のMELD 分岐全体。 */
  CALL(3, null),

  /** MELD 種類節点のCHI側。 */
  CHI(4, Action.Type.CHI),

  /** MELD 種類節点のPON側。 */
  PON(5, Action.Type.PON),

  /** MELD 種類節点のDAIMINKAN側。 */
  DAIMINKAN(6, Action.Type.DAIMINKAN),

  /** KAN対CONTINUE 二択の判定のKAN 分岐全体。 */
  KAN(7, null),

  /** KAN 種類節点のANKAN側。 */
  ANKAN(8, Action.Type.ANKAN),

  /** KAN 種類節点のKAKAN側。 */
  KAKAN(9, Action.Type.KAKAN);

  /** 選択肢スコア計算処理テンソルの固定列数。 */
  public static final int NETWORK_SIZE = values().length;

  private static final Action.Type[] MELD_TYPES = {
    Action.Type.CHI, Action.Type.PON, Action.Type.DAIMINKAN
  };
  private static final Action.Type[] KAN_TYPES = {Action.Type.ANKAN, Action.Type.KAKAN};

  static {
    Set<Integer> indexes = new HashSet<>();
    for (DecisionAlternative alternative : values()) {
      if (alternative.networkIndex < 0 || alternative.networkIndex >= NETWORK_SIZE) {
        throw new ExceptionInInitializerError("invalid alternative index: " + alternative);
      }
      if (!indexes.add(alternative.networkIndex)) {
        throw new ExceptionInInitializerError(
            "duplicate alternative index: " + alternative.networkIndex);
      }
    }
  }

  private final int networkIndex;
  private final Action.Type actionType;

  DecisionAlternative(int networkIndex, Action.Type actionType) {
    this.networkIndex = networkIndex;
    this.actionType = actionType;
  }

  /**
   * スコア計算処理テンソルと分岐オフセットで使う固定列インデックスを返す。
   *
   * @return 分岐スコアの列インデックス
   */
  public int networkIndex() {
    return networkIndex;
  }

  static Action.Type[] meldTypes() {
    return MELD_TYPES;
  }

  static Action.Type[] kanTypes() {
    return KAN_TYPES;
  }

  /**
   * 実行動候補を再利用領域して採点する選択肢かを返す。
   *
   * @return 単一の {@link Action.Type} に対応するなら {@code true}
   */
  public boolean hasActionType() {
    return actionType != null;
  }

  /**
   * 再利用領域対象となる実行動種別を返す。
   *
   * @return 再利用領域対象行動種別
   * @throws IllegalStateException CALL/KAN のような分岐-要約二択の判定の場合
   */
  public Action.Type actionType() {
    if (actionType == null) {
      throw new IllegalStateException("branch gate has no single action type: " + this);
    }
    return actionType;
  }

  /**
   * 実行動種類を採点する選択肢を返す。
   *
   * @param type 終端または条件付き種類節点に属する行動種類
   * @return 種類に対応する選択肢
   */
  static DecisionAlternative forActionType(Action.Type type) {
    return switch (type) {
      case RON_AGARI -> RON;
      case TSUMO_AGARI -> TSUMO;
      case KYUSHU_KYUHAI -> KYUSHU;
      case CHI -> CHI;
      case PON -> PON;
      case DAIMINKAN -> DAIMINKAN;
      case ANKAN -> ANKAN;
      case KAKAN -> KAKAN;
      default -> throw new IllegalArgumentException("action type has no alternative: " + type);
    };
  }
}
