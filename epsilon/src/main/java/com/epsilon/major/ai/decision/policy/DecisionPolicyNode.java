package com.epsilon.major.ai.decision.policy;

import com.epsilon.core.Action;
import java.util.HashSet;
import java.util.Set;

/**
 * 麻雀方策グラフを構成する条件付き意思決定節点。
 *
 * <p>上位二択分岐と下位の条件付き選択を別節点として表す。とくにCALLとKANは、合法な型の数を直接softmaxへ
 * 入れず、基準分岐と集約済み分岐を比較する二択分岐である。ネットワークインデックスは節点埋め込みと チェックポイントで使う明示的な座標で、enum宣言順には依存しない。
 */
public enum DecisionPolicyNode {
  /** RONを受理するかDECLINE_RONへ進むか。 */
  RON_GATE(0),
  /** TSUMOを受理するか続行するか。 */
  TSUMO_GATE(1),
  /** 九種九牌で流局するか続行するか。 */
  KYUSHU_GATE(2),
  /** PASSとMELDを比較する応答二択分岐。 */
  CALL_GATE(3),
  /** MELDを選んだ条件下でCHI・PON・DAIMINKANを比較する節点。 */
  MELD_TYPE(4),
  /** 同じMELD 種類内の形・赤牌種類を比較する節点。 */
  MELD_CANDIDATE(5),
  /** CONTINUEとKANを比較する手番二択分岐。 */
  KAN_GATE(6),
  /** KANを選んだ条件下でANKAN・KAKANを比較する節点。 */
  KAN_TYPE(7),
  /** 物理的に異なる打牌識別情報を比較する節点。 */
  DISCARD_IDENTITY(8),
  /** 同じ物理打牌に条件付けてDAMA・RIICHIを比較する節点。 */
  RIICHI_GATE(9);

  /** ネットワーク上に確保する方策節点埋め込み数。 */
  public static final int NETWORK_SIZE = values().length;

  private static final Action.Type[] MELD_TYPES = {
    Action.Type.CHI, Action.Type.PON, Action.Type.DAIMINKAN
  };
  private static final Action.Type[] KAN_TYPES = {Action.Type.ANKAN, Action.Type.KAKAN};

  static {
    Set<Integer> indexes = new HashSet<>();
    for (DecisionPolicyNode node : values()) {
      if (node.networkIndex < 0 || node.networkIndex >= NETWORK_SIZE) {
        throw new ExceptionInInitializerError("invalid Policy node index: " + node);
      }
      if (!indexes.add(node.networkIndex)) {
        throw new ExceptionInInitializerError("duplicate Policy node index: " + node.networkIndex);
      }
    }
  }

  private final int networkIndex;

  DecisionPolicyNode(int networkIndex) {
    this.networkIndex = networkIndex;
  }

  /**
   * 節点埋め込みと保存されたテンソルで使う列インデックスを返す。
   *
   * @return enum宣言順に依存しない安定インデックス
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
}
