package com.epsilon.ai.decision;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;
import java.nio.FloatBuffer;

/**
 * 一判断の終了ゲートに使う分岐比較教師値。
 *
 * <p>未完成の九種九牌では比較損失を与えず、通常PPOからゲートへの勾配も停止する。
 *
 * @param code 対象ゲート。0は通常PPO、1はロン、2はツモ、3は九種九牌
 * @param oldAcceptanceProbability 収集時点のゲートの宣言確率
 * @param acceptedUtility 宣言側の局境界効用
 * @param declinedUtility 続行側の局境界効用
 * @param complete 両枝の効用が確定しているか
 */
public record DecisionBranchTarget(
    int code,
    float oldAcceptanceProbability,
    float acceptedUtility,
    float declinedUtility,
    boolean complete) {
  public static final int WIDTH = 5;
  public static final int BYTES = Integer.BYTES + 3 * Float.BYTES + Byte.BYTES;
  public static final DecisionBranchTarget NONE = new DecisionBranchTarget(0, 0, 0, 0, false);
  public static final DecisionBranchTarget KYUSHU_ONLY =
      new DecisionBranchTarget(3, 0, 0, 0, false);

  public DecisionBranchTarget {
    if (code < 0
        || code > 3
        || !Float.isFinite(oldAcceptanceProbability)
        || !Float.isFinite(acceptedUtility)
        || !Float.isFinite(declinedUtility)
        || (complete
            && (code == 0 || oldAcceptanceProbability <= 0 || oldAcceptanceProbability >= 1))
        || (!complete && code != 0 && code != 3)) {
      throw new IllegalArgumentException("Invalid branch target");
    }
  }

  public static DecisionBranchTarget completed(
      DecisionBranchGate gate, float probability, float accepted, float declined) {
    return new DecisionBranchTarget(gate.scoreIndex() + 1, probability, accepted, declined, true);
  }

  public void writeTo(float[] values, int offset) {
    values[offset] = code;
    values[offset + 1] = oldAcceptanceProbability;
    values[offset + 2] = acceptedUtility;
    values[offset + 3] = declinedUtility;
    values[offset + 4] = complete ? 1 : 0;
  }

  public void writeTo(DataOutput out) throws IOException {
    out.writeInt(code);
    out.writeFloat(oldAcceptanceProbability);
    out.writeFloat(acceptedUtility);
    out.writeFloat(declinedUtility);
    out.writeBoolean(complete);
  }

  public void writeTo(FloatBuffer values, int offset) {
    values.put(offset, code);
    values.put(offset + 1, oldAcceptanceProbability);
    values.put(offset + 2, acceptedUtility);
    values.put(offset + 3, declinedUtility);
    values.put(offset + 4, complete ? 1 : 0);
  }

  public static DecisionBranchTarget readFrom(DataInput in) throws IOException {
    try {
      return new DecisionBranchTarget(
          in.readInt(), in.readFloat(), in.readFloat(), in.readFloat(), in.readBoolean());
    } catch (IllegalArgumentException invalid) {
      throw new IOException("Invalid branch target", invalid);
    }
  }
}
