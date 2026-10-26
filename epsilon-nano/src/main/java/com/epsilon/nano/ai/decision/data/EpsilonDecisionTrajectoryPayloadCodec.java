package com.epsilon.nano.ai.decision.data;

import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionTrainingSlabWriter;
import java.io.DataInputStream;
import java.io.DataOutputStream;
import java.io.IOException;
import java.nio.FloatBuffer;

/** 収集中の Decision 入力を1行ずつ保存・復元する。過去の保存形式は受理しない。 */
final class EpsilonDecisionTrajectoryPayloadCodec {

  static final int MAGIC = 0xED20_001B;

  private EpsilonDecisionTrajectoryPayloadCodec() {}

  static void write(
      DataOutputStream out,
      EpsilonDecisionTrajectoryPayload payload,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch)
      throws IOException {
    out.writeInt(MAGIC);
    DecisionInputRowCodec.write(out, payload.input(), scratch);
    EpsilonDecisionBinaryArrayCodec.writeFloatArray(out, payload.behaviorPolicy(), scratch);
    EpsilonDecisionBinaryArrayCodec.writeFloatArray(out, payload.rolloutPolicy(), scratch);
  }

  static EpsilonDecisionTrajectoryPayload read(
      DataInputStream in, EpsilonDecisionBinaryArrayCodec.Scratch scratch) throws IOException {
    requireMagic(in);
    DecisionHostBatch input = DecisionInputRowCodec.read(in, scratch);
    float[] behavior = EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch);
    float[] rollout = EpsilonDecisionBinaryArrayCodec.readFloatArray(in, scratch);
    return new EpsilonDecisionTrajectoryPayload(input.sliceRows(0, 1), behavior, rollout);
  }

  /** データ本体を一行サンプルへ戻さず、確定済みバッチの最終プリミティブ型連続バッファへ直接復号する。 */
  static void readTrainingRow(
      DataInputStream in,
      EpsilonDecisionBinaryArrayCodec.Scratch scratch,
      EpsilonDecisionTrainingSampleDescriptor descriptor,
      DecisionTrainingSlabWriter destination,
      int row,
      float actorWeight,
      float sampleWeight)
      throws IOException {
    requireMagic(in);
    DecisionBucket bucket = DecisionInputRowCodec.readHeader(in);
    if (!bucket.equals(descriptor.bucket()) || !bucket.equals(destination.bucket())) {
      throw new IOException(
          "Decision trajectory payload bucket mismatch: payload="
              + bucket
              + " descriptor="
              + descriptor.bucket()
              + " destination="
              + destination.bucket());
    }
    DecisionInputRowCodec.readInputs(in, bucket, scratch, destination, row);
    requireDescriptorInputIdentity(descriptor, destination, row);

    int legalActionCount = descriptor.legalActionCount();
    FloatBuffer behavior =
        EpsilonDecisionBinaryArrayCodec.readFloatBuffer(in, legalActionCount, scratch);
    float selectedBehavior = behavior.get(descriptor.chosenLegalSlot());
    if (Float.floatToIntBits(selectedBehavior) != Float.floatToIntBits(descriptor.behaviorProb())) {
      throw new IOException(
          "Decision behavior probability differs between payload and descriptor: payload="
              + selectedBehavior
              + " descriptor="
              + descriptor.behaviorProb());
    }
    destination.writeBehaviorPolicy(row, legalActionCount, behavior, true);

    FloatBuffer rollout =
        EpsilonDecisionBinaryArrayCodec.readFloatBuffer(in, legalActionCount, scratch);
    destination.writeRemainingTargetRow(
        row,
        legalActionCount,
        descriptor.chosenLegalSlot(),
        rollout,
        descriptor.valueTarget(),
        descriptor.advantage(),
        actorWeight,
        sampleWeight);
    destination.writeBranchTarget(row, descriptor.branchTarget());
  }

  private static void requireMagic(DataInputStream in) throws IOException {
    int magic = in.readInt();
    if (magic != MAGIC) {
      throw new IOException(
          "Unsupported Decision trajectory payload magic: 0x" + Integer.toHexString(magic));
    }
  }

  private static void requireDescriptorInputIdentity(
      EpsilonDecisionTrainingSampleDescriptor descriptor,
      DecisionTrainingSlabWriter destination,
      int row)
      throws IOException {
    if (destination.legalActionCount(row) != descriptor.legalActionCount()) {
      throw new IOException("Decision legal-action count differs between payload and descriptor");
    }
    for (int slot = 0; slot < descriptor.legalActionCount(); slot++) {
      if (destination.legalActionId(row, slot) != descriptor.legalActionIdBySlot(slot)) {
        throw new IOException(
            "Decision legal action differs between payload and descriptor at slot " + slot);
      }
    }
    if (destination.playerSeat(row) != descriptor.playerSeat()
        || destination.sourcePlayerRelativeSeat(row) != descriptor.sourcePlayerRelativeSeat()
        || destination.currentPlayerRelativeSeat(row) != descriptor.currentPlayerRelativeSeat()) {
      throw new IOException("Decision seat identity differs between payload and descriptor");
    }
  }
}
