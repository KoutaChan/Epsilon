package com.epsilon.prism.rows;

import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionBinaryArrayCodec;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.UncheckedIOException;

/** Nanoと同じ盤面・合法手の入力を保存し、変更不可の観測として推論と学習で共有する。 */
public final class ObservationCodec {
  /** 行の観測の先頭の識別子（"PRS1"）。行の形式の一部なので変えない。 */
  private static final int MAGIC = 0x50525331;
  private static final ThreadLocal<EpsilonDecisionBinaryArrayCodec.Scratch> SCRATCH =
      ThreadLocal.withInitial(EpsilonDecisionBinaryArrayCodec.Scratch::new);

  private ObservationCodec() {}

  /** 確定済みのバッチから一行だけ保存する。入力配列自体は複製しない。 */
  public static ObservationPayload encodePayload(DecisionHostBatch host, int row) {
    return ObservationPayload.fromDecisionInput(
        MAGIC, DecisionInputRowCodec.encode(host.sliceRows(row, 1)));
  }

  /** 保存データを複製せずに読み取り、復号した入力は返却値が独立して所有する。 */
  public static DecisionHostBatch decode(ObservationPayload encoded) {
    try (var in = new DataInputStream(encoded.openStream())) {
      if (in.readInt() != MAGIC) throw new IOException("Observation schema mismatch");
      DecisionHostBatch host = DecisionInputRowCodec.read(in, SCRATCH.get());
      if (in.read() != -1) throw new IOException("Unexpected trailing observation data");
      return host;
    } catch (IOException invalid) {
      throw new UncheckedIOException("Invalid observation", invalid);
    }
  }
}
