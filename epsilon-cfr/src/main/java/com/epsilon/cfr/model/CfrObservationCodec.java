package com.epsilon.cfr.model;

import com.epsilon.cfr.collection.DecisionContext;
import com.epsilon.cfr.data.ObservationPayload;
import com.epsilon.nano.ai.decision.data.DecisionInputRowCodec;
import com.epsilon.nano.ai.decision.data.EpsilonDecisionBinaryArrayCodec;
import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.util.List;

/** Nanoと同じ盤面・合法手の入力を保存し、変更不可の観測として推論と学習で共有する。 */
public final class CfrObservationCodec {
  private static final int MAGIC = 0x43465233;
  private static final ThreadLocal<EpsilonDecisionBinaryArrayCodec.Scratch> SCRATCH =
      ThreadLocal.withInitial(EpsilonDecisionBinaryArrayCodec.Scratch::new);

  private CfrObservationCodec() {}

  public record Decoded(DecisionHostBatch host) {}

  /** 推論から保存まで共有できる不変の観測を作る。保存先を直接所有し、中間のバイト列を複製しない。 */
  public static ObservationPayload encodePayload(DecisionContext context) {
    return encodePayload(build(context));
  }

  static Decoded build(DecisionContext context) {
    return build(context, new DecisionBatchBuilder.EncodingWorkspacePool(1));
  }

  static Decoded build(
      DecisionContext context, DecisionBatchBuilder.EncodingWorkspacePool workspace) {
    var builder =
        DecisionBatchBuilder.inference(
            1, DecisionBatchBuilder.selectInferenceBucket(context.legalActions()), workspace);
    builder.encodeObservedInferenceRow(0, context.observation(), context.legalActions());
    // 確定した入力配列は構築結果が所有し、再利用する作業領域には一時バッファだけを残す。
    return new Decoded(builder.buildEncodedInferenceRows());
  }

  static ObservationPayload encodePayload(Decoded row) {
    return encodePayload(row.host(), 0);
  }

  /** 確定済みのバッチから一行だけ保存する。入力配列自体は複製しない。 */
  public static ObservationPayload encodePayload(DecisionHostBatch host, int row) {
    return ObservationPayload.fromDecisionInput(
        MAGIC, DecisionInputRowCodec.encode(host.sliceRows(row, 1)));
  }

  /** 保存データを複製せずに読み取り、復号した入力は返却値が独立して所有する。 */
  public static Decoded decode(ObservationPayload encoded) {
    return decode(encoded.openStream());
  }

  private static Decoded decode(InputStream source) {
    try (var in = new DataInputStream(source)) {
      if (in.readInt() != MAGIC) throw new IOException("CFR observation schema mismatch");
      DecisionHostBatch host = DecisionInputRowCodec.read(in, SCRATCH.get());
      if (in.read() != -1) throw new IOException("Unexpected trailing CFR observation data");
      return new Decoded(host);
    } catch (IOException invalid) {
      throw new UncheckedIOException("Invalid CFR observation", invalid);
    }
  }

  /** 互換性のある容量区分の入力を、行順序を保って読み取り専用のバッチにまとめる。 */
  static DecisionHostBatch batch(List<Decoded> rows) {
    if (rows.size() == 1) return rows.getFirst().host();
    return DecisionHostBatch.concatenate(rows.stream().map(Decoded::host).toList());
  }
}
