package com.epsilon.prism.policy;

import com.epsilon.nano.ai.decision.input.DecisionBatchBuilder;
import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import com.epsilon.nano.ai.decision.input.DecisionInferenceStoragePool;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.model.EncodedRows;
import com.epsilon.prism.rows.ObservationCodec;
import com.epsilon.spi.PolicyExecutionContext;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.concurrent.CompletableFuture;

/**
 * 問い合わせの符号化と、s から σ への仕上げ。符号化は Nano の CPU エンコーダーで行う。規則事前 p0 は retain の問い合わせ（ラベル行か根の記録を書く判断）
 * と、s の最大が同点の問い合わせ（貪欲の同点を p0 で割る。白紙の網では全ての問い合わせ。貪欲は argmax p0）にだけ作る。同点のない問い合わせの貪欲は p0 に
 * 依らないので、完了のスレッドの仕事を増やさない。保存用の payload は retain の問い合わせにだけ作る。p0 の特徴（打牌後の向聴・受け入れ）はエンコーダーが
 * 既に書いた遷移特徴から読むので、追加の解析をしない。
 */
final class RequestEncoder {
  private final DecisionBatchBuilder.EncodingWorkspacePool workspaces =
      new DecisionBatchBuilder.EncodingWorkspacePool(1);

  CompletableFuture<EncodedRows> encode(
      List<PolicyRequest> rows,
      PolicyExecutionContext execution,
      DecisionInferenceStoragePool.Lease storage) {
    var grouped = new LinkedHashMap<DecisionBucket, List<Integer>>();
    for (int index = 0; index < rows.size(); index++)
      grouped
          .computeIfAbsent(
              DecisionBatchBuilder.selectInferenceBucket(rows.get(index).legal()),
              ignored -> new ArrayList<>())
          .add(index);
    List<DecisionBatchBuilder> builders = new ArrayList<>(grouped.size());
    List<int[]> indices = new ArrayList<>(grouped.size());
    grouped.forEach(
        (bucket, members) -> {
          builders.add(
              storage == null
                  ? DecisionBatchBuilder.inference(members.size(), bucket, workspaces)
                  : storage.inference(members.size(), bucket, workspaces));
          indices.add(members.stream().mapToInt(Integer::intValue).toArray());
        });
    PolicyExecutionContext.RangeEncoder range =
        (from, to) -> {
          int offset = 0;
          for (int group = 0; group < builders.size(); group++) {
            int[] members = indices.get(group);
            int first = Math.max(0, from - offset), end = Math.min(members.length, to - offset);
            if (first < end) {
              try (var session = builders.get(group).openEncoding()) {
                for (int row = first; row < end; row++) {
                  PolicyRequest request = rows.get(members[row]);
                  session.encodeObservedInferenceRow(row, request.observation(), request.legal());
                }
              }
            }
            offset += members.length;
          }
        };
    CompletableFuture<Void> encoded =
        execution == null
            ? CompletableFuture.runAsync(() -> range.encode(0, rows.size()), Runnable::run)
            : execution.encodeRowsAsync(rows.size(), range);
    return encoded.thenApply(
        ignored -> {
          List<EncodedRows.Group> groups = new ArrayList<>(builders.size());
          for (int group = 0; group < builders.size(); group++)
            groups.add(
                new EncodedRows.Group(
                    builders.get(group).buildEncodedInferenceRows(), indices.get(group)));
          return new EncodedRows(groups, rows.size());
        });
  }

  /**
   * s・σ と、retain か s の最大が同点なら p0、retain なら payload を書く。入力領域を返す前に呼ぶ。
   *
   * @param scores scores[i] は要求 i の合法手ごとの s(o, a)（FP32）
   * @param prior p0 の係数
   */
  static void finish(
      List<PolicyRequest> rows, EncodedRows encoded, float[][] scores, PriorSettings prior) {
    for (EncodedRows.Group group : encoded.groups()) {
      DecisionHostBatch host = group.host();
      for (int row = 0; row < host.size(); row++) {
        int index = group.indices()[row];
        PolicyRequest request = rows.get(index);
        int n = request.legal().size();
        request.scores = Arrays.copyOf(scores[index], n);
        if (request.retain() || topTied(request.scores, n))
          request.prior = Prior.compute(host, row, request.legal(), prior);
        // p0 がない（同点のない）問い合わせは番号で割っても同じ手になる
        request.sigma = greedy(request.scores, request.prior, n);
        if (request.retain()) request.payload = ObservationCodec.encodePayload(host, row);
      }
    }
  }

  /**
   * 貪欲の one-hot（{@link Greedy#argmax}: s の最大、同点は p0 の大きい手、それも同じなら番号の小さい手）。打つ方策は常に貪欲で、床 ε_b は行動方策が
   * 混ぜる。
   */
  static float[] greedy(float[] scores, float[] prior, int n) {
    float[] sigma = new float[n];
    sigma[Greedy.argmax(scores, prior, n)] = 1;
    return sigma;
  }

  /** 先頭 n 手の s の最大が 2 手以上で同じか。 */
  static boolean topTied(float[] scores, int n) {
    int best = 0;
    boolean tied = false;
    for (int a = 1; a < n; a++) {
      if (scores[a] > scores[best]) {
        best = a;
        tied = false;
      } else if (scores[a] == scores[best]) {
        tied = true;
      }
    }
    return tied;
  }
}
