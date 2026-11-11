package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.prism.TableWalk;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/** 方策のテストで使う要求を作るヘルパー。エンジンで打った判断の記録 O から、retain の要求を作る。 */
public final class PolicyRequests {

  private PolicyRequests() {}

  /**
   * 乱数の種 {@code seed} の半荘を 1 つ打ち、合法手が 2 つ以上ある判断の O を {@code limit} 個まで集める。
   *
   * @return O のワードの写し（判断の順）
   */
  public static List<long[]> observations(long seed, TableWalk.Policy policy, int limit) {
    ObservationWriter writer = new ObservationWriter(TENHOU_4P);
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    List<long[]> rows = new ArrayList<>();
    new TableWalk(TENHOU_4P, policy)
        .run(
            seed,
            1,
            (engine, slot) -> {
              Decisions decisions = engine.decisions();
              if (rows.size() >= limit || decisions.legalCount(slot) < 2) {
                return;
              }
              batch.clear();
              int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
              int start = batch.start(index);
              rows.add(Arrays.copyOfRange(batch.words(), start, start + batch.longs(index)));
            });
    return rows;
  }

  /** O のワードの列を全て 1 つのバッチに入れ、行の順に retain の要求を作る（答えのフィールドは空）。 */
  public static List<PolicyRequest> requests(List<long[]> rows) {
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    Observation reader = new Observation(TENHOU_4P);
    List<PolicyRequest> requests = new ArrayList<>(rows.size());
    for (long[] words : rows) {
      int index = batch.add(reader.bind(words, 0));
      requests.add(new PolicyRequest(null, 0, index, null, null, true, 0, batch, index, reader.bind(batch, index)));
    }
    return requests;
  }

  /** {@code request} と同じ O を指す retain の要求を新しく作る（答えのフィールドは空）。 */
  public static PolicyRequest copy(PolicyRequest request) {
    Observation reader = new Observation(TENHOU_4P);
    return new PolicyRequest(
        null, 0, request.record(), null, null, true, 0, request.records(), request.record(), request.bind(reader));
  }
}
