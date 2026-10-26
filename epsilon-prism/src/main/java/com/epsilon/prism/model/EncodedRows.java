package com.epsilon.prism.model;

import com.epsilon.nano.ai.decision.input.DecisionBucket;
import com.epsilon.nano.ai.decision.input.DecisionHostBatch;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/**
 * CPU で符号化した入力。容量区分ごとの入力と、各行が元の並びのどこに当たるか。デバイスに依存しないので、同じ入力を何本ものネットに通せる。
 */
public record EncodedRows(List<Group> groups, int size) {
  /** 1つの容量区分。indices[row] は元の並びの位置。 */
  public record Group(DecisionHostBatch host, int[] indices) {}

  /** 1行ずつの入力（復号した payload）を容量区分ごとにまとめる。 */
  public static EncodedRows of(List<DecisionHostBatch> rows) {
    var buckets = new LinkedHashMap<DecisionBucket, List<Integer>>();
    for (int i = 0; i < rows.size(); i++)
      buckets.computeIfAbsent(rows.get(i).bucket(), ignored -> new ArrayList<>()).add(i);
    List<Group> groups = new ArrayList<>(buckets.size());
    for (List<Integer> members : buckets.values()) {
      DecisionHostBatch host =
          members.size() == 1
              ? rows.get(members.getFirst())
              : DecisionHostBatch.concatenate(members.stream().map(rows::get).toList());
      groups.add(new Group(host, members.stream().mapToInt(Integer::intValue).toArray()));
    }
    return new EncodedRows(groups, rows.size());
  }
}
