package com.epsilon.prism.backend;

import com.epsilon.prism.TableWalk;
import com.epsilon.prism.input.HostBatch;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 学習済みの三人打ちのバージョン（-Dprism.snapshot）の s が BF16 の丸めでどれだけ揺れるかを、CPU で測る診断（{@link PrecisionSensitivity}）。
 * プロパティが無ければ飛ばす。行は、{@code GREEDY} と {@code CALLS} の打ち方で打った三人打ちの半荘の判断。
 */
public class ThreePlayerPrecisionSensitivityCheck {

  @Test
  public void measure() throws IOException {
    Path snapshot = PrecisionSensitivity.snapshot();
    List<long[]> rows = new ArrayList<>();
    rows.addAll(ThreePlayerRows.observations(31, 6, TableWalk.Policy.GREEDY, PrecisionSensitivity.ROWS / 2));
    rows.addAll(ThreePlayerRows.observations(37, 6, TableWalk.Policy.CALLS, PrecisionSensitivity.ROWS / 2));
    List<HostBatch> batches = new ArrayList<>();
    for (int from = 0; from < rows.size(); from += PrecisionSensitivity.BATCH_ROWS) {
      batches.add(
          ThreePlayerRows.encode(rows.subList(from, Math.min(rows.size(), from + PrecisionSensitivity.BATCH_ROWS))));
    }
    PrecisionSensitivity.measure(snapshot, batches);
  }
}
