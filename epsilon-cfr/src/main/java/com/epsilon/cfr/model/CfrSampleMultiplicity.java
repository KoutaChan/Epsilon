package com.epsilon.cfr.model;

import com.epsilon.cfr.data.CfrSampleId;
import com.epsilon.cfr.data.CfrTrainingRow;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;

/** 一回の抽出結果を、同じ行の出現回数を保って集約する。リプレイの抽出重みは重ねて掛けない。 */
final class CfrSampleMultiplicity {
  private CfrSampleMultiplicity() {}

  record Entry(CfrTrainingRow row, int count) {}

  static List<Entry> aggregate(List<CfrTrainingRow> rows) {
    var unique = new LinkedHashMap<CfrSampleId, Counted>();
    for (CfrTrainingRow row : rows) {
      Counted found = unique.get(row.id());
      if (found == null) unique.put(row.id(), new Counted(row));
      else {
        if (found.row != row && !found.row.contentEquals(row))
          throw new IllegalArgumentException("Conflicting CFR sample identity: " + row.id());
        found.count++;
      }
    }
    List<Entry> result = new ArrayList<>(unique.size());
    for (Counted value : unique.values()) result.add(new Entry(value.row, value.count));
    return result;
  }

  private static final class Counted {
    final CfrTrainingRow row;
    int count = 1;

    Counted(CfrTrainingRow row) {
      this.row = row;
    }
  }
}
