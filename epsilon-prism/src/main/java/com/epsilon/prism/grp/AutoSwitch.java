package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * grpSwitch = AUTO の切り替え（学習の JVM が反復 t の始め、Φ を決める前に呼ぶ）: READY でまだ phi.jsonl に入っていない候補のうち、学習の
 * 始めにした項目（parent）が phi.jsonl の最後の項目のものの一番新しい候補（取り置きが最も後、同じなら後の試行）を effectiveFrom = t で入れる
 * （{@link PhiHistory#appendAtIterationStart}）。parent が最後の項目でない候補（巻き戻しや手動の切り替えの後）と、入れた候補より古い候補は
 * refits.jsonl に STALE を書いて入れない。最後の項目が t より後に効く（巻き戻しの行が先の反復に書かれている）ときは何も入れず、候補は次の
 * 反復に残す。
 */
public final class AutoSwitch {
  private AutoSwitch() {}

  /**
   * 反復 t の始めに候補を入れる。
   *
   * @return 足した項目（なければ空）
   * @throws IOException 記録が読めないか書けないとき
   */
  public static Optional<PhiEntry> apply(GrpFiles files, PrismConfig config, int t) throws IOException {
    if (config.grpSwitch() != GrpSwitch.AUTO) return Optional.empty();
    List<PhiEntry> history = PhiHistory.read(files);
    if (history.isEmpty() || history.getLast().effectiveFrom() > t) return Optional.empty();
    String last = history.getLast().id();
    List<RefitRecord> pending = pending(RefitRecord.readAll(files), history);
    RefitRecord chosen = null;
    for (RefitRecord candidate : pending)
      if (candidate.parent().equals(last)) chosen = candidate;
    for (RefitRecord candidate : pending) {
      if (candidate == chosen) continue;
      String why =
          candidate.parent().equals(last)
              ? "superseded by " + chosen.id()
              : "trained against " + candidate.parent() + ", but the last entry is " + last;
      candidate.withStatus(RefitStatus.STALE, why).append(files);
    }
    if (chosen == null) return Optional.empty();
    PhiEntry entry = chosen.entry(t, "AUTO: " + chosen.id() + " at iteration " + t);
    PhiHistory.appendAtIterationStart(files, entry);
    return Optional.of(entry);
  }

  /** まだ phi.jsonl に入っていない READY の候補（取り置きの順、同じ取り置きは書いた順。同じ id の行は最後の行）。 */
  static List<RefitRecord> pending(List<RefitRecord> records, List<PhiEntry> history) {
    Map<String, RefitRecord> latest = new LinkedHashMap<>();
    for (RefitRecord record : records) latest.put(record.id(), record);
    List<RefitRecord> ready = new ArrayList<>();
    for (RefitRecord record : latest.values())
      if (record.status() == RefitStatus.READY
          && history.stream().noneMatch(entry -> entry.id().equals(record.id()))) ready.add(record);
    ready.sort(Comparator.comparingInt(record -> record.window().holdout()));
    return ready;
  }
}
