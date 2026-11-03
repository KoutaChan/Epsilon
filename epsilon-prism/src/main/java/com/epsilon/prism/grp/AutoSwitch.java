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
 * grpSwitch = AUTO の切り替え（学習の錠を持って呼ぶ）: 取り置き h で作った候補は反復 h + grpSwitchLag で効く。その反復が始まる前に READY なら
 * phi.jsonl に effectiveFrom = h + L の項目を足し（{@link PhiHistory#append}）、そうでなければ refits.jsonl に SKIPPED_NOT_READY を書いて遅れて
 * 切り替えはしない（同じ設定なら同じ Φ の歴史になる）。候補の学習の始めにした項目（parent）が phi.jsonl の最後の項目でなくなった候補（巻き戻しや
 * 手動の切り替えの後）と、最後の項目より前に効くはずだった候補も SKIPPED_NOT_READY にする。
 */
public final class AutoSwitch {
  private AutoSwitch() {}

  /**
   * 候補 1 つの扱い。
   *
   * @param record 候補の READY の行
   * @param effectiveFrom 効く反復 h + L
   * @param skip 入れない理由（入れるなら null）
   */
  record Step(RefitRecord record, int effectiveFrom, String skip) {}

  /**
   * 効く時期の来た候補を入れる。
   *
   * @return 足した項目（なければ空）
   * @throws IOException 記録が読めないか書けないとき
   */
  public static Optional<PhiEntry> apply(GrpFiles files, PrismConfig config) throws IOException {
    if (config.grpSwitch() != GrpSwitch.AUTO) return Optional.empty();
    List<PhiEntry> history = new ArrayList<>(PhiHistory.read(files));
    Optional<PhiEntry> added = Optional.empty();
    for (RefitRecord candidate : pending(RefitRecord.readAll(files), history)) {
      Step step = step(candidate, history, config.grpSwitchLag());
      String skip = step.skip();
      if (skip == null) {
        try {
          PhiEntry entry =
              PhiHistory.append(
                  files,
                  candidate.entry(
                      step.effectiveFrom(),
                      "AUTO: " + candidate.id() + " at holdout + " + config.grpSwitchLag()));
          history.add(entry);
          added = Optional.of(entry);
          continue;
        } catch (IllegalArgumentException late) {
          skip = "iteration " + step.effectiveFrom() + " has started: " + late.getMessage();
        }
      }
      skipped(candidate, skip).append(files);
    }
    return added;
  }

  /** まだ phi.jsonl に入っていない READY の候補（取り置きの順。同じ id の行は最後の行）。 */
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

  /** 候補 1 つを入れるか（phi.jsonl の始まった反復の確かめは {@link PhiHistory#append}）。 */
  static Step step(RefitRecord candidate, List<PhiEntry> history, int lag) {
    int effectiveFrom = candidate.window().holdout() + lag;
    if (history.isEmpty()) return new Step(candidate, effectiveFrom, "phi.jsonl has no entries");
    PhiEntry last = history.getLast();
    if (!candidate.parent().equals(last.id()))
      return new Step(
          candidate,
          effectiveFrom,
          "trained against " + candidate.parent() + ", but the last entry is " + last.id());
    if (effectiveFrom <= last.effectiveFrom())
      return new Step(
          candidate,
          effectiveFrom,
          "iteration " + effectiveFrom + " is not after " + last.id() + " (" + last.effectiveFrom() + ")");
    return new Step(candidate, effectiveFrom, null);
  }

  /** 候補を入れなかった行（候補のパスと sha は READY の行だけが持つ）。 */
  static RefitRecord skipped(RefitRecord candidate, String detail) {
    return new RefitRecord(
        candidate.id(),
        RefitStatus.SKIPPED_NOT_READY,
        candidate.cause(),
        candidate.window(),
        candidate.init(),
        candidate.parent(),
        "",
        "",
        candidate.gate(),
        detail,
        candidate.seconds());
  }
}
