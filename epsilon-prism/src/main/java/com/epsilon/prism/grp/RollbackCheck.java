package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.run.PrismConfig;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonParseException;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Optional;
import java.util.Set;
import java.util.function.IntPredicate;

/**
 * 切り替えの後の巻き戻しの点検: 新しい項目で集めたコホート {@value #COHORTS} つで、前の項目が対の MSE で 2 SE より良ければ巻き戻しを提案する。
 * grpSwitch = AUTO なら phi.jsonl に前の項目へ戻す行を足し、PROPOSE なら報告だけ。x1n3 による巻き戻しは常にユーザーの判断で、ここでは扱わない。
 *
 * <p>点検するのは、最新の封印済みのコホート t に効く項目 E が効き始めてから {@value #COHORTS} コホートの対局の記録がそろい、前の項目 P がある
 * とき。E が効き始めた反復は、同じ Φ の続きの行（境目の点検で拒んだ候補の代わりに E を効かせ直した行）をさかのぼった最初の行で決める
 * （続きの行を新しい切り替えとみなすと E を E 自身と比べてしまう）。項目ごとに 1 回で、結果は grp/rollback.jsonl に書く（巻き戻せなかった
 * ときも）。巻き戻しの行（id は P の id のまま＝grp-switch の巻き戻しと同じ。新しい id にすると学習の JVM が初めて効く候補とみなし、
 * refits.jsonl に READY がないので拒んでしまう。理由は rollback: で始まる）は点検しない（行ったり来たりしない）。巻き戻しの行は、学習の JVM が
 * 次に始めうる反復（state.json の次のコホート + 1）から効く。
 */
public final class RollbackCheck {

  /** 点検に使う新しい項目のコホートの数。 */
  public static final int COHORTS = 2;

  /** 巻き戻しの線（前の項目の対の MSE が今の項目より SE の何倍良いか）。 */
  static final double STANDARD_ERRORS = 2;

  private static final Gson GSON = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private RollbackCheck() {}

  /**
   * 点検の結果。
   *
   * @param current 点検した項目の id
   * @param previous 前の項目の id
   * @param paired 対の MSE の差（前 − 今）と SE
   * @param rollback 巻き戻すべきなら true
   * @param applied phi.jsonl に巻き戻しの行を足したなら true
   */
  public record Verdict(
      String current, String previous, CellStats.Paired paired, boolean rollback, boolean applied) {}

  /**
   * 点検する 2 つの項目。
   *
   * @param current 新しい項目 E
   * @param previous 前の項目 P
   */
  record Target(PhiEntry current, PhiEntry previous) {}

  /**
   * rollback.jsonl の 1 行。
   *
   * @param verdict 結果
   * @param cohorts 点検したコホート
   * @param detail 巻き戻さなかった理由や巻き戻しの行の id
   */
  record Line(Verdict verdict, List<Integer> cohorts, String detail) {}

  /**
   * 点検の時期が来ていれば点検する（項目ごとに 1 回）。
   *
   * @return 結果（点検しなければ空）
   * @throws IOException 記録が読めないか書けないとき
   */
  public static Optional<Verdict> check(GrpFiles files, PrismConfig config) throws IOException {
    if (config.phiSource() != PhiSource.SELF_GRP) return Optional.empty();
    Optional<Target> found =
        target(
            PhiHistory.read(files),
            checked(files),
            PhiHistory.nextIteration(files) - 1,
            t -> Files.exists(files.games(t)));
    if (found.isEmpty()) return Optional.empty();
    PhiEntry current = found.get().current();
    PhiEntry previous = found.get().previous();
    List<Integer> cohorts = new ArrayList<>();
    List<RecordedGame> games = new ArrayList<>();
    for (int t = current.effectiveFrom(); t < current.effectiveFrom() + COHORTS; t++) {
      CohortGames cohort = GameLog.read(files.games(t));
      if (!cohort.header().phiId().equals(current.id())) {
        Verdict none = new Verdict(current.id(), previous.id(), null, false, false);
        String detail = "cohort " + t + " was collected with " + cohort.header().phiId();
        record(files, new Line(none, List.of(t), detail));
        return Optional.of(none);
      }
      cohorts.add(t);
      games.addAll(cohort.games());
    }
    EpsilonUtilityProfile profile = EpsilonUtilityProfile.valueOf(config.utility());
    CellStats.Paired paired;
    try (HandEndPhi now = HandEndPhi.of(current, profile);
        HandEndPhi before = HandEndPhi.of(previous, profile)) {
      paired = CellStats.paired(games, now.gameValues(games), before.gameValues(games), profile);
    }
    boolean rollback = rollback(paired);
    boolean applied = false;
    String detail = "";
    if (rollback && config.grpSwitch() == GrpSwitch.AUTO) {
      PhiEntry entry = rollbackEntry(previous, current, PhiHistory.nextIteration(files) + 1, paired);
      try {
        PhiHistory.append(files, entry);
        applied = true;
        detail = entry.id();
      } catch (IllegalArgumentException refused) {
        detail = "phi.jsonl refused " + entry.id() + ": " + refused.getMessage();
      }
    }
    Verdict verdict = new Verdict(current.id(), previous.id(), paired, rollback, applied);
    record(files, new Line(verdict, cohorts, detail));
    return Optional.of(verdict);
  }

  /**
   * 点検する項目（ファイルに依らない本体）。
   *
   * @param history phi.jsonl の行
   * @param checked 点検済みの項目の id
   * @param lastSealed 最新の封印済みのコホート
   * @param recorded コホートの対局の記録があれば true
   */
  static Optional<Target> target(
      List<PhiEntry> history, Set<String> checked, int lastSealed, IntPredicate recorded) {
    PhiEntry current = PhiHistory.effectiveAt(history, lastSealed).orElse(null);
    // 同じ Φ の続きの行をさかのぼり、本当に切り替えた行にする
    while (current != null) {
      PhiEntry before = PhiHistory.effectiveAt(history, current.effectiveFrom() - 1).orElse(null);
      if (before == null || !before.sameValue(current)) break;
      current = before;
    }
    if (current == null
        || checked.contains(current.id())
        || isRollback(current)
        || current.effectiveFrom() + COHORTS - 1 > lastSealed) return Optional.empty();
    for (int t = current.effectiveFrom(); t < current.effectiveFrom() + COHORTS; t++)
      if (!recorded.test(t)) return Optional.empty();
    PhiEntry previous = PhiHistory.effectiveAt(history, current.effectiveFrom() - 1).orElse(null);
    return previous == null ? Optional.empty() : Optional.of(new Target(current, previous));
  }

  /** 前の項目が対の MSE で {@value #STANDARD_ERRORS} SE より良ければ true（差は前 − 今）。 */
  static boolean rollback(CellStats.Paired paired) {
    return paired.mseDifference() < -STANDARD_ERRORS * paired.standardError();
  }

  /** 前の項目へ戻す行（id と Φ は前の項目のまま、効く反復と理由だけ新しい）。 */
  static PhiEntry rollbackEntry(
      PhiEntry previous, PhiEntry current, int effectiveFrom, CellStats.Paired paired) {
    return previous.from(
        effectiveFrom,
        String.format(
            "rollback: %s beat %s by %.3g (SE %.3g)",
            previous.id(), current.id(), -paired.mseDifference(), paired.standardError()));
  }

  static boolean isRollback(PhiEntry entry) {
    return entry.reason().startsWith("rollback:");
  }

  /** 巻き戻しの点検の記録 grp/rollback.jsonl。 */
  static Path file(GrpFiles files) {
    return files.directory().resolve("rollback.jsonl");
  }

  private static Set<String> checked(GrpFiles files) throws IOException {
    Set<String> checked = new HashSet<>();
    for (String line : PhiHistory.lines(file(files))) {
      try {
        Line parsed = GSON.fromJson(line, Line.class);
        if (parsed != null && parsed.verdict() != null) checked.add(parsed.verdict().current());
      } catch (JsonParseException torn) {
        // 書きかけの最後の行は点検していないとみなす
      }
    }
    return checked;
  }

  /** rollback.jsonl に 1 行足す（書き手は学習の錠を持つ裏の作業だけ）。 */
  private static void record(GrpFiles files, Line line) throws IOException {
    PhiHistory.write(file(files), GSON.toJson(line));
  }
}
