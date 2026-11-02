package com.epsilon.prism.grp;

import com.epsilon.prism.run.PrismConfig;
import com.epsilon.prism.run.RunFiles;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParseException;
import com.google.gson.JsonParser;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/**
 * 学習の時期（裏の作業が学習の錠を持って呼ぶ）。手動の要求（grp/refit-request.json）が最優先。grpRefit が予定表なら、最新の封印済みの
 * コホート h が前の試行の取り置きから {@link PrismConfig#refitInterval} 以上進んだときに学習する。SCHEDULE_AND_DRIFT なら警報（{@link
 * DriftAlarm}）が ON で、前の切り替えが grpSwitchSettle まで入っていれば前倒しする。関門に落ちた後は grpRetryCooldown 反復待ち、連続の失敗が
 * grpMaxFailures に届いたら予定表を止める（手動の要求だけ受ける）。使用中が点数だけの Φ なら、窓の半荘が grpTrustGames に届くまで学習しない
 * （{@link TrustCriteria#ready}）。
 *
 * <p>切り替えの吸収: 項目が効き始めた反復 e から反復 t までの TEMPO の吸収の割合 1 − Π_{s=e}^{t} (1 − K′_s)。K′_s は iterations.jsonl の
 * filter.gain（記録のない反復は 0 と数える）。最初の項目（切り替えのない系譜）は吸収済み。
 */
public final class RefitScheduler {

  private RefitScheduler() {}

  /**
   * 学習の決定。
   *
   * @param holdout 取り置きのコホート h
   * @param cause 理由
   */
  public record Decision(int holdout, RefitCause cause) {}

  /**
   * 試行 1 回（refits.jsonl の同じ id の行をまとめ、最後の状態を持つ）。
   *
   * @param id 試行の id
   * @param holdout 取り置きのコホート
   * @param status 最後の状態
   */
  record Attempt(String id, int holdout, RefitStatus status) {}

  /**
   * 予定表の判断の材料（ファイルから組む）。
   *
   * @param lastSealed 対局の記録がある最新の封印済みのコホート h
   * @param attempts 試行（最初に現れた順）
   * @param active 反復 h + 1 に効く項目（FIXED_GRP の系譜では null）
   * @param pendingSwitch AUTO で、READY の候補がまだ phi.jsonl に入っていなければ true
   * @param driftOn 警報が ON なら true
   * @param absorbed active の吸収の割合
   * @param windowGames 取り置き h の窓の、対局の記録がある半荘の数
   * @param holdoutRecorded 取り置き h の対局の記録があれば true
   */
  record Inputs(
      int lastSealed,
      List<Attempt> attempts,
      PhiEntry active,
      boolean pendingSwitch,
      boolean driftOn,
      double absorbed,
      long windowGames,
      boolean holdoutRecorded) {}

  /**
   * 今学習するか。手動の要求は読んだら消す（取り置きがまだ封印されていなければ残して待つ）。
   *
   * @param lastSealed 対局の記録がある最新の封印済みのコホート
   * @return 学習の決定（しないなら空）
   * @throws IOException 記録が読めないとき
   */
  public static Optional<Decision> decide(GrpFiles files, PrismConfig config, int lastSealed)
      throws IOException {
    Path request = files.refitRequest();
    if (Files.exists(request)) {
      int holdout = requestedHoldout(Files.readString(request, StandardCharsets.UTF_8), lastSealed);
      if (holdout > lastSealed) return Optional.empty();
      Files.delete(request);
      return Optional.of(new Decision(holdout, RefitCause.MANUAL));
    }
    if (!config.grpRefit().scheduled()) return Optional.empty();
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry active = PhiHistory.effectiveAt(history, lastSealed + 1).orElse(null);
    boolean switched = active != null && history.indexOf(active) > 0;
    List<Attempt> attempts = attempts(RefitRecord.readAll(files));
    boolean pending = false;
    if (config.grpSwitch() == GrpSwitch.AUTO)
      for (Attempt attempt : attempts)
        if (attempt.status() == RefitStatus.READY
            && history.stream().noneMatch(entry -> entry.id().equals(attempt.id()))) pending = true;
    long windowGames = 0;
    Optional<RefitWindow> window = RefitWindow.ending(lastSealed, config.grpWindow());
    if (window.isPresent())
      for (int c = window.get().first(); c <= window.get().last(); c++)
        if (Files.exists(files.games(c))) windowGames += config.gamesPerCohort(c);
    double absorbed =
        switched ? absorbed(gains(files), active.effectiveFrom(), lastSealed) : 1;
    Inputs inputs =
        new Inputs(
            lastSealed,
            attempts,
            active,
            pending,
            config.grpRefit().drift() && DriftAlarm.isOn(files),
            absorbed,
            windowGames,
            Files.exists(files.games(lastSealed)));
    return schedule(config, inputs);
  }

  /**
   * 予定表と警報の判断（ファイルに依らない本体）。
   *
   * @return 学習の決定（しないなら空）
   */
  static Optional<Decision> schedule(PrismConfig config, Inputs in) {
    int h = in.lastSealed();
    if (!config.grpRefit().scheduled() || !in.holdoutRecorded()) return Optional.empty();
    int interval = config.refitInterval(h);
    if (interval == 0 || RefitWindow.ending(h, config.grpWindow()).isEmpty()) return Optional.empty();
    boolean scoreOnly = in.active() != null && in.active().kind().scoreBased();
    if (scoreOnly && !TrustCriteria.ready(config, in.windowGames())) return Optional.empty();
    if (failureStreak(in.attempts()) >= config.grpMaxFailures()) return Optional.empty();
    Attempt last = in.attempts().isEmpty() ? null : in.attempts().getLast();
    if (last != null && h <= last.holdout()) return Optional.empty();
    if (last != null && last.status().failure()) {
      // 落ちた後は待ちだけで決め、警報でも前倒ししない
      return h >= last.holdout() + Math.max(1, config.grpRetryCooldown())
          ? Optional.of(new Decision(h, RefitCause.SCHEDULE))
          : Optional.empty();
    }
    if (last == null || h - last.holdout() >= interval)
      return Optional.of(new Decision(h, RefitCause.SCHEDULE));
    if (config.grpRefit().drift()
        && in.driftOn()
        && !scoreOnly
        && !in.pendingSwitch()
        && in.absorbed() >= config.grpSwitchSettle())
      return Optional.of(new Decision(h, RefitCause.DRIFT));
    return Optional.empty();
  }

  /**
   * 反復 t の時点で、使用中の項目の切り替えが grpSwitchSettle まで入っていれば true（SELF_GRP 以外と最初の項目は常に true）。
   *
   * @throws IOException phi.jsonl か iterations.jsonl が読めないとき
   */
  static boolean settled(GrpFiles files, PrismConfig config, int t) throws IOException {
    if (config.phiSource() != PhiSource.SELF_GRP) return true;
    List<PhiEntry> history = PhiHistory.read(files);
    PhiEntry active = PhiHistory.effectiveAt(history, t).orElse(null);
    if (active == null || history.indexOf(active) == 0) return true;
    return absorbed(gains(files), active.effectiveFrom(), t) >= config.grpSwitchSettle();
  }

  /** 効き始めた反復 from から反復 to までの TEMPO の吸収の割合（記録のない反復のゲインは 0）。 */
  static double absorbed(Map<Integer, Double> gains, int from, int to) {
    double remaining = 1;
    for (int t = from; t <= to; t++) remaining *= 1 - gains.getOrDefault(t, 0.0);
    return 1 - remaining;
  }

  /** 末尾から続く失敗の数。 */
  static int failureStreak(List<Attempt> attempts) {
    int streak = 0;
    for (int i = attempts.size() - 1; i >= 0 && attempts.get(i).status().failure(); i--) streak++;
    return streak;
  }

  /** refits.jsonl の行を試行ごとにまとめる（最初に現れた順、状態は最後の行）。 */
  static List<Attempt> attempts(List<RefitRecord> records) {
    Map<String, Attempt> byId = new LinkedHashMap<>();
    for (RefitRecord record : records)
      byId.put(record.id(), new Attempt(record.id(), record.window().holdout(), record.status()));
    return new ArrayList<>(byId.values());
  }

  /** 手動の要求の取り置き（鍵がなければ最新の封印済みのコホート）。 */
  static int requestedHoldout(String json, int lastSealed) {
    try {
      JsonElement root = JsonParser.parseString(json);
      if (root.isJsonObject()) {
        JsonObject object = root.getAsJsonObject();
        if (object.has(GrpRefitWorkflow.HOLDOUT) && !object.get(GrpRefitWorkflow.HOLDOUT).isJsonNull())
          return object.get(GrpRefitWorkflow.HOLDOUT).getAsInt();
      }
    } catch (JsonParseException | IllegalStateException | NumberFormatException malformed) {
      // 読めない要求は最新のコホートでの学習の要求とみなす
    }
    return lastSealed;
  }

  /** iterations.jsonl の反復ごとの TEMPO のゲイン K′（filter.gain）。 */
  static Map<Integer, Double> gains(GrpFiles files) throws IOException {
    Map<Integer, Double> gains = new HashMap<>();
    Path file = files.run().resolve("reports").resolve(RunFiles.ITERATIONS);
    for (String line : PhiHistory.lines(file)) {
      JsonElement root;
      try {
        root = JsonParser.parseString(line);
      } catch (JsonParseException torn) {
        continue; // 書きかけの最後の行
      }
      if (!root.isJsonObject()) continue;
      JsonObject record = root.getAsJsonObject();
      // 収集だけで止めた反復など、filter のない行は数えない
      if (record.has("iteration")
          && record.get("filter") instanceof JsonObject filter
          && filter.has("gain"))
        gains.put(record.get("iteration").getAsInt(), filter.get("gain").getAsDouble());
    }
    return gains;
  }
}
