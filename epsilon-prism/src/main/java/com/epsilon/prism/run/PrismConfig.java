package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.collect.EventLedger;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnScheme;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.hip.HipLibrary;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.learn.LedgerFit;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.model.PrismShape;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.PriorSettings;
import com.google.gson.Gson;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;
import com.google.gson.JsonPrimitive;
import java.lang.reflect.RecordComponent;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/**
 * run 全体で凍結する設定。run の開始時に config.json へ保存し、再開ではそれを読む（{@link #read}）。鍵は欄の名前（設定の [epsilon.prism] と
 * 同じ）で、config.json の鍵は欄とちょうど同じでなければならない。
 *
 * <p>学習器は系譜の形に決まっている: 生徒 1 本を TEMPO（コホート t だけを、学習の網 θ_t を錨にして当てる再帰平均。{@link #filterGain}）で温めて
 * 当て、損失は差分整合（κ = 0、KL の基準は C の上の一様分布）。生徒の行は TRAIN のラベル行と根の記録（{@link Selection#LABELS_AND_ROOTS}）で、
 * 教師は運の精算の帳簿で補正した Ỹ = Y − β_c·L（{@link LedgerFit}）。収集は打牌の類で全ての合法手を測り、ラベルの枠ごとに帳簿を書く
 * （{@link #sampling}）。版 1 は白紙の網（形は {@link #shape}）。
 *
 * @param format config.json の形式（{@value #FORMAT}）
 * @param seed 牌山・抽選・網の初期値の種
 * @param gamesPerCohort 1 コホートの半荘の数（gamesSchedule が空のとき）
 * @param gamesSchedule 1 コホートの半荘の数の段: 「始めの反復:半荘の数」を読点で区切り始めの昇順（例 "1:8192,9:16384,17:32768,25:65536"。
 *     反復 t は始めが t 以下で最も後の段の数）。空なら gamesPerCohort のまま
 * @param iterations 反復の数
 * @param utility common の順位効用の名前
 * @param collectDevices 収集と評価のデバイス（cpu か gpuN の列）
 * @param inference 網の推論の道（{@link InferencePath}。HIP でも、HIP の持たない形とデバイスは eager。{@link InferencePath#resolve}）
 * @param epsilonB 行動方策 b の一様な床 ε_b（全席）
 * @param exploreTemperature 主経路の探索の温度 τ（効用の単位。主経路の判断だけ softmax(s/τ) に床を混ぜて引き、分岐と写しの続きは b のまま。0 なら
 *     探索しない。{@link Behaviour#explore}）
 * @param holdoutFraction 保留（HOLDOUT）にする半荘の割合
 * @param labelRateDiscardEarly 脅威のない序盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardMid 脅威のない中盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardLate 脅威のない終盤の打牌のラベル行の率 ρ
 * @param labelRateCommit 確約の類のラベル行の率 ρ（オーラスの和了のほか）
 * @param labelRateFinalEnd オーラスの和了（{@link DecisionClass#FINAL_END}）のラベル行の率 ρ（既定 1 で全て分岐する）
 * @param maxCandidates 確約の類の候補集合の大きさの上限
 * @param epsilonC 確約の類の候補集合に一様な 1 手を足す確率 ε_C
 * @param rollinRateDiscard 打牌の類の中立ロールインの率 β
 * @param rollinRateCommit 確約の類の中立ロールインの率 β
 * @param rollinClassRates 類ごとの中立ロールインの率の上書き（"類=率" を読点で区切る。類は {@link DecisionClass} の名前。空なら上書きしない）
 * @param epsilonQ 逸脱の分布 q の一様な床 ε_q
 * @param rhoBranch 写しの中の逸脱した席の判断をラベル行にする率 ρ_br
 * @param maxCopyLabels 写しごとのラベル行の上限
 * @param priorAlpha 規則事前の向聴 1 つあたりの nat
 * @param priorGamma 規則事前の受け入れ 1 枚あたりの nat
 * @param learnerDevice 学習のデバイス（先頭が正本で、残りはデータ並列の複製）
 * @param depth 網の背骨の層の数（2 か 3。幅は標準の形 H 256・F 512・G 256）
 * @param frontier 網を前線の腕にするか（{@link PrismShape#frontier}）
 * @param auxOracle 読みの頭の重み α_o（0 以上）
 * @param auxResult 候補の結果の頭の重み α_r（0 以上）
 * @param keepCohorts 残す行のコホートの数（古いものから消す）
 * @param fitSteps 生徒の歩数の上限
 * @param fitEpochs 生徒が 1 行を見る回数の上限（歩数は {@link #steps}）
 * @param batchSize 1 歩の行の数
 * @param lrPeak 学習率の山
 * @param lrFloor 学習率の床
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param decodeThreads 行の読み出しと符号化のスレッドの数
 * @param lambdaDiscard 打牌の類の温度 λ
 * @param lambdaCommit 確約の類の温度 λ
 * @param averaging 方策の記憶: 版の重みの平均の決め方（{@link Averaging}: none・ema・power）
 * @param averageHalfLife ema の半減期（反復）
 * @param averagePower power の重みの指数 p（版 k の重み k^p。0 で一様、1 で線形）
 * @param averageFrom 平均を始める版（それより前の版は平均せず、打つ網 = 学習の網）
 * @param collectWith 収集で打つ網（{@value Averaging#AVERAGE} = 平均の網、{@value Averaging#ITERATE} = 学習の網）
 * @param learnFrom 生徒を始める網と TEMPO の錨（{@value Averaging#ITERATE} か {@value Averaging#AVERAGE}）
 * @param evaluateIterate 打つ網に加えて学習の網も 3×Heuristic で測るか（平均があるときだけ意味がある）
 * @param filterFrom TEMPO のゲインの数え始めの反復 t0
 * @param filterGain TEMPO のゲインの始めの値 K0（(0, 1)）
 * @param filterTau TEMPO のゲインの減りの時定数 τ（反復、正）
 * @param filterPower TEMPO のゲインの減りの冪 p（0 以上。0 なら一定のゲイン K0）
 * @param filterLrMode TEMPO の学習率の倍率の決め方（{@link FilterLrMode}。GAIN = min(1, K′_t / {@value #FILTER_LR_GAIN})、FULL = 1。K′_t の予定は
 *     変えない）
 * @param phiTemperatures Φ の較正: 次局の局番号ごとの温度を読点で区切ったもの（GRP の順位分布を p' ∝ p^(1/T) にする。{@link
 *     GrpPhi#calibrated}。空なら較正しない）
 * @param phiCalibration Φ の効用の空間の較正の表（{@link PhiCalibration} の JSON のパス。空なら使わない。phiTemperatures の後に掛ける）
 * @param phi 局末の値 Φ の種類（{@value #PHI_GRP} = 固定の GRP、{@value #PHI_SCORE_RANK} = 局末の点数をそのまま最終順位とみなす試験用の代用）
 * @param grpCheckpoint Φ の固定の GRP（phi = grp の train では必須）
 * @param grpDevice GRP のデバイス
 * @param grpBatch GRP の 1 回の評価の局末の数
 * @param evalEvery 3×Heuristic の評価の間隔（反復。0 なら測らない）
 * @param evalGames 3×Heuristic の評価の半荘の数（evaluate の既定の半荘の数）
 */
public record PrismConfig(
    int format,
    long seed,
    int gamesPerCohort,
    String gamesSchedule,
    int iterations,
    String utility,
    // 収集
    List<String> collectDevices,
    int maxActiveGames,
    int advanceWorkers,
    int encoders,
    int slots,
    int readyBatches,
    int inferBatch,
    long batchWaitMicros,
    int rowWriters,
    InferencePath inference,
    // 行動方策と保留
    double epsilonB,
    double exploreTemperature,
    double holdoutFraction,
    // ラベル行
    double labelRateDiscardEarly,
    double labelRateDiscardMid,
    double labelRateDiscardLate,
    double labelRateCommit,
    double labelRateFinalEnd,
    int maxCandidates,
    double epsilonC,
    // 中立ロールイン
    double rollinRateDiscard,
    double rollinRateCommit,
    String rollinClassRates,
    double epsilonQ,
    double rhoBranch,
    int maxCopyLabels,
    // 規則事前 p0
    double priorAlpha,
    double priorGamma,
    // 網
    String learnerDevice,
    int depth,
    boolean frontier,
    double auxOracle,
    double auxResult,
    // 生徒
    int keepCohorts,
    int fitSteps,
    double fitEpochs,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    int decodeThreads,
    double lambdaDiscard,
    double lambdaCommit,
    // 方策の記憶（版の重みの平均）
    String averaging,
    double averageHalfLife,
    double averagePower,
    int averageFrom,
    String collectWith,
    String learnFrom,
    boolean evaluateIterate,
    // TEMPO
    int filterFrom,
    double filterGain,
    double filterTau,
    double filterPower,
    FilterLrMode filterLrMode,
    // Φ の較正
    String phiTemperatures,
    String phiCalibration,
    // Φ
    String phi,
    String grpCheckpoint,
    String grpDevice,
    int grpBatch,
    // 評価
    int evalEvery,
    int evalGames) {

  /** config.json の形式。 */
  public static final int FORMAT = 2;

  /** Φ の種類: 固定の GRP（本線）。 */
  public static final String PHI_GRP = "grp";

  /** Φ の種類: 局末の点数をそのまま最終順位とみなす代用（GRP のない CPU の試験だけ）。 */
  public static final String PHI_SCORE_RANK = "score-rank";

  /**
   * TEMPO の学習率の基準のゲイン: filterLrMode = GAIN なら学習率を lrPeak・lrFloor の min(1, K′_t / これ) 倍にする（錨の重みが大きいほど 1 歩を
   * 小さく）。
   */
  public static final double FILTER_LR_GAIN = 0.25;

  /** 設定の鍵の接頭辞（{@link #fromSettings}）。 */
  public static final String SETTINGS = "epsilon.prism.";

  /** 1コホートの半荘の数の上限。半荘の ID は (t−1)·この値 + 番号。 */
  public static final int MAX_GAMES_PER_COHORT = 1 << 20;

  /** 形式の欄の名前（設定からは読まない）。 */
  private static final String FORMAT_FIELD = "format";

  private static final Gson GSON = new Gson();

  /**
   * 半荘の数の段の 1 つ。
   *
   * @param from 始めの反復（1 から）
   * @param games 1 コホートの半荘の数
   */
  public record GamesStep(int from, int games) {}

  /**
   * gamesSchedule を読む（書いた順。始めの昇順は {@link #validate} が確かめる）。
   *
   * @return 段（空なら空の列）
   * @throws IllegalArgumentException 「始め:数」の形でないとき
   */
  public List<GamesStep> gamesSteps() {
    if (gamesSchedule.isBlank()) return List.of();
    List<GamesStep> steps = new ArrayList<>();
    for (String entry : gamesSchedule.split(",")) {
      String[] pair = entry.strip().split(":");
      if (pair.length != 2)
        throw new IllegalArgumentException("gamesSchedule entries are from:games, got " + entry);
      try {
        steps.add(new GamesStep(Integer.parseInt(pair[0].strip()), Integer.parseInt(pair[1].strip())));
      } catch (NumberFormatException invalid) {
        throw new IllegalArgumentException("gamesSchedule entries are from:games, got " + entry, invalid);
      }
    }
    return steps;
  }

  /**
   * 反復 t のコホートの半荘の数（gamesSchedule の段のうち始めが t 以下で最も後のもの。段がなければ gamesPerCohort）。
   *
   * @param t 反復
   * @return 半荘の数
   */
  public int gamesPerCohort(int t) {
    int games = gamesPerCohort;
    for (GamesStep step : gamesSteps()) {
      if (step.from() <= t) games = step.games();
    }
    return games;
  }

  /** コホート t の半荘 ID（(t−1)·{@value #MAX_GAMES_PER_COHORT} + 番号）。 */
  public static List<Long> cohortGames(int t, int games) {
    List<Long> ids = new ArrayList<>(games);
    for (int game = 0; game < games; game++) ids.add((long) (t - 1) * MAX_GAMES_PER_COHORT + game);
    return ids;
  }

  /** コホート t の種（局のキーと硬貨の鍵の素）。 */
  public static long cohortSeed(long seed, int t) {
    return CrnSeeds.derive(seed, CrnSeeds.Domain.HAND_KEY, t);
  }

  /**
   * 新しい run の設定を、設定の [epsilon.prism]（同梱の prism/settings.toml に --settings と -Depsilon.prism.* を重ねたもの）から作る。鍵は欄の
   * 名前で、列（collectDevices）は読点で区切る。形式は設定から読まない。知らない鍵は拒む（{@link #requireKnownSettings}）。
   */
  public static PrismConfig fromSettings(SettingsLoader settings) {
    requireKnownSettings(settings);
    var tree = new JsonObject();
    tree.addProperty(FORMAT_FIELD, FORMAT);
    for (RecordComponent component : PrismConfig.class.getRecordComponents()) {
      String name = component.getName();
      if (name.equals(FORMAT_FIELD)) continue;
      String value = settings.values().get(SETTINGS + name);
      if (value == null) throw new IllegalArgumentException("Missing setting " + SETTINGS + name);
      tree.add(name, json(component, value));
    }
    return GSON.fromJson(tree, PrismConfig.class);
  }

  /**
   * 設定の {@value #SETTINGS} で始まる鍵が全て欄の名前であることを確かめる（形式は設定の鍵でない）。この接頭辞の鍵で欄でないのは、HIP の実行系の
   * 検査つきの組み立てを選ぶ系の性質（{@link HipLibrary#CHECKED_PROPERTY}）だけ。再開の train も、設定を使わないがこれで確かめる。
   *
   * @throws IllegalArgumentException 知らない鍵があるとき（打ち間違いか、なくなった設定）
   */
  public static void requireKnownSettings(SettingsLoader settings) {
    Set<String> known = new HashSet<>();
    for (RecordComponent component : PrismConfig.class.getRecordComponents())
      if (!component.getName().equals(FORMAT_FIELD)) known.add(SETTINGS + component.getName());
    for (String key : settings.values().keySet())
      if (key.startsWith(SETTINGS) && !key.equals(HipLibrary.CHECKED_PROPERTY) && !known.contains(key))
        throw new IllegalArgumentException("Unknown setting " + key);
  }

  /**
   * config.json を読む。鍵は欄の名前とちょうど同じでなければならない（知らない鍵も欠けた鍵も拒む）。形式と値の範囲は {@link #validate} が
   * 確かめる。
   *
   * @param json config.json の中身
   * @throws IllegalArgumentException JSON の物でないか、鍵が欄と合わないとき
   */
  public static PrismConfig read(String json) {
    JsonElement parsed = JsonParser.parseString(json);
    if (!parsed.isJsonObject()) throw new IllegalArgumentException("config.json is not a JSON object");
    return read(parsed.getAsJsonObject());
  }

  /** config.json の木を読む（{@link #read(String)} と同じ決まり）。 */
  static PrismConfig read(JsonObject tree) {
    Set<String> fields = new HashSet<>();
    for (RecordComponent component : PrismConfig.class.getRecordComponents()) {
      String name = component.getName();
      fields.add(name);
      JsonElement value = tree.get(name);
      if (value == null || value.isJsonNull()) throw new IllegalArgumentException("Missing config.json key " + name);
    }
    for (String key : tree.keySet())
      if (!fields.contains(key)) throw new IllegalArgumentException("Unknown config.json key " + key);
    return GSON.fromJson(tree, PrismConfig.class);
  }

  /**
   * 欄 name の 1 つの値（数・真偽・文字列・enum）を、設定と同じ読み方で config.json の値にする（refit の上書き）。
   *
   * @throws IllegalArgumentException 知らない欄か、列の欄か形式か、型の合わない値のとき
   */
  static JsonElement scalarValue(String name, String value) {
    for (RecordComponent component : PrismConfig.class.getRecordComponents())
      if (component.getName().equals(name) && !name.equals(FORMAT_FIELD) && component.getType() != List.class)
        return json(component, value);
    throw new IllegalArgumentException("Unknown or non-scalar config field: " + name);
  }

  /** 設定の値を config.json の値にする。数は config.json を読むときと同じ形で持ち、enum は定数の名前に限る。 */
  private static JsonElement json(RecordComponent component, String value) {
    Class<?> type = component.getType();
    if (type == String.class) return new JsonPrimitive(value);
    if (type.isEnum()) {
      String text = value.trim();
      for (Object constant : type.getEnumConstants())
        if (((Enum<?>) constant).name().equals(text)) return new JsonPrimitive(text);
      throw new IllegalArgumentException(
          component.getName() + " must be one of " + Arrays.toString(type.getEnumConstants()) + ": " + value);
    }
    if (type == boolean.class) {
      String text = value.trim();
      if (!text.equalsIgnoreCase("true") && !text.equalsIgnoreCase("false"))
        throw new IllegalArgumentException(
            component.getName() + " must be true or false: " + value);
      return new JsonPrimitive(Boolean.parseBoolean(text));
    }
    if (type == List.class) {
      var array = new JsonArray();
      for (String part : value.split(",")) array.add(part.trim());
      return array;
    }
    JsonElement parsed = JsonParser.parseString(value.trim().replace("_", ""));
    if (!parsed.isJsonPrimitive() || !parsed.getAsJsonPrimitive().isNumber())
      throw new IllegalArgumentException(component.getName() + " must be a number: " + value);
    return parsed;
  }

  /** 設定の範囲を確かめる。新しい run と、再開で読んだ config.json に使う。 */
  public void validate() {
    if (format != FORMAT)
      throw new IllegalArgumentException("config.json format " + format + " is not " + FORMAT);
    devices(collectDevices);
    learnerDevices();
    device(grpDevice);
    // 収集の率は Sampling と Behaviour が、p0 の係数は PriorSettings が、網の形は PrismShape が確かめる
    sampling();
    behaviour();
    priorSettings();
    shape();
    if (!EpsilonUtilityProfile.valueOf(utility).rankBased())
      throw new IllegalArgumentException("Prism requires a rank-based utility profile: " + utility);
    if (gamesPerCohort < 1 || gamesPerCohort > MAX_GAMES_PER_COHORT || iterations < 1)
      throw new IllegalArgumentException("Invalid run size");
    int previous = 0;
    for (GamesStep step : gamesSteps()) {
      if (step.from() <= previous || step.games() < 1 || step.games() > MAX_GAMES_PER_COHORT)
        throw new IllegalArgumentException("Invalid gamesSchedule " + gamesSchedule);
      previous = step.from();
    }
    if (maxActiveGames < 1
        || advanceWorkers < 1
        || encoders < 1
        || slots < 1
        || readyBatches < 1
        || inferBatch < 1
        || batchWaitMicros < 0
        || rowWriters < 1) throw new IllegalArgumentException("Invalid collection settings");
    if (keepCohorts < 1
        || fitSteps < 0
        || !(fitEpochs > 0)
        || batchSize < 1
        || decodeThreads < 1) throw new IllegalArgumentException("Invalid fit settings");
    if (!(lrPeak > 0) || !Double.isFinite(lrPeak) || !(lrFloor >= 0 && lrFloor <= lrPeak))
      throw new IllegalArgumentException("Invalid learning rates " + lrPeak + "/" + lrFloor);
    if (!(gradientClip >= 0) || !Float.isFinite(gradientClip))
      throw new IllegalArgumentException("Invalid gradient clip " + gradientClip);
    if (!(lambdaDiscard > 0) || !(lambdaCommit > 0))
      throw new IllegalArgumentException("Invalid loss settings " + lambdaDiscard + "/" + lambdaCommit);
    policyMemory();
    for (String net : new String[] {collectWith, learnFrom})
      if (!Averaging.AVERAGE.equals(net) && !Averaging.ITERATE.equals(net))
        throw new IllegalArgumentException(
            "collectWith and learnFrom must be "
                + Averaging.AVERAGE
                + " or "
                + Averaging.ITERATE
                + ": "
                + net);
    if (filterFrom < 1 || !(filterGain > 0 && filterGain < 1) || !(filterTau > 0) || !(filterPower >= 0))
      throw new IllegalArgumentException(
          "Invalid filter settings " + filterFrom + "/" + filterGain + "/" + filterTau + "/" + filterPower);
    if (!(auxOracle >= 0) || !Double.isFinite(auxOracle))
      throw new IllegalArgumentException("auxOracle must be a finite non-negative weight: " + auxOracle);
    if (!(auxResult >= 0) || !Double.isFinite(auxResult))
      throw new IllegalArgumentException("auxResult must be a finite non-negative weight: " + auxResult);
    if (!phi.equals(PHI_GRP) && !phi.equals(PHI_SCORE_RANK))
      throw new IllegalArgumentException(
          "phi must be " + PHI_GRP + " or " + PHI_SCORE_RANK + ": " + phi);
    if (grpBatch < 1 || evalEvery < 0 || evalGames < 1)
      throw new IllegalArgumentException("Invalid Φ or evaluation settings");
  }

  /**
   * train の設定を確かめる（{@link #validate} に加えて、Φ が決まっていること）。phi = grp なら grpCheckpoint が要る。黙って点数の順位の代用に
   * 落ちると、ラベルと根が全て別の目的の値になるので、代用は phi = score-rank と明示したときだけ許す。evaluate は Φ を使わないので
   * {@link #validate} だけを呼ぶ。
   */
  public void validateTraining() {
    validate();
    if (phi.equals(PHI_GRP) && grpCheckpoint.isEmpty())
      throw new IllegalArgumentException(
          "phi = grp needs grpCheckpoint (set phi = \"" + PHI_SCORE_RANK + "\" only for tests)");
  }

  /**
   * 収集の率。打牌の類（脅威のない序盤・中盤・終盤）では全ての合法手を測り、ラベルの枠ごとに運の精算の帳簿（{@link EventLedger}）を行に書く。
   */
  public Sampling sampling() {
    return new Sampling(
        holdoutFraction,
        labelRateDiscardEarly,
        labelRateDiscardMid,
        labelRateDiscardLate,
        labelRateCommit,
        labelRateFinalEnd,
        maxCandidates,
        true,
        epsilonC,
        rollinRateDiscard,
        rollinRateCommit,
        epsilonQ,
        rhoBranch,
        maxCopyLabels,
        rollinOverrides(),
        true);
  }

  /** 類ごとの中立ロールインの率の上書き（rollinClassRates。書かない類は NaN）。 */
  public double[] rollinOverrides() {
    double[] rates = Sampling.noOverrides();
    if (rollinClassRates.isBlank()) return rates;
    for (String entry : rollinClassRates.split(",")) {
      String[] parts = entry.strip().split("=");
      if (parts.length != 2) throw new IllegalArgumentException("Bad rollinClassRates entry: " + entry);
      rates[DecisionClass.valueOf(parts[0].strip()).ordinal()] = Double.parseDouble(parts[1].strip());
    }
    return rates;
  }

  /** 収集の全席の行動方策 b = (1 − ε_b)·貪欲 + ε_b/n（乱数の使い方は {@link CrnScheme#V2}）。 */
  public Behaviour behaviour() {
    return new Behaviour(epsilonB, exploreTemperature, CrnScheme.V2);
  }

  /** Φ の温度（次局の局番号ごと。phiTemperatures を読点で区切る。空なら較正しない）。 */
  public double[] phiTemperatureTable() {
    if (phiTemperatures.isBlank()) return new double[0];
    String[] parts = phiTemperatures.split(",");
    double[] temperatures = new double[parts.length];
    for (int i = 0; i < parts.length; i++) {
      temperatures[i] = Double.parseDouble(parts[i].strip());
      if (!(temperatures[i] > 0)) throw new IllegalArgumentException("phiTemperatures must be positive: " + phiTemperatures);
    }
    return temperatures;
  }

  /** Φ の較正の表を使うか（phiCalibration が空でない）。 */
  public boolean hasPhiCalibration() {
    return !phiCalibration.isBlank();
  }

  /**
   * TEMPO の反復 t のゲイン K_t = K0·(1 + max(0, t − t0)/τ)^(−p)。教師は (1 − K_t)·s_θt + K_t·Y で、K_t が減るほど網の出力は過去の全ての
   * コホートのラベルの平均に近づき、ラベルの雑音の床が K_t に比例して下がる（記憶が伸び続ける）。
   */
  public double filterGain(int t) {
    return filterGain * Math.pow(1 + Math.max(0, t - filterFrom) / filterTau, -filterPower);
  }

  /**
   * 教師の雑音を R² だけ除いたときの TEMPO の反復 t のゲイン K′_t = K_t / (K_t + (1 − K_t)(1 − R²))。K_t を雑音の分散 σ² の下の最適なゲインと
   * みなすと、雑音が (1 − R²)·σ² に減ったときの最適なゲインがこの K′_t（雑音の得を古いラベルの平均に回さない）。R² は教師に掛けた補正の保留の
   * R²（行の中で中心化した Y。{@link LedgerFit}）で、0 以下か NaN なら K_t そのもの。
   *
   * @throws IllegalArgumentException R² が 1 を超えるとき
   */
  public double filterGain(int t, double r2) {
    double gain = filterGain(t);
    if (!(r2 > 0)) return gain;
    if (!(r2 <= 1)) throw new IllegalArgumentException("R² must not exceed 1: " + r2);
    return gain / (gain + (1 - gain) * (1 - r2));
  }

  /**
   * TEMPO の反復 t の学習率の倍率: K′_t（{@link #filterGain(int, double)}）を filterLrMode で倍率にしたもの（GAIN なら
   * min(1, K′_t / {@value #FILTER_LR_GAIN})、FULL なら 1。{@link FilterLrMode#scale}）。教師と補助の頭の K′_t は filterLrMode に依らない。
   */
  public double filterLrScale(int t, double r2) {
    return filterLrMode.scale(filterGain(t, r2));
  }

  /** 網の形（標準の形 H 256・F 512・G 256 に、深さ depth と前線の腕 frontier。範囲は {@link PrismShape} が確かめる）。 */
  public PrismShape shape() {
    return PrismShape.standard(depth).withFrontier(frontier);
  }

  /** 補助の頭の重み（読みの頭 α_o は auxOracle、候補の結果の頭 α_r は auxResult）。 */
  public AuxiliaryLoss auxiliaryLoss() {
    return new AuxiliaryLoss((float) auxOracle, (float) auxResult);
  }

  /** 方策の記憶（版の重みの平均）の決め方。 */
  public Averaging policyMemory() {
    return new Averaging(averaging, averageHalfLife, averagePower, averageFrom);
  }

  /** 規則事前 p0 の係数。 */
  public PriorSettings priorSettings() {
    return new PriorSettings(priorAlpha, priorGamma);
  }

  /**
   * 行が rows 本のときの歩数: 上限 cap と「周回数 epochs × 行 ÷ バッチ」の小さい方（cap が 0 か行がなければ 0、それ以外は 1 以上）。雑音の大きい
   * 教師を何周もすると覚え込むので、1 行を見る回数を周回数で抑える。
   */
  public static int steps(int cap, double epochs, int rows, int batchSize) {
    if (cap == 0 || rows == 0) return 0;
    long byEpochs = (long) Math.ceil(epochs * rows / batchSize);
    return (int) Math.max(1, Math.min(cap, byEpochs));
  }

  /** 歩数 steps の当てはめの設定（歩数は {@link #steps} で決める）。 */
  public FitSettings fitSettings(int steps) {
    return fitSettings(steps, 1);
  }

  /** 歩数 steps で、学習率を lrScale 倍にした当てはめの設定（TEMPO の {@link #filterLrScale}）。 */
  public FitSettings fitSettings(int steps, double lrScale) {
    return new FitSettings(
        steps, batchSize, lrPeak * lrScale, lrFloor * lrScale, gradientClip, lambdaDiscard, lambdaCommit);
  }

  /** デバイス名（cpu か gpuN）。 */
  public static Device device(String name) {
    return name.equals("cpu") ? Device.cpu() : Device.gpu(Integer.parseInt(name.substring(3)));
  }

  /** デバイスの列。cpu 1つか、重複のない GPU の列でなければならない。 */
  public static List<Device> devices(List<String> names) {
    List<Device> devices = names.stream().map(PrismConfig::device).toList();
    boolean cpu = devices.stream().anyMatch(device -> !device.isGpu());
    if ((cpu && devices.size() > 1) || devices.stream().distinct().count() < devices.size())
      throw new IllegalArgumentException("Devices must be one cpu or distinct GPUs: " + names);
    return devices;
  }

  /** 学習のデバイス。先頭が正本で、残りはデータ並列の複製。 */
  public NetworkDevices learnerDevices() {
    return NetworkDevices.of(
        devices(Arrays.asList(learnerDevice.split(","))).toArray(Device[]::new));
  }
}
