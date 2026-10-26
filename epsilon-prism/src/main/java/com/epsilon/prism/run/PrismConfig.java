package com.epsilon.prism.run;

import ai.djl.Device;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.network.NetworkDevices;
import com.epsilon.prism.collect.Behaviour;
import com.epsilon.prism.collect.Sampling;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.model.ModelConfig;
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
 * run 全体で凍結する設定。run の開始時に config.json へ保存し、再開ではそれを読む。鍵は欄の名前（設定の [epsilon.prism] と同じ）。
 *
 * @param seed 牌山・抽選・網の初期値の種
 * @param gamesPerCohort 1 コホートの半荘の数
 * @param iterations 反復の数
 * @param utility common の順位効用の名前
 * @param collectDevices 収集と評価のデバイス（cpu か gpuN の列）
 * @param epsilonB 行動方策 b の一様な床 ε_b（全席）
 * @param holdoutFraction 保留（HOLDOUT）にする半荘の割合
 * @param labelRateDiscardEarly 脅威のない序盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardMid 脅威のない中盤の打牌のラベル行の率 ρ
 * @param labelRateDiscardLate 脅威のない終盤の打牌のラベル行の率 ρ
 * @param labelRateCommit 確約の類のラベル行の率 ρ
 * @param maxCandidates 候補集合の大きさの上限
 * @param measureAllDiscards 打牌の類（脅威のない序盤・中盤・終盤）では全ての合法手を測るか（候補集合を全ての合法手にする）
 * @param epsilonC 候補集合に一様な 1 手を足す確率 ε_C
 * @param rollinRateDiscard 打牌の類の中立ロールインの率 β
 * @param rollinRateCommit 確約の類の中立ロールインの率 β
 * @param epsilonQ 逸脱の分布 q の一様な床 ε_q
 * @param rhoBranch 写しの中の逸脱した席の判断をラベル行にする率 ρ_br
 * @param maxCopyLabels 写しごとのラベル行の上限
 * @param priorAlpha 規則事前の向聴 1 つあたりの nat
 * @param priorGamma 規則事前の受け入れ 1 枚あたりの nat
 * @param learnerDevice 学習のデバイス（先頭が正本で、残りはデータ並列の複製）
 * @param hidden 網の幅（白紙から始めるとき。startSnapshot からなら版の幅）
 * @param startSnapshot 版 1 にする epsilon-prism の版のディレクトリ（空なら白紙の網）
 * @param memoryCohorts データの記憶: 生徒と蒸留に使う直近のコホートの数の上限（{@link #memoryKeep}）
 * @param memoryDecay データの記憶: 年齢 a（最新が 0）のコホートのラベル行のうち使う割合 d^a の d（1 なら全てを同じ重みで使う）
 * @param keepCohorts 残す行のコホートの数（古いものから消す。memoryCohorts 以上）
 * @param fitSteps 生徒 1 本の歩数の上限
 * @param distillSteps 蒸留の歩数の上限
 * @param fitEpochs 生徒が 1 行を見る回数の上限（歩数は {@link #steps}）
 * @param distillEpochs 蒸留が 1 行を見る回数の上限
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param lambdaDiscard 打牌の類の温度 λ
 * @param lambdaCommit 確約の類の温度 λ
 * @param kappa 差分整合の C の外の項の係数 κ
 * @param klReference 差分整合の KL の基準（{@value #KL_UNIFORM} = C の上の一様分布で、最適は s = Y + 定数。{@value #KL_PRIOR} = 規則事前
 *     p0_C で、最適は s = Y + λ_c·log p0_C + 定数となり、貪欲が向聴・受け入れの良い手へ傾く）
 * @param learner 学習の網の作り方（{@value #LEARNER_CROSSFIT} = 組 A・B の生徒を交差適合し、その平均を蒸留して認証する。{@value
 *     #LEARNER_SINGLE} = 生徒 1 本を両方の組の TRAIN の行で当てて、そのまま学習の網にする。蒸留と認証はしない）
 * @param averaging 方策の記憶: 版の重みの平均の決め方（{@link Averaging}: none・ema・power）
 * @param averageHalfLife ema の半減期（反復）
 * @param averagePower power の重みの指数 p（版 k の重み k^p。0 で一様、1 で線形）
 * @param averageFrom 平均を始める版（それより前の版は平均せず、打つ網 = 学習の網）
 * @param collectWith 収集で打つ網（{@value Averaging#AVERAGE} = 平均の網、{@value Averaging#ITERATE} = 学習の網）
 * @param learnFrom 生徒と蒸留を始める網（{@value Averaging#ITERATE} か {@value Averaging#AVERAGE}）
 * @param evaluateIterate 打つ網に加えて学習の網も 3×Heuristic で測るか（平均があるときだけ意味がある）
 * @param certify 認証で π_{t+1} を決めるか（false なら蒸留した網を常に受け入れる。認証の量は報告する）
 * @param certifyTolerance 確約の類の下限（判断 1 つあたりの改善、効用の単位）がこれの負を超えれば受け入れる
 * @param certifyMaxOutsideShare 確約の類の乗り換えのうち C の外（q の台の外で測れない）だった割合の上限。超えたら受け入れない
 * @param phi 局末の値 Φ の種類（{@value #PHI_GRP} = 固定の GRP、{@value #PHI_SCORE_RANK} = 局末の点数をそのまま最終順位とみなす試験用の代用）
 * @param grpCheckpoint Φ の固定の GRP（phi = grp の train では必須）
 * @param evalEvery 3×Heuristic の評価の間隔（反復。0 なら測らない）
 */
public record PrismConfig(
    int format,
    long seed,
    int gamesPerCohort,
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
    int multiTransitionBatch,
    long batchWaitMicros,
    int rowWriters,
    // 行動方策と保留
    double epsilonB,
    double holdoutFraction,
    // ラベル行（M1）
    double labelRateDiscardEarly,
    double labelRateDiscardMid,
    double labelRateDiscardLate,
    double labelRateCommit,
    int maxCandidates,
    boolean measureAllDiscards,
    double epsilonC,
    // 中立ロールイン（M2）
    double rollinRateDiscard,
    double rollinRateCommit,
    double epsilonQ,
    double rhoBranch,
    int maxCopyLabels,
    // 規則事前 p0
    double priorAlpha,
    double priorGamma,
    // 網
    String learnerDevice,
    int hidden,
    String precision,
    String startSnapshot,
    // 生徒と蒸留（M3）とデータの記憶
    int memoryCohorts,
    double memoryDecay,
    int keepCohorts,
    int fitSteps,
    int distillSteps,
    double fitEpochs,
    double distillEpochs,
    int batchSize,
    double lrPeak,
    double lrFloor,
    float gradientClip,
    int decodeThreads,
    double lambdaDiscard,
    double lambdaCommit,
    double kappa,
    String klReference,
    String learner,
    // 方策の記憶（版の重みの平均）
    String averaging,
    double averageHalfLife,
    double averagePower,
    int averageFrom,
    String collectWith,
    String learnFrom,
    boolean evaluateIterate,
    // 認証（S）
    boolean certify,
    double certifyTolerance,
    double certifyMaxOutsideShare,
    // Φ
    String phi,
    String grpCheckpoint,
    String grpDevice,
    int grpBatch,
    // 評価
    int evalEvery,
    int evalGames) {
  public static final int FORMAT = 1;

  /** Φ の種類: 固定の GRP（本線）。 */
  public static final String PHI_GRP = "grp";

  /** Φ の種類: 局末の点数をそのまま最終順位とみなす代用（GRP のない CPU の試験だけ）。 */
  public static final String PHI_SCORE_RANK = "score-rank";

  /** KL の基準: C の上の一様分布（E1 の PRISM と同じ。貪欲は argmax Y）。 */
  public static final String KL_UNIFORM = "uniform";

  /** KL の基準: 規則事前 p0_C（貪欲は argmax (Y + λ_c·log p0)）。 */
  public static final String KL_PRIOR = "prior";

  /** 学習の網の作り方: 交差適合した生徒 A・B の平均を蒸留する。 */
  public static final String LEARNER_CROSSFIT = "crossfit";

  /** 学習の網の作り方: 全データの生徒 1 本。 */
  public static final String LEARNER_SINGLE = "single";

  /** 設定の鍵の接頭辞（{@link #fromSettings}）。 */
  public static final String SETTINGS = "epsilon.prism.";

  /** 1コホートの半荘の数の上限。半荘の ID は (t−1)·この値 + 番号。 */
  public static final int MAX_GAMES_PER_COHORT = 1 << 20;

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
   * 名前で、列（collectDevices）は読点で区切る。形式は設定から読まない。知らない鍵は打ち間違いとして拒む。
   */
  public static PrismConfig fromSettings(SettingsLoader settings) {
    var tree = new JsonObject();
    tree.addProperty("format", FORMAT);
    Set<String> keys = new HashSet<>();
    for (RecordComponent component : PrismConfig.class.getRecordComponents()) {
      String name = component.getName();
      if (name.equals("format")) continue;
      keys.add(SETTINGS + name);
      String value = settings.values().get(SETTINGS + name);
      if (value == null) throw new IllegalArgumentException("Missing setting " + SETTINGS + name);
      tree.add(name, json(component, value));
    }
    for (String key : settings.values().keySet())
      if (key.startsWith(SETTINGS)
          && key.indexOf('.', SETTINGS.length()) < 0
          && !keys.contains(key)) throw new IllegalArgumentException("Unknown setting " + key);
    return new Gson().fromJson(tree, PrismConfig.class);
  }

  /** 設定の値を config.json の値にする。数は config.json を読むときと同じ形で持つ。 */
  private static JsonElement json(RecordComponent component, String value) {
    Class<?> type = component.getType();
    if (type == String.class) return new JsonPrimitive(value);
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
    // 網の幅・精度・効用・学習率・勾配の上限は ModelConfig が、収集の率は Sampling と Behaviour が、p0 の係数は PriorSettings が確かめる
    modelConfig(Device.cpu(), hidden);
    sampling();
    behaviour();
    priorSettings();
    if (gamesPerCohort < 1 || gamesPerCohort > MAX_GAMES_PER_COHORT || iterations < 1)
      throw new IllegalArgumentException("Invalid run size");
    if (maxActiveGames < 1
        || advanceWorkers < 1
        || encoders < 1
        || slots < 1
        || readyBatches < 1
        || inferBatch < 1
        || multiTransitionBatch < 1
        || batchWaitMicros < 0
        || rowWriters < 1) throw new IllegalArgumentException("Invalid collection settings");
    if (memoryCohorts < 1
        || !(memoryDecay > 0 && memoryDecay <= 1)
        || keepCohorts < memoryCohorts
        || fitSteps < 0
        || distillSteps < 0
        || !(fitEpochs > 0)
        || !(distillEpochs > 0)
        || batchSize < 1
        || decodeThreads < 1) throw new IllegalArgumentException("Invalid fit settings");
    if (!(lrPeak > 0) || !(lrFloor >= 0 && lrFloor <= lrPeak))
      throw new IllegalArgumentException("Invalid learning rates " + lrPeak + "/" + lrFloor);
    if (!(lambdaDiscard > 0)
        || !(lambdaCommit > 0)
        || !(kappa >= 0)
        || !(certifyTolerance >= 0)
        || !(certifyMaxOutsideShare >= 0 && certifyMaxOutsideShare <= 1))
      throw new IllegalArgumentException("Invalid loss or certification settings");
    if (!KL_UNIFORM.equals(klReference) && !KL_PRIOR.equals(klReference))
      throw new IllegalArgumentException(
          "klReference must be " + KL_UNIFORM + " or " + KL_PRIOR + ": " + klReference);
    if (!LEARNER_CROSSFIT.equals(learner) && !LEARNER_SINGLE.equals(learner))
      throw new IllegalArgumentException(
          "learner must be " + LEARNER_CROSSFIT + " or " + LEARNER_SINGLE + ": " + learner);
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
    if (!phi.equals(PHI_GRP) && !phi.equals(PHI_SCORE_RANK))
      throw new IllegalArgumentException(
          "phi must be " + PHI_GRP + " or " + PHI_SCORE_RANK + ": " + phi);
    if (grpBatch < 1 || evalEvery < 0 || evalGames < 1)
      throw new IllegalArgumentException("Invalid Φ or evaluation settings");
  }

  /**
   * train の設定を確かめる（{@link #validate} に加えて、Φ が決まっていること）。phi = grp なら grpCheckpoint が要る。黙って点数の順位の代用に
   * 落ちると、ラベル・根・認証・保留の指標が全て別の目的の値になるので、代用は phi = score-rank と明示したときだけ許す。evaluate は Φ を使わない
   * ので {@link #validate} だけを呼ぶ。
   */
  public void validateTraining() {
    validate();
    if (phi.equals(PHI_GRP) && grpCheckpoint.isEmpty())
      throw new IllegalArgumentException(
          "phi = grp needs grpCheckpoint (set phi = \"" + PHI_SCORE_RANK + "\" only for tests)");
  }

  /** 収集の率。 */
  public Sampling sampling() {
    return new Sampling(
        holdoutFraction,
        labelRateDiscardEarly,
        labelRateDiscardMid,
        labelRateDiscardLate,
        labelRateCommit,
        maxCandidates,
        measureAllDiscards,
        epsilonC,
        rollinRateDiscard,
        rollinRateCommit,
        epsilonQ,
        rhoBranch,
        maxCopyLabels);
  }

  /** 収集の全席の行動方策 b = (1 − ε_b)·貪欲 + ε_b/n。 */
  public Behaviour behaviour() {
    return new Behaviour(epsilonB);
  }

  /** 方策の記憶（版の重みの平均）の決め方。 */
  public Averaging policyMemory() {
    return new Averaging(averaging, averageHalfLife, averagePower, averageFrom);
  }

  /** 学習の網を全データの生徒 1 本で作るか（{@value #LEARNER_SINGLE}）。 */
  public boolean singleStudent() {
    return LEARNER_SINGLE.equals(learner);
  }

  /** データの記憶: 年齢 age（最新のコホートが 0）のコホートのラベル行のうち使う割合 memoryDecay^age。 */
  public double memoryKeep(int age) {
    return Math.pow(memoryDecay, age);
  }

  /** 規則事前 p0 の係数。 */
  public PriorSettings priorSettings() {
    return new PriorSettings(priorAlpha, priorGamma);
  }

  /**
   * 行が rows 本のときの歩数: 上限 cap と「周回数 epochs × 行 ÷ バッチ」の小さい方（cap が 0 か行がなければ 0、それ以外は 1 以上）。雑音の大きい
   * 教師を何周もすると覚え込んで弱くなる（E1: 分岐行を 3.7 周で全ての損失が悪化、0.6 周なら悪化なし）ので、1 行を見る回数を周回数で抑える。
   */
  public static int steps(int cap, double epochs, int rows, int batchSize) {
    if (cap == 0 || rows == 0) return 0;
    long byEpochs = (long) Math.ceil(epochs * rows / batchSize);
    return (int) Math.max(1, Math.min(cap, byEpochs));
  }

  /** 歩数 steps の当てはめの設定（生徒と蒸留の歩数は {@link #steps} で決める）。 */
  public FitSettings fitSettings(int steps) {
    return new FitSettings(
        steps,
        batchSize,
        lrPeak,
        lrFloor,
        gradientClip,
        lambdaDiscard,
        lambdaCommit,
        kappa,
        KL_UNIFORM.equals(klReference));
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

  /** 幅 hidden の網の設定。初期値の種は seed + 100、weight decay は 0。 */
  public ModelConfig modelConfig(Device device, int hidden) {
    return new ModelConfig(
        hidden,
        // 1 歩のバッチを容量区分ごとに 1 回で通す
        Math.min(4096, batchSize),
        (float) lrPeak,
        0f,
        gradientClip,
        seed + 100,
        inferBatch,
        DecisionComputePrecision.valueOf(precision),
        device.isGpu() ? DecisionTensorTransfer.PINNED_BUFFER : DecisionTensorTransfer.DIRECT_BUFFER,
        EpsilonUtilityProfile.valueOf(utility));
  }
}
