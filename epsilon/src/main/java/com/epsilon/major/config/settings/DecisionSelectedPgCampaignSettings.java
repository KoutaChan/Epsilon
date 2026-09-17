package com.epsilon.major.config.settings;

import com.epsilon.config.settings.Default;
import com.epsilon.config.settings.MultipleOf;
import com.epsilon.config.settings.NonNegative;
import com.epsilon.config.settings.Positive;
import com.epsilon.config.settings.Range;
import com.epsilon.config.settings.Setting;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.config.settings.SettingsPrefix;

/**
 * {@code train-decision} が実行する選択行動の方策勾配による学習一連の学習実行の設定です。
 *
 * <p>{@code macrosPerDuel}ごとに現在の変更不可対局収集用の採用モデルと評価します。PROMOTEDなら候補を次の対局実行処理
 * 採用モデルとして同じオプティマイザーを引き継ぎ、UNRESOLVEDなら同じ候補学習の継続状態を継続します。HARMFULまたは学習検証条件違反は 一連の学習実行を終了します。{@code
 * maximumMacros}は今回の呼び出しで追加実行する収集・更新単位上限です。 対戦評価区間の途中で停止した場合も、次回は保存した同じ評価間隔から再開します。
 *
 * @param macrosPerDuel 対局実行処理対戦評価間に収集・更新する収集・更新単位数
 * @param maximumMacros 今回の呼び出しで追加実行する収集・更新単位上限
 * @param gamesPerMacro 1 収集・更新単位で生成する対局数
 * @param microBatchSize 全学習器デバイスを合わせた1 一括処理の最大サンプル行数
 * @param maximumDeviceTransitionCells 1 デバイスが同時に保持する行×行動×遷移格納枠上限
 * @param ppoEpochs 同じ新しく生成した収集・更新単位を固定して再評価するPPO エポック数
 * @param optimizerStepsPerEpoch 各PPO エポックで新しく生成した収集・更新単位全体を分割するパラメーター更新数
 * @param optimizer 方策 KL制御と固定価値オプティマイザーの設定
 * @param causalTraceLambda 価値/方策の価値計算用と方策更新用の二つの遷移列で使うスカラー値のトレース係数
 * @param dahaiWeight 打牌判断機会を学習へ採用する確率重み
 * @param riichiWeight RIICHI対DAMA 判断機会を学習へ採用する確率重み
 * @param reactionWeight 応答判断機会を学習へ採用する確率重み
 * @param policyUpdateClipRange {@code piCurrent/piRollout} に適用するPPO クリップ幅
 * @param explorationCreditMix 探索適用後の探索へ残す選択行動学習への寄与の線形混合率
 * @param entropyCoefficient 現在の合法最終的な行動分布のエントロピーを支える固定係数
 * @param maximumOptimizerShardMeanKl オプティマイザー分割ファイルに許容する {@code KL(piRollout || piCurrent)} 平均
 * @param debugActorValidationEnabled 更新前後の方策診断を有効にするなら {@code true}
 * @param debugFinalAuditEnabled 収集・更新単位更新後の全サンプル監査を有効にするなら {@code true}
 * @param finalAuditSampleLimit 監査で一度に読む最大まとまり行数
 * @param runSeedBase 一連の学習実行の決定的乱数シード基点。0なら実行 UUIDから生成し、非0ならその値を固定使用する
 */
@SettingsPrefix("epsilon.decision.train.selectedPg")
public record DecisionSelectedPgCampaignSettings(
    @Setting("macrosPerDuel") @Default("10") @Positive int macrosPerDuel,
    @Setting("maximumMacros")
        @Default("50")
        @Range(min = 0.0, minInclusive = false, max = 1_000_000.0, maxInclusive = false)
        int maximumMacros,
    @Setting("gamesPerMacro") @Default("8192") @Positive @MultipleOf(4) int gamesPerMacro,
    @Setting("microBatchSize") @Default("512") @Positive int microBatchSize,
    @Setting("maximumDeviceTransitionCells") @Default("65536") @Positive
        int maximumDeviceTransitionCells,
    @Setting("ppoEpochs") @Default("1") @Positive int ppoEpochs,
    @Setting("optimizerStepsPerEpoch") @Default("32") @Positive int optimizerStepsPerEpoch,
    @Setting("optimizer") OptimizerSettings optimizer,
    @Setting("causalTraceLambda") @Default("0.95") @Range(min = 0.0, max = 1.0)
        float causalTraceLambda,
    @Setting("dahaiWeight") @Default("1.0") @Range(min = 0.0, minInclusive = false, max = 1.0)
        float dahaiWeight,
    @Setting("riichiWeight") @Default("1.0") @Range(min = 0.0, minInclusive = false, max = 1.0)
        float riichiWeight,
    @Setting("reactionWeight") @Default("1.0") @Range(min = 0.0, minInclusive = false, max = 1.0)
        float reactionWeight,
    @Setting("policyUpdateClipRange")
        @Default("0.2")
        @Range(min = 0.0, minInclusive = false, max = 1.0, maxInclusive = false)
        float policyUpdateClipRange,
    @Setting("explorationCreditMix") @Range(min = 0.0, minInclusive = false, max = 1.0)
        float explorationCreditMix,
    @Setting("entropyCoefficient") @Default("0.001") @Range(min = 0.0, max = 0.05)
        float entropyCoefficient,
    @Setting("maximumOptimizerShardMeanKl") @Default("0.01") @NonNegative
        float maximumOptimizerShardMeanKl,
    @Setting("debugActorValidationEnabled") @Default("false") boolean debugActorValidationEnabled,
    @Setting("debugFinalAuditEnabled") @Default("false") boolean debugFinalAuditEnabled,
    @Setting("finalAuditSampleLimit") @Default("4096") @Positive int finalAuditSampleLimit,
    @Setting("runSeedBase") @Default("0") long runSeedBase) {

  /** 全席を一度ずつ担当する対応をそろえた自己対局の席順の組合せ数。 */
  public static final int SEAT_ROTATIONS = 4;

  /** 一連の学習実行予算、無作為抽出重み、KLの相互制約を検証する。 */
  public DecisionSelectedPgCampaignSettings {
    try {
      Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    } catch (ArithmeticException overflow) {
      throw new IllegalArgumentException("selected PG total optimizer steps overflow", overflow);
    }
    if (optimizerStepsPerEpoch > 1
        && (Float.compare(dahaiWeight, riichiWeight) != 0
            || Float.compare(dahaiWeight, reactionWeight) != 0)) {
      throw new IllegalArgumentException(
          "selected PG multi-step PPO epoch currently requires equal opportunity weights");
    }
    if (optimizer == null) {
      throw new IllegalArgumentException("selected PG optimizer settings must not be null");
    }
    long plannedUpdates = optimizer.actorKlControl().targetDecay().plannedOptimizerSteps();
    long calibrationUpdates =
        (long) optimizer.actorKlControl().warmupMacros()
            * Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
    if (plannedUpdates > 0L && plannedUpdates <= calibrationUpdates) {
      throw new IllegalArgumentException(
          "Actor KL target plan must extend beyond calibration updates");
    }
  }

  /** 保存中の対戦評価区間からの再開も含め、今回の呼び出しで実行しうる最大対戦評価回数。 */
  public int maximumDuels() {
    return Math.ceilDiv(maximumMacros, macrosPerDuel);
  }

  /** 一つの対戦評価区間で生成する対局数。 */
  public int trainingGamesPerDuel() {
    return Math.multiplyExact(macrosPerDuel, gamesPerMacro);
  }

  /** 今回の呼び出しで生成する最大学習対局数。 */
  public int maximumTrainingGames() {
    return Math.multiplyExact(maximumMacros, gamesPerMacro);
  }

  /** 一つの新しく生成した収集・更新単位から実行する方策・価値オプティマイザー確定総数。 */
  public int optimizerStepsPerMacro() {
    return Math.multiplyExact(ppoEpochs, optimizerStepsPerEpoch);
  }

  /** この設定項目に対応する、系列に同梱された既定値を返す。 */
  public static DecisionSelectedPgCampaignSettings defaults() {
    return EpsilonSettings.defaults().bind(DecisionSelectedPgCampaignSettings.class);
  }

  /** 選択行動の方策勾配学習の方策・価値オプティマイザー設定。 */
  public record OptimizerSettings(
      @Setting("valueLearningRate") @Default("1.25e-6") @Positive float valueLearningRate,
      @Setting("actorKlControl") ActorKlControlSettings actorKlControl) {

    /** 入れ子の設定の必須性を明示する。 */
    public OptimizerSettings {
      if (actorKlControl == null) {
        throw new IllegalArgumentException(
            "selected PG actor KL control settings must not be null");
      }
    }
  }

  /** 収集・更新単位で方策 LRを実測KLへ追従させる設定。ウォームアップで決めた基準目標にtargetDecayの倍率を掛ける。 */
  public record ActorKlControlSettings(
      @Setting("initialLearningRate") @Default("1.25e-6") @Positive float initialLearningRate,
      @Setting("hardMinimumLearningRate") @Default("2.5e-7") @Positive
          float hardMinimumLearningRate,
      @Setting("hardMaximumLearningRate") @Default("5.0e-6") @Positive
          float hardMaximumLearningRate,
      @Setting("warmupMacros") @Default("3") @Positive int warmupMacros,
      @Setting("medianWindowMacros") @Default("3") @Positive int medianWindowMacros,
      @Setting("targetMultiplier") @Default("1.25") @Range(min = 1.0) float targetMultiplier,
      @Setting("deadbandFactor") @Default("1.25") @Range(min = 1.0, minInclusive = false)
          float deadbandFactor,
      @Setting("maximumAdjustmentFactorPerMacro")
          @Default("1.02")
          @Range(min = 1.0, minInclusive = false)
          float maximumAdjustmentFactorPerMacro,
      @Setting("targetDecay") TargetKlDecaySettings targetDecay) {

    /** 学習率の上下限の順序を検証する。 */
    public ActorKlControlSettings {
      if (hardMinimumLearningRate > initialLearningRate
          || initialLearningRate > hardMaximumLearningRate) {
        throw new IllegalArgumentException(
            "selected PG Actor learning rates must satisfy hardMinimum <= initial <= hardMaximum");
      }
      if (targetDecay == null) {
        throw new IllegalArgumentException("Actor KL target plan must not be null");
      }
    }
  }

  /**
   * 初回に設定する方策目標KLの計画。実行収集・更新単位上限とは独立した受理オプティマイザー更新数を使う。
   *
   * @param plannedOptimizerSteps 累計予定更新数。0なら予備評価として較正後の目標を固定する
   * @param finalScale 初期目標に対する終端倍率。0.5は実験用初期値であり最適値ではない
   */
  public record TargetKlDecaySettings(
      @Setting("plannedOptimizerSteps") @Default("43200") @NonNegative long plannedOptimizerSteps,
      @Setting("finalScale") @Default("0.5") @Range(min = 0.0, minInclusive = false, max = 1.0)
          float finalScale) {}

  /** 指定された確定済み設定から、このレコードに対応する項目を読み込む。 */
  public static DecisionSelectedPgCampaignSettings load(SettingsLoader settings) {
    return settings.bind(DecisionSelectedPgCampaignSettings.class);
  }
}
