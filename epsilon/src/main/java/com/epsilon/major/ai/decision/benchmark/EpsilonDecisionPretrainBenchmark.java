package com.epsilon.major.ai.decision.benchmark;

import ai.djl.Model;
import com.epsilon.config.settings.DecisionComputePrecision;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.config.settings.SettingsLoader;
import com.epsilon.core.Action;
import com.epsilon.core.Tile;
import com.epsilon.major.ai.decision.input.DecisionActionRouteEncoder;
import com.epsilon.major.ai.decision.input.DecisionBucket;
import com.epsilon.major.ai.decision.input.DecisionFeatureCodec;
import com.epsilon.major.ai.decision.input.DecisionHostBatch;
import com.epsilon.major.ai.decision.input.DecisionHostInputs;
import com.epsilon.major.ai.decision.input.DecisionInputSchema;
import com.epsilon.major.ai.decision.input.DecisionInputWriter;
import com.epsilon.major.ai.decision.input.DecisionTrainingTargets;
import com.epsilon.major.ai.decision.training.EpsilonDecisionPretrainer;
import com.epsilon.major.ai.network.NetworkDevices;
import com.epsilon.major.ai.network.NetworkFactory;
import com.epsilon.major.config.settings.DecisionPretrainSettings;
import com.epsilon.major.config.settings.DecisionSettings;
import com.epsilon.runtime.DecisionExecutionContext;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * 実際の学習デバイス上で、事前学習の転送・順伝播・逆伝播・更新処理の速度を比較する。
 *
 * <p>計測行は応答局面と自摸局面を交互に生成し、一行内では方策グラフ上で両立する行動だけを使う。これにより、架空の混合集合から生じる
 * 非正規化方策や無効な勾配を測定へ混ぜず、転送・順伝播・逆伝播・オプティマイザー同期を本番と同じ経路で比較する。
 */
public final class EpsilonDecisionPretrainBenchmark {

  private EpsilonDecisionPretrainBenchmark() {}

  /**
   * 指定構成をウォームアップ後に計測し、本番学習パスの処理速度を返す。
   *
   * @param tensorTransfer ホスト側の連続バッファの転送方式
   * @param optimizerBatchRows 一パラメーター更新を構成する総行数
   * @param maximumDeviceBatchRows 一デバイス順伝播へ入れる最大行数
   * @param warmupSteps JIT・メモリ割り当て処理を安定させる非計測更新段階数
   * @param measuredSteps 指標へ含める更新段階数
   * @param legalActionCapacity 人工的なバッチの行動容量区分幅
   * @param actionTransitionCapacity 一行動あたりの遷移容量区分幅
   * @return デバイス構成、実測時間、損失、処理速度を含むレポート
   * @throws IOException 学習ワーカーの実行に失敗した場合
   */
  public static Report run(
      DecisionTensorTransfer tensorTransfer,
      int optimizerBatchRows,
      int maximumDeviceBatchRows,
      int warmupSteps,
      int measuredSteps,
      int legalActionCapacity,
      int actionTransitionCapacity)
      throws IOException {
    return run(
        tensorTransfer,
        optimizerBatchRows,
        maximumDeviceBatchRows,
        warmupSteps,
        measuredSteps,
        legalActionCapacity,
        actionTransitionCapacity,
        com.epsilon.major.config.settings.EpsilonSettings.defaults());
  }

  public static Report run(
      DecisionTensorTransfer tensorTransfer,
      int optimizerBatchRows,
      int maximumDeviceBatchRows,
      int warmupSteps,
      int measuredSteps,
      int legalActionCapacity,
      int actionTransitionCapacity,
      SettingsLoader config)
      throws IOException {
    if (warmupSteps < 1 || measuredSteps < 1) {
      throw new IllegalArgumentException("pretrain benchmark steps must be positive");
    }
    DecisionBucket bucket = new DecisionBucket(legalActionCapacity, actionTransitionCapacity);
    DecisionHostBatch batch = syntheticBatch(optimizerBatchRows, bucket, config);
    NetworkDevices devices =
        NetworkFactory.getLearnerDevices(
            config.bind(com.epsilon.config.settings.DeviceSettings.class));
    DecisionPretrainSettings settings =
        config
            .bind(DecisionPretrainSettings.class)
            .forBenchmark(optimizerBatchRows, maximumDeviceBatchRows, tensorTransfer);
    try (DecisionExecutionContext executionContext = new DecisionExecutionContext();
        Model model =
            NetworkFactory.createDecisionModel(
                devices.primary(),
                true,
                config.bind(DecisionSettings.class).hidden(),
                config.bind(DecisionSettings.class).utilityProfile());
        EpsilonDecisionPretrainer pretrainer =
            EpsilonDecisionPretrainer.open(
                model, settings, devices, executionContext, config.bind(DecisionSettings.class))) {
      pretrainer.trainEpoch(Collections.nCopies(warmupSteps, batch));
      EpsilonDecisionPretrainer.EpochMetrics metrics =
          pretrainer.trainEpoch(Collections.nCopies(measuredSteps, batch));
      return new Report(
          tensorTransfer,
          settings.computePrecision(),
          devices.toString(),
          pretrainer.activeDevices().toString(),
          optimizerBatchRows,
          maximumDeviceBatchRows,
          warmupSteps,
          measuredSteps,
          legalActionCapacity,
          actionTransitionCapacity,
          metrics.rows(),
          metrics.elapsedMillis(),
          metrics.rowsPerSecond(),
          metrics.loss(),
          metrics.behaviorCloningLoss(),
          metrics.valueLoss());
    }
  }

  public static DecisionHostBatch syntheticBatch(int rows, DecisionBucket bucket) {
    return syntheticBatch(
        rows, bucket, com.epsilon.major.config.settings.EpsilonSettings.defaults());
  }

  public static DecisionHostBatch syntheticBatch(
      int rows, DecisionBucket bucket, SettingsLoader config) {
    DecisionHostInputs inputs = new DecisionHostInputs(rows, bucket);
    DecisionTrainingTargets targets = new DecisionTrainingTargets(rows, bucket);
    float actionProbability = 1.0f / bucket.legalActionCapacity();
    float[] policy = new float[bucket.legalActionCapacity()];
    java.util.Arrays.fill(policy, actionProbability);
    List<Action> responseActions = syntheticActions(true, bucket.legalActionCapacity());
    List<Action> turnActions = syntheticActions(false, bucket.legalActionCapacity());
    DecisionActionRouteEncoder.Scratch routeScratch = new DecisionActionRouteEncoder.Scratch();
    for (int row = 0; row < rows; row++) {
      DecisionInputWriter writer = inputs.writer(row);
      List<Action> legalActions = row % 2 == 0 ? responseActions : turnActions;
      for (int action = 0; action < bucket.legalActionCapacity(); action++) {
        Action legalAction = legalActions.get(action);
        Action.Type actionType = legalAction.type();
        writer.action(
            action,
            DecisionInputSchema.ActionInt.TYPE,
            DecisionFeatureCodec.actionType(actionType));
        writer.action(
            action, DecisionInputSchema.ActionInt.ID, DecisionFeatureCodec.actionId(legalAction));
        writer.action(
            action,
            DecisionInputSchema.ActionInt.GROUP,
            DecisionFeatureCodec.actionGroup(actionType.group()));
        writer.action(
            action,
            DecisionInputSchema.ActionInt.PRIMARY_TILE,
            Tile.isValidType(legalAction.tileType()) ? legalAction.tileType() + 1 : 0);
        writer.action(
            action,
            DecisionInputSchema.ActionInt.TILE_SELECTION,
            legalAction.tileSelection().ordinal() + 1);
        if (actionType == Action.Type.CHI) {
          int[] chiTileTypes = legalAction.chiTileTypes();
          writer.action(action, DecisionInputSchema.ActionInt.CHI_BASE, chiTileTypes[0] + 1);
          writer.action(
              action,
              DecisionInputSchema.ActionInt.CHI_CALLED_POSITION,
              legalAction.tileType() - chiTileTypes[0] + 1);
        }
        writer.action(
            action,
            DecisionInputSchema.ActionInt.DISCARD_IDENTITY,
            legalAction.discardIdentityIndex() + 1);
        boolean discard = actionType == Action.Type.DAHAI || actionType == Action.Type.RIICHI_DAHAI;
        DecisionInputSchema.ActionTransitionKind transitionKind =
            discard
                ? DecisionInputSchema.ActionTransitionKind.DISCARD
                : switch (actionType) {
                  case DAIMINKAN, ANKAN, KAKAN ->
                      DecisionInputSchema.ActionTransitionKind.RINSHAN_PENDING;
                  case TSUMO_AGARI, RON_AGARI, KYUSHU_KYUHAI ->
                      DecisionInputSchema.ActionTransitionKind.TERMINAL;
                  default -> DecisionInputSchema.ActionTransitionKind.IDENTITY;
                };
        writer.transition(
            action, 0, DecisionInputSchema.ActionTransitionInt.KIND, transitionKind.ordinal() + 1);
        writer.transition(
            action,
            0,
            DecisionInputSchema.ActionTransitionInt.DISCARD_CONTEXT,
            (discard
                        ? DecisionInputSchema.DiscardContext.TURN
                        : DecisionInputSchema.DiscardContext.NONE)
                    .ordinal()
                + 1);
        if (discard) {
          Action canonicalDiscard =
              actionType == Action.Type.RIICHI_DAHAI
                  ? Action.dahai(legalAction.tileType(), legalAction.tileSelection())
                  : legalAction;
          writer.transition(
              action,
              0,
              DecisionInputSchema.ActionTransitionInt.DISCARD_ACTION_ID,
              DecisionFeatureCodec.actionId(canonicalDiscard));
          writer.transition(
              action,
              0,
              DecisionInputSchema.ActionTransitionInt.DISCARD_TILE,
              legalAction.tileType() + 1);
          writer.transition(
              action,
              0,
              DecisionInputSchema.ActionTransitionInt.TILE_SELECTION,
              legalAction.tileSelection().ordinal() + 1);
        }
        writer.transition(
            action,
            0,
            DecisionInputSchema.ActionTransitionInt.RESULTING_RON_FURITEN_KIND,
            DecisionInputSchema.RonFuritenKind.NONE.ordinal() + 1);
        writer.transition(action, 0, DecisionInputSchema.ActionTransitionInt.PRESENT, 1);
        for (int tileType = 0; tileType < Tile.NUM_TILE_TYPES; tileType++) {
          writer.transitionTile(
              action,
              0,
              tileType,
              DecisionFeatureCodec.transitionTile(
                  0,
                  0,
                  false,
                  false,
                  false,
                  discard && tileType == legalAction.tileType(),
                  false,
                  false));
        }
      }
      DecisionActionRouteEncoder.encode(legalActions, writer, routeScratch);
      float valueTarget =
          config.bind(DecisionSettings.class).utilityProfile().utilityForRank(row % 4);
      targets.writeRow(
          row,
          bucket.legalActionCapacity(),
          row % bucket.legalActionCapacity(),
          policy,
          policy,
          valueTarget,
          0.0f,
          0.0f,
          1.0f);
    }
    return DecisionHostBatch.takeEncodedTrainingBatch(
        rows,
        bucket,
        inputs.denseCategories(),
        inputs.denseNumerics(),
        targets.categoricalSlab(),
        targets.numericSlab());
  }

  private static List<Action> syntheticActions(boolean response, int count) {
    ArrayList<Action> actions = new ArrayList<>(count);
    if (response) {
      actions.add(Action.pass());
      if (count > 1) {
        actions.add(Action.ronAgari());
      }
    } else {
      actions.add(Action.tsumoAgari());
      if (count > 1) {
        actions.add(Action.kyushuKyuhai());
      }
    }
    for (int actionId = 0;
        actions.size() < count && actionId < Action.ACTION_SPACE_SIZE;
        actionId++) {
      Action candidate = Action.fromIndex(actionId);
      if (candidate.type().isResponse() == response && !containsActionId(actions, actionId)) {
        actions.add(candidate);
      }
    }
    if (actions.size() != count) {
      throw new IllegalArgumentException("insufficient synthetic actions for bucket: " + count);
    }
    return actions;
  }

  private static boolean containsActionId(List<Action> actions, int actionId) {
    for (Action action : actions) {
      if (action.toIndex() == actionId) {
        return true;
      }
    }
    return false;
  }

  /**
   * 事前学習小規模な性能測定の再現条件と結果。
   *
   * @param tensorTransfer ホストからデバイスへの転送方式
   * @param computePrecision 順伝播/逆伝播の計算精度
   * @param availableDevices 検出された学習器デバイス
   * @param activeDevices 計測で実際に使用したデバイス
   * @param optimizerBatchRows 一更新段階の総行数
   * @param maximumDeviceBatchRows 一デバイス順伝播の最大行数
   * @param warmupSteps 計測前に実行した更新段階数
   * @param measuredSteps 計測したパラメーター更新数
   * @param legalActionCapacity 行動容量区分幅
   * @param actionTransitionCapacity 一行動あたりの遷移容量区分幅
   * @param measuredRows 計測区間で処理した総行数
   * @param elapsedMillis 計測区間の経過時間
   * @param rowsPerSecond 全デバイス合算の処理行数/秒
   * @param loss 総目的関数の更新段階平均
   * @param behaviorCloningLoss 方策教師損失の更新段階平均
   * @param valueLoss 価値損失の更新段階平均
   */
  public record Report(
      DecisionTensorTransfer tensorTransfer,
      DecisionComputePrecision computePrecision,
      String availableDevices,
      String activeDevices,
      int optimizerBatchRows,
      int maximumDeviceBatchRows,
      int warmupSteps,
      int measuredSteps,
      int legalActionCapacity,
      int actionTransitionCapacity,
      int measuredRows,
      double elapsedMillis,
      double rowsPerSecond,
      double loss,
      double behaviorCloningLoss,
      double valueLoss) {}
}
