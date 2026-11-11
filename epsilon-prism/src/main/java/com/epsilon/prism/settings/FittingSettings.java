package com.epsilon.prism.settings;

import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.learn.FitSettings;
import com.epsilon.prism.learn.Selection;
import com.epsilon.prism.model.AuxiliaryLoss;
import java.util.Map;
import java.util.Objects;

/**
 * 生徒の学習のセクション。ステップ数は min(ステップ数の上限, ceil(エポック数 × 行数 ÷ バッチ))。学習率は始めの
 * {@link FitSettings#WARMUP} の割合のステップでピークまで上げ、cosine で下限まで下げる。損失の温度は、打牌のクラスで 1 つ、
 * ほかのクラスで 1 つの値だけを受け付ける。
 *
 * @param rows 生徒の行の選び方
 * @param maxSteps ステップ数の上限
 * @param epochs 1 行を見る回数の上限（エポック数）
 * @param batch 1 ステップの行数
 * @param lrPeak 学習率のピーク
 * @param lrFloor 学習率の下限
 * @param gradientClip 勾配のノルムの上限（0 なら切らない）
 * @param temperature 判断のクラスごとの差分整合の温度
 * @param auxOracle 読みのヘッドの重み
 * @param auxResult 候補の結果のヘッドの重み
 * @param keepCohorts 行を残すコホートの数（古いものから消す）
 */
public record FittingSettings(
    @Key(doc = "生徒の行の選び方", value = "LABELS_AND_ROOTS") Selection rows,
    @Key(doc = "ステップ数の上限", value = "2000") @Range(min = 0) int maxSteps,
    @Key(doc = "1 行を見る回数の上限（エポック数）", value = "1.0") @Range(min = 0, minInclusive = false) double epochs,
    @Key(doc = "1 ステップの行数", value = "4096") @Range(min = 1) int batch,
    @Key(doc = "学習率のピーク", value = "0.0012") @Range(min = 0, minInclusive = false) double lrPeak,
    @Key(doc = "学習率の下限", value = "0.00006") @Range(min = 0) double lrFloor,
    @Key(doc = "勾配のノルムの上限（0 なら切らない）", value = "0.5") @Range(min = 0) double gradientClip,
    @Key(doc = "判断のクラスごとの差分整合の温度", presence = Presence.REQUIRED)
        Map<DecisionClass, @Range(min = 0, minInclusive = false) Double> temperature,
    @Key(doc = "読みのヘッドの重み", value = "0.01") @Range(min = 0) double auxOracle,
    @Key(doc = "候補の結果のヘッドの重み", value = "0.01") @Range(min = 0) double auxResult,
    @Key(doc = "行を残すコホートの数（古いものから消す）", value = "16") @Range(min = 1) int keepCohorts) {

  public FittingSettings {
    Objects.requireNonNull(rows, "rows");
    temperature = ClassTables.copy(temperature);
    if (lrFloor > lrPeak) {
      throw new InvalidSetting("lrFloor", "lrFloor " + lrFloor + " exceeds lrPeak " + lrPeak);
    }
    Double discard = null;
    Double commitment = null;
    for (Map.Entry<DecisionClass, Double> entry : temperature.entrySet()) {
      Double first = entry.getKey().commitment() ? commitment : discard;
      if (first != null && !first.equals(entry.getValue())) {
        throw new InvalidSetting(
            "temperature",
            "the loss takes one temperature for the discard classes and one for the other classes, so "
                + entry.getKey() + " = " + entry.getValue() + " must equal " + first);
      }
      if (entry.getKey().commitment()) {
        commitment = entry.getValue();
      } else {
        discard = entry.getValue();
      }
    }
  }

  /** 行が {@code rows} 行のときのステップ数（上限か行数が 0 なら 0、それ以外は 1 以上）。 */
  public int steps(int rows) {
    if (maxSteps == 0 || rows == 0) {
      return 0;
    }
    long byEpochs = (long) Math.ceil(epochs * rows / batch);
    return (int) Math.max(1, Math.min(maxSteps, byEpochs));
  }

  /**
   * ステップ数と学習率の倍率を決めた、学習の設定を返す。
   *
   * @param steps ステップ数
   * @param learningRateScale 学習率のピークと下限に掛ける倍率
   */
  public FitSettings fit(int steps, double learningRateScale) {
    return new FitSettings(
        steps,
        batch,
        lrPeak * learningRateScale,
        lrFloor * learningRateScale,
        (float) gradientClip,
        lambda(false),
        lambda(true));
  }

  /** 補助のヘッドの重み。 */
  public AuxiliaryLoss auxiliary() {
    return new AuxiliaryLoss((float) auxOracle, (float) auxResult);
  }

  private double lambda(boolean commitment) {
    for (Map.Entry<DecisionClass, Double> entry : temperature.entrySet()) {
      if (entry.getKey().commitment() == commitment) {
        return entry.getValue();
      }
    }
    throw new IllegalStateException("No temperature for the " + (commitment ? "commitment" : "discard") + " classes");
  }
}
