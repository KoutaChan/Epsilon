package com.epsilon.prism.settings;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.SeatingSettings;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.runtime.RuntimeSettings;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.grp.FinalRound;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.ClassTables;
import com.epsilon.prism.core.DecisionClass;
import com.epsilon.prism.eval.OpponentRules;
import com.epsilon.prism.grp.LateSettings;
import com.epsilon.prism.grp.PhiSettings;
import com.epsilon.prism.noise.NoiseSettings;
import com.epsilon.prism.variant.PrismVariant;
import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * PRISM の run の設定の全体。run の途中では、設定の記録の 1 行がこの record の値を全て持つ。
 *
 * <p>セクションをまたいで次のことを確かめる。
 *
 * <ul>
 *   <li>判断のクラスをキーにした表（ロールインの率・損失の温度・ラベル行の率）は、そのルールのクラスをちょうど 1 回ずつ持つ。
 *   <li>専用のカーネルを使って GPU で推論するなら、計算の型は BFLOAT16 にする。
 *   <li>評価の卓の相手は、そのルールで打てるモデルの系列、PRISM のチェックポイント、系譜のバージョン（王者、version:N、version:-d）の
 *       どれか。
 *   <li>牌山ごとに 1 半荘打つ卓（BY_GAME）は、決めた数の牌山を全て打つ。牌山の数は席の数の倍数にする。
 *   <li>全ての席の割り当てを打つ卓（EVERY_PLACEMENT）では、どの分け方も候補と相手の 2 スロットに分け、2 つの席の数の和が卓の
 *       席の数になる。
 *   <li>PRISM の王者との決闘は「良い」の判定だけで昇格させ、確かめの卓を打たないので、確かめの卓を書いた卓は拒む。
 *   <li>LATE のモードを使い始める設定は、最後の局を数え上げられるルール（{@link FinalRound#VARIANTS}）でだけ受け付ける。
 * </ul>
 *
 * @param run run のルールと乱数の種
 * @param network ゼロから学習するバージョン 1 のネットワークの形
 * @param phi 局末の値 Φ の出どころ
 * @param schedule 反復の数とコホートの大きさ
 * @param inference 収集と評価の推論
 * @param table 収集と評価の卓
 * @param runtime 実行の環境
 * @param learner 学習器
 * @param behaviour 行動方策
 * @param labels ラベル行と中立ロールイン
 * @param fit 生徒の学習
 * @param memory 方策の記憶
 * @param noise ラベルのノイズの対策
 * @param evaluation 評価と決闘
 * @param diagnostics 診断
 */
public record PrismSettings(
    @Key(doc = "run のルールと乱数の種", change = Change.FIXED) RunSettings run,
    @Key(doc = "ゼロから学習するバージョン 1 のネットワークの形", value = "V2", change = Change.FIXED) NetworkSettings network,
    @Key(doc = "局末の値 Φ の出どころ", value = "SELF_GRP", change = Change.FIXED) PhiSettings phi,
    @Key(doc = "反復の数とコホートの大きさ", change = Change.FIT) ScheduleSettings schedule,
    @Key(doc = "収集と評価の推論", change = Change.MACHINE) InferenceSettings inference,
    @Key(doc = "収集と評価の卓", change = Change.MACHINE) TableSettings table,
    @Key(doc = "実行の環境", change = Change.MACHINE) RuntimeSettings runtime,
    @Key(doc = "学習器", change = Change.MACHINE) LearnerSettings learner,
    @Key(doc = "行動方策", change = Change.COLLECT) BehaviourSettings behaviour,
    @Key(doc = "ラベル行と中立ロールイン", change = Change.COLLECT) LabelSettings labels,
    @Key(doc = "生徒の学習", change = Change.FIT) FittingSettings fit,
    @Key(doc = "方策の記憶", value = "NONE", change = Change.FIT) MemorySettings memory,
    @Key(doc = "ラベルのノイズの対策", change = Change.COLLECT) NoiseSettings noise,
    @Key(doc = "評価と決闘", change = Change.SCHEDULE) EvaluationSettings evaluation,
    @Key(doc = "診断", change = Change.MACHINE) DiagnosticsSettings diagnostics) {

  public PrismSettings {
    Objects.requireNonNull(run, "run");
    Objects.requireNonNull(network, "network");
    Objects.requireNonNull(phi, "phi");
    Objects.requireNonNull(schedule, "schedule");
    Objects.requireNonNull(inference, "inference");
    Objects.requireNonNull(table, "table");
    Objects.requireNonNull(runtime, "runtime");
    Objects.requireNonNull(learner, "learner");
    Objects.requireNonNull(behaviour, "behaviour");
    Objects.requireNonNull(labels, "labels");
    Objects.requireNonNull(fit, "fit");
    Objects.requireNonNull(memory, "memory");
    Objects.requireNonNull(noise, "noise");
    Objects.requireNonNull(evaluation, "evaluation");
    Objects.requireNonNull(diagnostics, "diagnostics");
    Variant variant = run.variant();
    PrismVariant module = moduleOf(variant);
    requireClasses("labels.rollin", labels.rollin().keySet(), variant);
    requireClasses("fit.temperature", fit.temperature().keySet(), variant);
    requireClasses("noise.rates.table", noise.rates().table().keySet(), variant);
    requireSubset("noise.rates.allActions", noise.rates().allActions(), variant);
    requireSubset("noise.rates.open", noise.rates().open().keySet(), variant);
    requireSubset("labels.copies", labels.copies().keySet(), variant);
    for (Map.Entry<DecisionClass, CopySettings> copy : labels.copies().entrySet()) {
      requireSubset("labels.copies." + copy.getKey() + ".laterRates", copy.getValue().laterRates().keySet(), variant);
    }
    if (inference.kernels() == Kernels.NATIVE
        && inference.networkDevices().primary().isGpu()
        && inference.precision() != Precision.BFLOAT16) {
      throw new InvalidSetting("inference.precision", "PRISM's native kernels compute in BFLOAT16");
    }
    requireEvaluation(evaluation, module.opponents(), variant);
    if (phi.lateSettings().map(LateSettings::from).orElse(0) > 0 && !FinalRound.VARIANTS.contains(variant)) {
      throw new InvalidSetting(
          "phi.late.from", "the late mode enumerates the final round of " + FinalRound.VARIANTS + " only, not " + variant);
    }
  }

  /** run のルールでの順位の効用を返す。効用は入力の形式が決める。 */
  public RankUtility utility() {
    return module().input().utility();
  }

  /** run のルールに対応する PRISM のモジュールを返す。 */
  public PrismVariant module() {
    return moduleOf(run.variant());
  }

  private static PrismVariant moduleOf(Variant variant) {
    try {
      return PrismVariant.of(variant);
    } catch (IllegalStateException missing) {
      throw new InvalidSetting("run.variant", missing.getMessage());
    }
  }

  /** 表か集合に書いた判断のクラスが、全てそのルールのクラスであることを確かめる。全てのクラスを書かなくてよい表と集合に使う。 */
  private static void requireSubset(String component, Set<DecisionClass> classes, Variant variant) {
    if (!ClassTables.classes(variant).containsAll(classes)) {
      throw new InvalidSetting(component, "the classes must be classes of " + variant);
    }
  }

  private static void requireClasses(String component, Set<DecisionClass> keys, Variant variant) {
    String mismatch = ClassTables.mismatch(keys, variant);
    if (mismatch != null) {
      throw new InvalidSetting(component, mismatch);
    }
  }

  private static void requireEvaluation(EvaluationSettings evaluation, OpponentRules rules, Variant variant) {
    for (Map.Entry<String, TableSpec> entry : evaluation.tables().entrySet()) {
      String path = "evaluation.tables." + entry.getKey();
      TableSpec table = entry.getValue();
      if (table.confirm() != null) {
        throw new InvalidSetting(
            path + ".confirm", "a PRISM duel promotes on its verdict alone and plays no confirm table");
      }
      switch (table.seating()) {
        case SeatingSettings.ByGame byGame -> {
          if (!(table.stop() instanceof StopSettings.Fixed)) {
            throw new InvalidSetting(path + ".stop", "a BY_GAME table plays all its walls, so it stops FIXED");
          }
          if (table.walls() % variant.seats() != 0) {
            throw new InvalidSetting(
                path + ".walls",
                "a BY_GAME table plays a multiple of " + variant.seats() + " walls, not " + table.walls());
          }
        }
        case SeatingSettings.EveryPlacement every -> {
          try {
            every.splits(variant);
          } catch (IllegalArgumentException mismatch) {
            throw new InvalidSetting(path + ".seating.splits", mismatch.getMessage());
          }
        }
      }
      switch (TableSpec.kind(table.opponent())) {
        case HEURISTIC -> requireSeries(path, Series.RULE_BOT, rules);
        case CHECKPOINT -> {
          Series series = Series.parse(table.opponent()).series();
          if (series != Series.PRISM) {
            requireSeries(path, series, rules);
          }
        }
        case CHAMPION, VERSION, DISTANCE -> {}
      }
    }
  }

  private static void requireSeries(String path, Series series, OpponentRules rules) {
    if (!rules.series().contains(series)) {
      throw new InvalidSetting(path + ".opponent", "this variant plays " + rules.series() + ", not " + series);
    }
  }
}
