package com.epsilon.classic.app;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.eval.OnWorseSettings;
import com.epsilon.agent.eval.StopSettings;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.learn.GrpTraining;
import com.epsilon.agent.play.TableSettings;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkDevices;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.classic.learn.Exploration;
import com.epsilon.classic.learn.KlLearningRate;
import com.epsilon.classic.learn.SelfPlayLearner;
import com.epsilon.mahjong.rules.Variant;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * classic のアプリ全体の設定。campaign の run では、設定の行の 1 行がこの record の全ての値を持つ。コマンドの引数は、コマンドと同じ
 * 名前のセクション（{@code [pretrain]} など）にも書ける。自己対局の macro とは、収集と学習を 1 回ずつ行う反復のこと。
 *
 * <p>セクションをまたぐ検査: campaign が予定で打つ評価の卓は、王者を替える決闘の卓だけ。ほかの卓は予定の間隔を 0 にして、決闘の判定から
 * 呼ぶ。王者との決闘で「悪い」が続いたときは、アラートを書いて続けるか一時停止するかのどちらかで、王者へは戻さない。
 *
 * @param run campaign のルール・種・始めのネットワーク・GRP
 * @param schedule macro の数と大きさ
 * @param inference 自己対局と評価の推論
 * @param table 自己対局と評価の卓
 * @param learner 学習器のデバイスと計算の型
 * @param selfPlay 自己対局の相手
 * @param explore 自己対局で手を選ぶ方策の探索
 * @param ppo 自己対局の学習器で方策を学ぶ設定
 * @param actorRate 方策の学習率の制御。途中の状態が持つ KL の数は、ウォームアップの macro の数と中央値を取る直近の macro の数で
 *     決まるので、run の途中では変えない
 * @param valueRate 価値の学び方
 * @param vtrace 価値の目標
 * @param batches 学習器のバッチの大きさ。バッチの分け方で変わるのは、勾配を足す順だけ
 * @param evaluation 王者との決闘と評価の卓
 * @param grp 牌譜で GRP を学習する設定
 */
public record ClassicSettings(
    @Key(doc = "campaign のルール・種・始めのネットワーク・GRP", change = Change.FIXED) Run run,
    @Key(doc = "macro の数と大きさ", change = Change.FIT) Schedule schedule,
    @Key(doc = "自己対局と評価の推論", change = Change.MACHINE) InferenceSettings inference,
    @Key(doc = "自己対局と評価の卓", change = Change.MACHINE) TableSettings table,
    @Key(doc = "学習器のデバイスと計算の型", change = Change.MACHINE) Learner learner,
    @Key(doc = "自己対局の相手", change = Change.COLLECT) SelfPlay selfPlay,
    @Key(doc = "自己対局で手を選ぶ方策の探索", change = Change.COLLECT) Exploration.Settings explore,
    @Key(doc = "自己対局の学習器で方策を学ぶ設定", change = Change.FIT) SelfPlayLearner.Ppo ppo,
    @Key(doc = "方策の学習率の制御", change = Change.FIXED) KlLearningRate.Settings actorRate,
    @Key(doc = "価値の学び方", change = Change.FIT) SelfPlayLearner.ValueRate valueRate,
    @Key(doc = "価値の目標", change = Change.FIT) SelfPlayLearner.VTrace vtrace,
    @Key(doc = "学習器のバッチの大きさ", change = Change.MACHINE) SelfPlayLearner.Batches batches,
    @Key(doc = "王者との決闘と評価の卓", change = Change.SCHEDULE) EvaluationSettings evaluation,
    @Key(doc = "牌譜で GRP を学習する設定", change = Change.FIT) Grp grp) {

  public ClassicSettings {
    Objects.requireNonNull(run, "run");
    Objects.requireNonNull(evaluation, "evaluation");
    if (evaluation.onWorse() instanceof OnWorseSettings.Rollback) {
      throw new InvalidSetting(
          "evaluation.onWorse",
          "the classic campaign reports or pauses when the champion duels stay worse; it does not roll back");
    }
    for (Map.Entry<String, TableSpec> entry : evaluation.tables().entrySet()) {
      TableSpec spec = entry.getValue();
      boolean promotes = spec.stop() instanceof StopSettings.Test test && test.promote();
      if (spec.every() > 0 && !promotes) {
        throw new InvalidSetting(
            "evaluation.tables." + entry.getKey() + ".every",
            "the classic campaign schedules only the table that promotes the champion; set every = 0 and play this"
                + " table from the duel");
      }
    }
  }

  /**
   * campaign の run のセクション。どのキーも run の間は変えない。
   *
   * @param variant ルール
   * @param seed 自己対局と学習の乱数の種
   * @param start 学習器の始めのネットワーク（classic のチェックポイントの絶対パス）。campaign の最初の起動でバージョン 1 に写す
   * @param grp 自己対局の局の始まりに、価値の入力の周辺確率を軌跡に書く GRP（epsilon-grp のチェックポイントの絶対パス）。無ければ書かず、
   *     価値の入力は一様になる
   */
  public record Run(
      @Key(doc = "ルール", value = "TENHOU_4P") Variant variant,
      @Key(doc = "自己対局と学習の乱数の種", value = "1") long seed,
      @Key(
              doc = "学習器の始めのネットワーク（classic のチェックポイントの絶対パスで、campaign を回すときに要る）",
              presence = Presence.ABSENT)
          Path start,
      @Key(
              doc = "局の始まりに価値の入力の周辺確率を軌跡に書く GRP（epsilon-grp のチェックポイントの絶対パスで、無ければ価値の入力は一様になる）",
              presence = Presence.ABSENT)
          Path grp) {

    public Run {
      Objects.requireNonNull(variant, "variant");
      requireAbsolute("start", start);
      requireAbsolute("grp", grp);
    }

    private static void requireAbsolute(String component, Path path) {
      if (path != null && !path.isAbsolute()) {
        throw new InvalidSetting(component, "expected an absolute path, not " + path);
      }
    }
  }

  /**
   * macro の数と大きさのセクション。
   *
   * @param macros 回す macro の数
   * @param games 1 回の macro で集める半荘の数
   */
  public record Schedule(
      @Key(doc = "回す macro の数", value = "1350", change = Change.FIT) @Range(min = 1) int macros,
      @Key(doc = "1 回の macro で集める半荘の数", value = "32768", change = Change.COLLECT) @Range(min = 1) int games) {}

  /**
   * 学習器のセクション。計算の型は、主のデバイスで決まる型（GPU なら BFLOAT16、CPU なら FLOAT32）だけを受け付ける。
   *
   * @param devices 学習のデバイスの列（先頭が主のデバイス、残りはデータ並列のレプリカ）
   * @param precision 学習器の計算の型
   */
  public record Learner(
      @Key(doc = "学習のデバイスの列（先頭が主のデバイス、残りはデータ並列のレプリカ）", value = "[cpu]") List<String> devices,
      @Key(doc = "学習器の計算の型", value = "FLOAT32", change = Change.FIT) Precision precision) {

    public Learner {
      devices = List.copyOf(devices);
      Objects.requireNonNull(precision, "precision");
      NetworkDevices parsed;
      try {
        parsed = NetworkDevices.parse(String.join(",", devices));
      } catch (IllegalArgumentException invalid) {
        throw new InvalidSetting("devices", invalid.getMessage());
      }
      Precision computed = parsed.primary().isGpu() ? Precision.BFLOAT16 : Precision.FLOAT32;
      if (precision != computed) {
        throw new InvalidSetting(
            "precision", "the learner computes in " + computed + " on " + parsed.primary() + ", not " + precision);
      }
    }

    /** 学習のデバイスの列。先頭が主のデバイス。 */
    public NetworkDevices networkDevices() {
      return NetworkDevices.parse(String.join(",", devices));
    }
  }

  /**
   * 自己対局の相手のセクション。macro ごとに相手を 1 人選び、学習器のほかの全ての席に座らせる。
   *
   * @param championProbability macro の相手に今の王者を選ぶ確率（ほかは過去の王者から 1 人を一様に選ぶ）
   * @param pool 相手に選べる過去の王者の数の上限（新しい王者から数える）
   */
  public record SelfPlay(
      @Key(doc = "macro の相手に今の王者を選ぶ確率（ほかは過去の王者から 1 人を一様に選ぶ）", value = "0.7")
          @Range(min = 0, max = 1)
          double championProbability,
      @Key(doc = "相手に選べる過去の王者の数の上限（新しい王者から数える）", value = "32") @Range(min = 1) int pool) {}

  /**
   * 牌譜で GRP を学習する設定。
   *
   * @param training GRP の学び方
   */
  public record Grp(@Key(doc = "GRP の学び方") GrpTraining training) {}

  /** 自己対局の学習器の設定を返す。方策と価値の学び方、学習率の制御、価値の目標、バッチの大きさをまとめたもの。 */
  public SelfPlayLearner.Settings learning() {
    return new SelfPlayLearner.Settings(ppo, actorRate, valueRate, vtrace, batches);
  }
}
