package com.epsilon.prism;

import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRow;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Settings;
import com.epsilon.prism.app.PrismApp;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * テスト用の PRISM の設定を作る。アノテーションの既定値と四人打ちの層の上に、テストのルールと乱数の種、プリセット、「キーのパス=値」の上書きを
 * この順に重ねる（{@code settings create} と同じ重ね方）。上書きは 1 つずつ別の層になるので、同じキーは後の上書きが勝つ。ルールと種も上書きできる。
 */
public final class TestSettings {

  /** テストの run の乱数の種。 */
  public static final long SEED = 20261010L;

  /**
   * CPU で回せる小さな系譜の上書き。学習器は本番と同じ形で、1 コホートは 2 局。標準の幅のネットワークは CPU の 1 スレッドでは遅く、
   * 収集する判断の大半は分岐なので、ラベル行とロールインの率、候補の数を本番より小さくする。評価の卓は持たない。
   */
  private static final List<String> CPU_LINEAGE =
      List.of(
          "schedule.cohorts=[{from = 1, games = 2}]",
          "schedule.iterations=10",
          "table.activeGames=8",
          "table.workers=2",
          "table.batchWaitMicros=1000",
          "table.writers=1",
          "inference.encoders=2",
          "inference.slots=1",
          "inference.readyBatches=2",
          "inference.maxBatch=256",
          "noise.rates.table.DISCARD_EARLY=0.003",
          "noise.rates.table.DISCARD_MID=0.003",
          "noise.rates.table.DISCARD_LATE=0.003",
          "noise.rates.table.THREAT=0.03",
          "noise.rates.table.RIICHI=0.03",
          "noise.rates.table.CALL=0.03",
          "noise.rates.table.KAN=0.03",
          "noise.rates.table.END=0.03",
          "labels.maxCandidates=4",
          "labels.rollin.DISCARD_EARLY=0.001",
          "labels.rollin.DISCARD_MID=0.001",
          "labels.rollin.DISCARD_LATE=0.001",
          "labels.rollin.THREAT=0.015",
          "labels.rollin.RIICHI=0.015",
          "labels.rollin.CALL=0.015",
          "labels.rollin.KAN=0.015",
          "labels.rollin.END=0.015",
          "labels.rollin.FINAL_END=0.015",
          "labels.copyLabelRate=0.2",
          "fit.batch=16",
          "learner.decodeThreads=2");

  /** 方策の記憶を、バージョン 2 から EMA で平均する上書き。 */
  public static final String[] EMA_FROM_2 = {"memory.scheme=EMA", "memory.from=2"};

  private TestSettings() {}

  /** CPU で回せる小さな系譜の上書きの後ろに、追加の上書きをつなぐ（後の上書きが勝つ）。 */
  public static String[] cpuLineage(String... more) {
    return concat(CPU_LINEAGE.toArray(String[]::new), more);
  }

  /** 2 つの上書きの列を、この順につなぐ。 */
  public static String[] concat(String[] first, String... second) {
    List<String> all = new ArrayList<>(List.of(first));
    all.addAll(List.of(second));
    return all.toArray(String[]::new);
  }

  /**
   * 評価の卓を 1 つ足す上書き。1 半荘ごとに席を回し、ツモは席ごとの列から引く。牌山は全てのバージョンで同じものを使い、決めた数を全て
   * 打ってから順位で判定する。
   *
   * @param name 卓の名前
   * @param opponent 相手（heuristic か nano:〈絶対パス〉）
   * @param walls 牌山の数（席の数の倍数）
   * @param every 測るバージョンの間隔
   * @param from 測る最初のバージョン
   */
  public static String[] table(String name, String opponent, int walls, int every, int from) {
    String prefix = "evaluation.tables." + name + ".";
    return new String[] {
      prefix + "opponent=" + opponent,
      prefix + "seating=BY_GAME",
      prefix + "draws=SEAT_STREAMS",
      prefix + "wallSet=FIXED",
      prefix + "walls=" + walls,
      prefix + "stop=FIXED",
      prefix + "score=RANK",
      prefix + "every=" + every,
      prefix + "from=" + from
    };
  }

  /** 上書きを重ねた設定を返す。 */
  public static PrismSettings of(String... assignments) {
    return bind(List.of(), assignments).value();
  }

  /**
   * プリセットと上書きを重ねた設定を作る。
   *
   * @param presets 同梱のプリセットの名前（重ねる順）
   * @param assignments 「キーのパス=値」の上書き
   * @return 重ねた設定
   */
  public static Settings<PrismSettings> bind(List<String> presets, String... assignments) {
    List<Layer> layers = new ArrayList<>();
    layers.add(Layer.assignments("test run", List.of("run.variant=TENHOU_4P", "run.seed=" + SEED)));
    for (String preset : presets) {
      layers.add(PrismApp.application().preset(preset));
    }
    for (String assignment : assignments) {
      layers.add(Layer.assignments("test", List.of(assignment)));
    }
    return Settings.bind(PrismSettings.class, PrismApp.application().stack(layers));
  }

  /** 上書きを重ねた設定を、run のディレクトリの設定の 1 行目に書く（{@code settings create} が書く行と同じ）。 */
  public static PrismSettings create(Path run, String... assignments) throws IOException {
    return create(run, List.of(), assignments);
  }

  /** プリセットと上書きを重ねた設定を、run のディレクトリの設定の 1 行目に書く。 */
  public static PrismSettings create(Path run, List<String> presets, String... assignments) throws IOException {
    Settings<PrismSettings> settings = bind(presets, assignments);
    try (SettingsRows.Locked locked = new SettingsRows(new RunDirectory(run)).lock()) {
      locked.append(1, "test", SettingsRow.CREATED, List.of(), settings.tree());
    }
    return settings.value();
  }
}
