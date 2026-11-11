package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.classic.app.ClassicSettings;
import com.epsilon.classic.learn.TrajectoryBatch;
import com.epsilon.classic.learn.UncertaintyCdf;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;

/**
 * campaign の自己対局（収集の段の中身）。計画の今の王者と過去の王者から macro の相手を 1 人選び、学習器のバージョンと相手を席に置いて
 * macro の半荘を打ち、学習器の席の判断を軌跡に書く。結果は計画だけで決まり、ワーカーの数や推論のバッチの組み方に依らない。
 */
public interface Collector extends AutoCloseable {

  /**
   * 1 回の macro の半荘を打つ。
   *
   * @return 学習器の席の軌跡（半荘ごとに 1 つ）、この macro で数えた迷いの分布、選んだ相手
   */
  Collected collect(Macro macro) throws Exception;

  @Override
  void close();

  /**
   * 1 回の macro の計画。
   *
   * @param number macro の番号（反復の番号）
   * @param seed macro の乱数の種
   * @param settings 反復に固定した設定の行（半荘の数・探索・相手の選び方）
   * @param learner 学習器が打つバージョンのチェックポイント
   * @param champion 今の王者
   * @param pool 相手に選べる過去の王者（王者になった順。設定の上限まで、新しい王者から）
   * @param calibration 前の macro で数えた迷いの分布（最初の macro は空）
   */
  record Macro(
      int number,
      long seed,
      ClassicSettings settings,
      Path learner,
      Champion champion,
      List<Champion> pool,
      UncertaintyCdf calibration) {

    public Macro {
      if (number < 1) {
        throw new IllegalArgumentException("A macro number starts at 1: " + number);
      }
      Objects.requireNonNull(settings, "settings");
      Objects.requireNonNull(learner, "learner");
      Objects.requireNonNull(champion, "champion");
      pool = List.copyOf(pool);
      Objects.requireNonNull(calibration, "calibration");
    }
  }

  /**
   * 1 回の macro の結果。
   *
   * @param trajectories 学習器の席の軌跡（半荘ごとに 1 つ）
   * @param calibration この macro で数えた迷いの分布（次の macro の探索が引く）
   * @param opponent この macro の相手（今の王者か、計画の過去の王者の 1 人）
   */
  record Collected(TrajectoryBatch trajectories, UncertaintyCdf calibration, Champion opponent) {

    public Collected {
      Objects.requireNonNull(trajectories, "trajectories");
      Objects.requireNonNull(calibration, "calibration");
      Objects.requireNonNull(opponent, "opponent");
    }
  }

  /** 起動の行の設定と JVM のリソースで、自己対局を開く。 */
  @FunctionalInterface
  interface Factory {

    /**
     * @param machine 起動の行の設定（推論のデバイス・バッチ・卓の大きさ）
     * @param runtime 自己対局と評価の推論のランタイム（閉じるのは呼び出し元）
     * @param table 自己対局と評価の卓のコア（閉じるのは呼び出し元）
     */
    Collector open(ClassicSettings machine, NetworkRuntime runtime, TableLoop table) throws IOException;
  }
}
