package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.EvaluationSettings;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.run.Iteration;
import com.epsilon.agent.run.RunControl;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 決闘の段と昇格の段（{@link DuelStep}・{@link DecideStep}）が使う macro。campaign の反復が実装する。候補は macro が公開したバージョン
 * （{@link #published}）で、王者を替える卓をこの macro で打つかも、決闘と確かめの卓のジョブの名前も、このバージョンで決める。
 */
public interface DuelMacro extends Iteration {

  /** macro が公開したバージョン（学習器の今のネットワーク）。決闘に出す候補のバージョン。 */
  int published();

  /** run のディレクトリ。王者の保存先は {@link RunDirectory#champion}。 */
  RunDirectory run();

  /** 反復に固定した設定の行の、評価と決闘のセクション。 */
  EvaluationSettings evaluation();

  /** 打つルール。 */
  Variant variant();

  /**
   * 決闘の効用（学習の目的と同じもの）。
   *
   * @throws IOException 学習器のチェックポイントが読めないとき
   */
  RankUtility utility() throws IOException;

  /** run の種。決闘ごとの牌山の種は、この種とジョブの名前から作る。 */
  long seed();

  /**
   * 学習器の系列（Nano か Major）。候補と王者の打ち手の書き方の接頭辞にする。
   *
   * @throws IOException 学習器のチェックポイントが読めないとき
   */
  Series series() throws IOException;

  /** 卓のコア。段の間は、ほかに使わない。 */
  TableLoop table();

  /** 安全点の確認。ジョブのチャンクの後に呼ぶ。 */
  RunControl control();

  /** 打ち手を開く設定。 */
  PolicyOptions options();

  /** GPU の打ち手が載る推論のランタイム（CPU だけなら null）。閉じるのは campaign。 */
  NetworkRuntime runtime();

  /**
   * 学習器の今のネットワークを、チェックポイントとして新しいディレクトリに書く。全てを書き終えてから置くので、ディレクトリがあれば中身は
   * そろっている。
   *
   * @param directory まだ無いディレクトリ
   */
  void writeLearner(Path directory) throws IOException;

  /**
   * 打ち手を開く。classic は PRISM のモジュールを import しないので、どの系列もクラスパスにある系列の実装（{@link ModelFamily}）で開く。
   *
   * @param spec 打ち手の書き方（heuristic、nano:絶対パス、major:絶対パス、prism:絶対パス）
   */
  default Policy open(String spec) throws IOException {
    Series.Spec parsed = Series.parse(spec);
    return ModelFamily.of(parsed.series()).open(parsed.checkpoint(), variant(), options(), runtime());
  }
}
