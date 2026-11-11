package com.epsilon.agent.run;

import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.RecordFile;
import java.nio.file.Path;

/** run の種類ごとの、ジョブの中身。 */
public interface JobRunner {

  /**
   * ジョブを初めて始めるときに、変わらない計画を決める。王者や「何バージョン前」のように指す先が変わっていく相手は、ここで具体的な
   * バージョンに決める。計画には、打ち手（系譜のバージョンか、系譜の外のチェックポイントとその SHA-256）、牌山の種と最初の番号、
   * 席の決め方、ツモの順、1 チャンクの牌山の数、判定の時点と臨界値を書く。
   *
   * @return 計画の木（{@link RecordFile#tree}）
   */
  Node.Table plan(JobFile job) throws Exception;

  /**
   * 計画の続きを打つ処理を開く。games.tsv と looks.tsv は、打ち終えた牌山までの長さに切り詰めてある。
   *
   * @param directory ジョブのディレクトリ（続きの集計は games.tsv を読み直して作る）
   * @return 続きを打つ処理
   */
  JobWork open(JobFile job, Path directory) throws Exception;

  /**
   * 試行が落ちた後に、ジョブを回す共有のリソース（推論のランタイムなど）がまだ使えることを確かめる。使えなければ例外を投げる。その例外で、
   * 次の試行もほかのジョブも始めずにジョブの並びを止める（開けないジョブの落ちた回数を増やさない）。
   */
  default void requireUsable() throws Exception {}

  /**
   * 済んだジョブの結果を反映する（王者の保存先、state にあるリグレッションの数など）。result.json を書いた後、報告の行を足す前に
   * 呼ぶ。落ちた後にもう一度呼ぶことがあるので、何度呼んでも 1 度呼んだのと同じ結果にする。
   *
   * @param result result.json の木
   */
  default void settle(JobFile job, Node.Table result) throws Exception {}
}
