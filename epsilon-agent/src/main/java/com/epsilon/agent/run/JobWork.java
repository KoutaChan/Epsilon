package com.epsilon.agent.run;

import java.util.Objects;

/** ジョブの続きを打つ処理。閉じると、相手の方策を閉じて推論のリソースを返す。 */
public interface JobWork extends AutoCloseable {

  /**
   * 打ち終えた牌山の次からチャンクを 1 つ打つ。半荘の行は牌山の番号の順に games.tsv へ、判定の時点の行は looks.tsv へ書く。
   * 早く終わった牌山の行は、前の牌山が終わるまで手元に置く。
   *
   * @param walls 打ち終えた牌山の数
   * @param outputs 書き足す先
   * @return このチャンクまでに打ち終えた牌山の数と、決着したならその結果
   */
  Chunk next(long walls, JobOutputs outputs) throws Exception;

  /**
   * チャンク 1 つの結果。
   *
   * @param walls このチャンクまでに打ち終えた牌山の数
   * @param result 決着したときの結果（result.json に書く record）。続けるなら null
   */
  record Chunk(long walls, Record result) {

    public static Chunk more(long walls) {
      return new Chunk(walls, null);
    }

    public static Chunk decided(long walls, Record result) {
      return new Chunk(walls, Objects.requireNonNull(result, "result"));
    }
  }
}
