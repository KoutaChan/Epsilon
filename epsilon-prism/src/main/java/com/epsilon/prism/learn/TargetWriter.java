package com.epsilon.prism.learn;

import com.epsilon.agent.observe.Observation;
import com.epsilon.prism.input.InputFormat;
import com.epsilon.prism.model.AuxiliaryLabels;
import com.epsilon.prism.model.AuxiliaryLoss;
import com.epsilon.prism.rows.Row;

/**
 * バッチの行から補助のヘッドの教師（候補と読み）を作るライター。符号化のスレッドごとに 1 つ持つ。バッチごとに {@link #clear} を呼び、
 * 行ごとに {@link #add} を呼び、最後に {@link #labels} で教師を受け取る。
 */
public interface TargetWriter {

  /**
   * 補助のヘッドを計算する必要があるかを返す。どちらかの群の重みが正なら必要になる。
   *
   * @param loss 補助のヘッドの損失
   * @return 補助のヘッドを計算するなら true
   */
  static boolean needed(AuxiliaryLoss loss) {
    return loss.resultWeight() > 0 || loss.oracleWeight() > 0;
  }

  /** バッチの教師を捨てる。 */
  void clear();

  /**
   * 1 行の教師を足す。その行の入力を書いた直後に、同じ記録と同じ行のライターを渡して呼ぶ。
   *
   * @param row バッチの中の行の番号
   * @param firstAction その行の最初の手が、バッチの中で何番目の手か
   * @param source 行
   * @param o 行の記録 O
   * @param writer 直前にこの行を書いた行のライター（その行の色の並べ替えを使う）
   */
  void add(int row, int firstAction, Row source, Observation o, InputFormat.RowWriter writer);

  /**
   * 足した行の教師を返す。返す配列はコピーなので、返した後にライターを使い回してよい。
   *
   * @return 教師
   */
  AuxiliaryLabels labels();
}
