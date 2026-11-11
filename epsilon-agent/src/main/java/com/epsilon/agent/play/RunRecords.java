package com.epsilon.agent.play;

import com.epsilon.agent.learn.StageTimer;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;

/**
 * 経路（{@link Run}）の問い合わせに使う、記録 O のバッチ。判断の時点の始めに、強制でない全ての判断の O をここへ取り込む。
 *
 * <p>所有者は経路で、スレッドではない。書くのは、その判断の時点を進めている進行のスレッドだけ。打ち手と系列の処理は、答えを書き終えて
 * エンジンを進めるまで読むだけにする。バッチは、次の判断の時点の {@link #clear} で使い直す。書き込みと、打ち手のスレッドの読み込みの
 * 順は、問い合わせの受け渡し（ドライバのイベントの列）が保証する。取り込みにかかった時間は経路に貯め、{@link #flush} でタイマーに足す。
 */
final class RunRecords {

  /** バッチを最初に確保するときの {@code long} の数（判断 3 つ分の O が入る）。 */
  private static final int BATCH_LONGS = 128;

  private ObservationBatch batch;
  private long nanos;
  private long captures;

  /** 判断の時点の始めに、前の時点の O を捨てる。前の時点の答えと記録は書き終えている。 */
  void clear() {
    if (batch != null) {
      batch.clear();
    }
  }

  /** O のバッチ（最初の取り込みの前は null）。 */
  ObservationBatch batch() {
    return batch;
  }

  /**
   * 判断の O を取り込んでバッチに足し、スレッドごとのヘルパーにある Observation をその O に付ける。局面と判断は、呼び出しの間だけ
   * 読む。
   *
   * @return バッチの中の O の番号
   */
  int capture(Table.Tools tools, Variant variant, SeatView view, Decisions decisions, int slot) {
    long started = System.nanoTime();
    if (batch == null) {
      batch = new ObservationBatch(variant, BATCH_LONGS);
    }
    int index = tools.writer.capture(view, decisions, slot, batch);
    tools.reader.bind(batch, index);
    nanos += System.nanoTime() - started;
    captures++;
    return index;
  }

  /** 取り込みにかかった時間と回数をタイマーに足し、0 に戻す。経路を 1 回進めるごとに、その終わりに 1 回呼ぶ。 */
  void flush(StageTimer<TableStage> timer) {
    if (captures == 0) {
      return;
    }
    timer.add(TableStage.CAPTURE, nanos, captures);
    nanos = 0;
    captures = 0;
  }
}
