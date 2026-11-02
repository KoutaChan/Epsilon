package com.epsilon.prism.collect;

import com.epsilon.prism.metrics.StageTimer;
import com.epsilon.prism.observe.ObservationArena;
import com.epsilon.prism.observe.ObservationWriter;
import com.epsilon.prism.policy.PolicyRequest;
import com.epsilon.prism.policy.PolicyService;

/**
 * 経路（{@link GameRunner}）の問い合わせの記録 O の置き場。prepare が、記録 O を読む方策に出す問い合わせ（{@link PolicyService#readsRecord}）と
 * 行に残す判断の O だけをここへ捕捉する（{@link PolicyRequest#capture}。強制の判断と、O を読まない方策で残さない判断は捕捉しない）。
 *
 * <p>持ち主は経路で、スレッドではない: 経路は境界ごとに別の進行ワーカーへ移る（{@link CollectionDriver}）。書くのはその境界を進めている進行ワーカー
 * だけで、符号化のスレッドと行の記録（{@link RowRecorder}）は、答えを書き終えて commit するまで読むだけ。置き場は次の境界の {@link #clear}（prepare
 * の最初）で使い直す。問い合わせの受け渡し（ドライバのキュー）が、書き込みと符号化のスレッドの読み込みの順を作る。
 *
 * <p>書き手（{@link ObservationWriter}）は状態を持たない作業域なので、進行ワーカーのスレッドに 1 つ持つ。置き場は最初の捕捉で作る（O を捕捉しない
 * 経路は持たない）。捕捉の時間は経路に貯め、{@link #flush} で {@link StageTimer.Stage#V2_CAPTURE} に足す。
 */
final class QueryRecords {

  /** 置き場の初めの {@code long} の数（判断 3 つ分の O が入る）。 */
  private static final int ARENA_LONGS = 128;

  /** 進行ワーカーのスレッドごとの書き手（スレッド安全でない）。 */
  private static final ThreadLocal<ObservationWriter> WRITERS =
      ThreadLocal.withInitial(ObservationWriter::new);

  private ObservationArena arena;
  private long nanos;
  private long captures;

  /** 境界の始め（prepare の最初）: 前の境界の O を捨てる。前の境界の答えと記録は書き終えている。 */
  void clear() {
    if (arena != null) {
      arena.clear();
    }
  }

  /**
   * 問い合わせの O を置き場に捕捉する（{@link PolicyRequest#capture}）。観測と合法手は呼び出しの間だけ読む。
   *
   * @param request 捕捉する問い合わせ
   */
  void capture(PolicyRequest request) {
    long started = System.nanoTime();
    if (arena == null) {
      arena = new ObservationArena(ARENA_LONGS);
    }
    request.capture(WRITERS.get(), arena);
    nanos += System.nanoTime() - started;
    captures++;
  }

  /** 捕捉の時間と回数を timer に足して空にする（{@link GameRunner#advance} の終わりに 1 回）。 */
  void flush(StageTimer timer) {
    if (captures == 0) {
      return;
    }
    timer.add(StageTimer.Stage.V2_CAPTURE, nanos, captures);
    nanos = 0;
    captures = 0;
  }
}
