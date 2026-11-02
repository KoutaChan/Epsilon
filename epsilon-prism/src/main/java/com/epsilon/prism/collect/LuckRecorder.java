package com.epsilon.prism.collect;

import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import com.epsilon.prism.luck.LuckBatch;
import com.epsilon.prism.luck.LuckCapture;
import com.epsilon.prism.luck.LuckSampling;
import com.epsilon.prism.luck.LuckSink;

/**
 * 経路（{@link GameRunner}）ごとの運の網 Ω の標本の捕捉（標本を集める収集だけ。{@link CollectContext#luck}）。{@link
 * EventLedger.Tracker#observe} が境界ごとに復元した出来事を {@link LuckCapture} に渡し（硬貨・O°・層・重み）、局末にその局の標本の束を渡す。
 * 状態を読むだけなので、進行・乱数・行は変えない。
 *
 * <p>根（硬貨の鍵の経路と根の判断）は {@link GameRunner} が置く（置き方は {@link GameRunner} の説明）。捕捉が失敗したら受け口
 * （{@link LuckSampleSink}）に失敗を渡して、以後は何もしない（収集は止めない）。1 度に触るのは 1 スレッドだけ。
 */
final class LuckRecorder implements LuckSink {

  private final LuckSampleSink samples;
  private final LuckCapture capture;
  private final long gameId;

  /** この局で硬貨を引いた出来事の数。 */
  private long events;

  /**
   * 経路の捕捉を、根を置いて作る。
   *
   * @param samples コホートの標本の受け口
   * @param gameId 半荘 ID
   * @param path 根の経路（{@link LuckSampling#MAIN_PATH} か写しの根の判断 ID）
   * @param root 根の判断 ID（主経路の局の始めは {@link LuckSampling#handRoot}）
   */
  LuckRecorder(LuckSampleSink samples, long gameId, long path, long root) {
    this.samples = samples;
    this.gameId = gameId;
    capture = new LuckCapture(samples.sampling());
    capture.root(gameId, path, root);
  }

  /** 根を置き直す（出来事の序数を 0 に戻す。束はそのまま）。 */
  void root(long path, long root) {
    capture.root(gameId, path, root);
  }

  @Override
  public void reveal(GameState state, int indicator, TurnEvent.Draw laterDraw) {
    if (!samples.active()) {
      return;
    }
    events++;
    try {
      capture.reveal(state, indicator, laterDraw);
    } catch (RuntimeException problem) {
      samples.fail(problem);
    }
  }

  @Override
  public void draw(GameState state, TurnEvent.Draw draw) {
    if (!samples.active()) {
      return;
    }
    events++;
    try {
      capture.draw(state, draw);
    } catch (RuntimeException problem) {
      samples.fail(problem);
    }
  }

  /**
   * 局末: その局の標本を渡し、硬貨を引いた出来事の数を受け口に足す。
   *
   * @return その局の標本（なければ null。失敗した後も null）
   */
  LuckBatch detach() {
    LuckBatch batch = capture.detach();
    samples.ended(events, batch);
    events = 0;
    return batch != null && samples.active() ? batch : null;
  }
}
