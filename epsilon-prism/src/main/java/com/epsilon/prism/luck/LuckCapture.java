package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.core.TurnEvent;
import java.util.Objects;

/**
 * 1 本の経路（主経路・写し・分岐）の運の出来事の捕捉。出来事ごとに硬貨（{@link LuckSampling}）を引き、標本にする出来事だけ観測 O° を
 * 書いて層の鍵と重みを付け、その局の {@link LuckBatch} に足す。標本にしない出来事は序数を進めるだけで、O° を書かない。
 *
 * <p>呼ぶ場所:
 *
 * <ul>
 *   <li>{@link #root}: {@code GameRunner} が経路の根を置くところ（{@link LuckSampling} の根の置き方）。分岐と写しは作った
 *       とき、主経路は局の始めと、写しか分岐を作った判断の後。
 *   <li>{@link #reveal}・{@link #draw}: {@code EventLedger.Tracker.observe}（{@link LuckSink} の呼ぶ場所）。
 *   <li>{@link #detach}: {@code GameRunner} の endHand（{@link LuckBatch} の受け渡し）。
 * </ul>
 *
 * <p>経路ごとに 1 つ持ち、一度に触るのは 1 スレッドだけ（{@code GameRunner} と同じ）。書き手・読み手・解析器は進行ワーカーのスレッドごと
 * に持つ。O° は標本の置き場に直に書き、置き場を伸ばすときと局の最初の標本で {@link LuckBatch} を作るときを除いて割り当てをしない。
 */
public final class LuckCapture implements LuckSink {

  private static final ThreadLocal<Tools> TOOLS = ThreadLocal.withInitial(Tools::new);

  private final LuckSampling sampling;
  private long gameId;
  private long path;
  private long root;
  private int ordinal;
  private boolean rooted;
  private LuckBatch batch;

  /**
   * 捕捉を作る。
   *
   * @param sampling 標本を引く硬貨
   */
  public LuckCapture(LuckSampling sampling) {
    this.sampling = Objects.requireNonNull(sampling, "sampling");
  }

  /**
   * 経路の根を置き、出来事の序数を 0 に戻す。
   *
   * @param gameId 半荘 ID
   * @param path 経路（{@link LuckSampling#MAIN_PATH} か写しの根の判断 ID）
   * @param root 根の判断 ID（主経路の局の始めは {@link LuckSampling#handRoot}）
   */
  public void root(long gameId, long path, long root) {
    this.gameId = gameId;
    this.path = path;
    this.root = root;
    ordinal = 0;
    rooted = true;
  }

  /**
   * {@inheritDoc}
   *
   * @throws IllegalStateException 根を置く前のとき
   */
  @Override
  public void reveal(GameState state, int indicator, TurnEvent.Draw laterDraw) {
    if (!sampled()) {
      return;
    }
    Tools tools = TOOLS.get();
    LuckBatch target = batch();
    add(tools, target, tools.writer.captureReveal(state, indicator, laterDraw, target.arena()));
  }

  /**
   * {@inheritDoc}
   *
   * @throws IllegalStateException 根を置く前のとき
   */
  @Override
  public void draw(GameState state, TurnEvent.Draw draw) {
    if (!sampled()) {
      return;
    }
    Tools tools = TOOLS.get();
    LuckBatch target = batch();
    add(tools, target, tools.writer.captureDraw(state, draw, target.arena()));
  }

  /**
   * 今の局の標本を渡し、次の標本は新しい束に入れる。
   *
   * @return 今の局の標本（なければ null）
   */
  public LuckBatch detach() {
    LuckBatch detached = batch;
    batch = null;
    return detached;
  }

  /** 次の出来事の硬貨を引き、序数を進める。 */
  private boolean sampled() {
    if (!rooted) {
      throw new IllegalStateException("luck capture has no root");
    }
    return sampling.sampled(gameId, path, root, ordinal++);
  }

  private LuckBatch batch() {
    if (batch == null) {
      batch = new LuckBatch(gameId);
    }
    return batch;
  }

  private void add(Tools tools, LuckBatch target, int base) {
    int stratum = tools.stratum.of(tools.observation.bind(target.arena().words(), base));
    target.add(base, stratum, sampling.weight());
  }

  /** 進行ワーカーのスレッドごとの書き手・読み手・解析器。 */
  private static final class Tools {
    final LuckWriter writer = new LuckWriter();
    final LuckObservation observation = new LuckObservation();
    final LuckStratum stratum = new LuckStratum();
  }
}
