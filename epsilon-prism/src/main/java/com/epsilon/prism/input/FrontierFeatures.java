package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinConditions;
import com.epsilon.core.Action;
import com.epsilon.prism.Limits;
import com.epsilon.prism.codec.Bfloat16;
import com.epsilon.prism.observe.Observation;
import java.nio.ByteOrder;
import java.nio.ShortBuffer;
import java.util.Arrays;

/**
 * 記録 O から 1 行の前線（{@link FrontierField}）を {@link FrontierBatch} に書く。O の純関数で、前線を使う網だけが使う。{@link Tokenizer} とは
 * 別に回し、基準の入力の費用に入れない。符号化のスレッドに 1 つ持ち、書くときに割り当てをしない。
 *
 * <p>遷移は Tokenizer と同じ順と同じ後の手（{@link TransitionHands}）でたどる。前線は形の枠（{@link ShapeAnalysis#callSlot} などの共有の枠）が
 * 同じ後の手で同じになる（枚数・副露の牌・捨てる牌で決まり、赤を見ない）ので、共有の枠ごとに 1 回だけ探す。自分の河は捨てる牌を足して前線の振聴に
 * 使う。
 */
public final class FrontierFeatures {

  private static final FrontierField[] FIELDS = FrontierField.values();
  private static final int WIDTH = FrontierField.WIDTH;
  private static final int SHARED_SLOTS = ShapeAnalysis.UNSHARED;

  private final EventTable events = new EventTable();
  private final TileCounts tiles = new TileCounts();
  private final TileSafety safety = new TileSafety();
  private final TransitionHands hands = new TransitionHands();
  private final TenpaiFrontier frontier = new TenpaiFrontier();
  private final short[] draft = new short[Limits.MAX_TRANSITIONS * WIDTH];
  private final short[] shared = new short[SHARED_SLOTS * WIDTH];
  private final long[] found = new long[(SHARED_SLOTS + Long.SIZE - 1) / Long.SIZE];
  private FrontierBatch batch;
  private ShortBuffer shorts;

  /**
   * O から 1 行の前線を書く。行の全ての遷移の列を書く（前線のない遷移は 0）。
   *
   * @param o 行の記録 O（{@link BatchPlan#add} でこの行に足したもの）
   * @param out 書く塊（付けた網の入力の塊の計画を封じたもの）
   * @param row 塊の中の行の番号
   * @throws IllegalStateException 計画を封じていないとき
   * @throws IllegalArgumentException 行の遷移の数が O と合わないとき
   */
  public void write(Observation o, FrontierBatch out, int row) {
    BatchPlan plan = out.host().plan();
    if (!plan.sealed()) {
      throw new IllegalStateException("batch plan must be sealed before writing rows");
    }
    int transitions = plan.transitions(row);
    if (transitions != o.transitions()) {
      throw new IllegalArgumentException(
          "row " + row + " holds " + transitions + " transitions, the observation " + o.transitions());
    }
    if (out != batch) {
      // 塊ごとに 1 回だけ、このスレッドの複製を作る（絶対位置の書き込みだけを使う）。
      batch = out;
      shorts = out.blob().duplicate().clear().order(ByteOrder.LITTLE_ENDIAN).asShortBuffer();
    }
    events.load(o);
    tiles.load(o, events);
    safety.load(events);
    hands.load(o, tiles, events);
    Arrays.fill(found, 0L);
    Arrays.fill(draft, 0, transitions * WIDTH, (short) 0);
    WinConditions conditions =
        WinConditions.publicDecision(
            SeatFacts.seatWindTile(o, 0), o.roundWindTile(), RiichiState.NONE);
    int transition = 0;
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (o.representative(action) != action) {
        continue;
      }
      boolean terminal = TransitionHands.terminal(o.action(action).type());
      for (int index = 0, count = o.transitionsOf(action); index < count; index++, transition++) {
        AfterHand after = hands.move(action, index);
        if (!terminal) {
          writeTransition(after, conditions, transition * WIDTH);
        }
      }
    }
    shorts.put(plan.firstTransition(row) * WIDTH, draft, 0, transitions * WIDTH);
  }

  /** 1 つの遷移の前線を下書きの at から書く（共有の枠で探してあれば写す）。 */
  private void writeTransition(AfterHand after, WinConditions conditions, int at) {
    int slot = hands.slot();
    boolean sharable = slot != ShapeAnalysis.UNSHARED;
    long bit = 1L << slot;
    if (sharable && (found[slot >>> 6] & bit) != 0L) {
      System.arraycopy(shared, slot * WIDTH, draft, at, WIDTH);
      return;
    }
    Action discard = hands.discard();
    long river = safety.river(0) | (discard == null ? 0L : 1L << discard.tileType());
    frontier.analyze(after, river, tiles, conditions);
    for (FrontierField field : FIELDS) {
      FrontierField.Measure measure = field.measure();
      int value = measure.count(frontier.mask(field.reach()), tiles);
      draft[at + field.ordinal()] = Bfloat16.fromFloat(value / measure.scale());
    }
    if (sharable) {
      System.arraycopy(draft, at, shared, slot * WIDTH, WIDTH);
      found[slot >>> 6] |= bit;
    }
  }
}
