package com.epsilon.prism4p.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.codec.Bfloat16;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.FourPlayerFormat;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.EventTable;
import com.epsilon.prism.input.PublicSeatFacts;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.TileSafety;
import com.epsilon.prism.input.TransitionHands;
import java.util.Arrays;

/**
 * 判断の記録から、1 行分のフロンティア（{@link FrontierField}）を求める。判断の記録だけで決まる純関数で、{@link Tokenizer} とは別に動かす。
 * スレッドごとに 1 つ持ち、求めるときにメモリを確保しない。
 *
 * <p>遷移は Tokenizer と同じ順にたどり、遷移の後の手も Tokenizer と同じもの（{@link TransitionHands}）を使う。形を共有するスロット
 * （{@link ShapeAnalysis#callSlot} など）が同じ遷移では、フロンティアも同じになる（枚数・副露の牌・捨てる牌で決まり、赤を見ない）。そのため共有のスロットごとに 1 回だけ探す。
 * フロンティアの振聴は、自分の河に捨てる牌を足して見る。
 */
public final class FrontierFeatures {

  private static final FrontierField[] FIELDS = FrontierField.values();
  private static final int WIDTH = FrontierField.WIDTH;
  private static final int SHARED_SLOTS = ShapeAnalysis.UNSHARED;

  private final EventTable events;
  private final TileCounts tiles;
  private final SeatFacts seats;
  private final TileSafety safety;
  private final TransitionHands hands = new TransitionHands();
  private final TenpaiFrontier frontier;
  private final short[] draft = new short[Limits.MAX_TRANSITIONS * WIDTH];
  private final short[] shared = new short[SHARED_SLOTS * WIDTH];
  private final long[] found = new long[(SHARED_SLOTS + Long.SIZE - 1) / Long.SIZE];
  private Observation o;

  public FrontierFeatures() {
    events = new EventTable(FourPlayerFormat.VARIANT);
    tiles = new TileCounts(FourPlayerFormat.VARIANT);
    seats = new SeatFacts(FourPlayerFormat.VARIANT);
    safety = new TileSafety(FourPlayerFormat.SEATS);
    frontier = new TenpaiFrontier(FourPlayerFormat.VARIANT);
  }

  /**
   * 判断の記録から 1 行分のフロンティアを求める。遷移の順に、遷移ごとに {@link FrontierField#WIDTH} 個の列を宣言の順に並べる。値は
   * {@link FrontierField.Measure#scale} で割ってから BF16 に丸めたもので、フロンティアのない遷移は 0。
   *
   * @param o 行の判断の記録
   * @return 先頭から判断の記録の遷移の数 × {@link FrontierField#WIDTH} 個の値（次に求めるまで有効）
   */
  public short[] values(Observation o) {
    int transitions = o.transitions();
    this.o = o;
    events.load(o, tiles, seats);
    safety.load(events, seats);
    hands.load(o, tiles, seats);
    Arrays.fill(found, 0L);
    Arrays.fill(draft, 0, transitions * WIDTH, (short) 0);
    int seatWind = PublicSeatFacts.seatWindTile(o, 0);
    int transition = 0;
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (o.representative(action) != action) {
        continue;
      }
      boolean terminal = TransitionHands.terminal(o.action(action).type());
      for (int index = 0, count = o.transitionsOf(action); index < count; index++, transition++) {
        AfterHand after = hands.move(action, index);
        if (!terminal) {
          writeTransition(after, seatWind, transition * WIDTH);
        }
      }
    }
    return draft;
  }

  /** 1 つの遷移のフロンティアを、下書きの {@code at} の位置から書く。共有のスロットで探してあれば、その結果を写す。 */
  private void writeTransition(AfterHand after, int seatWind, int at) {
    int slot = hands.slot();
    boolean sharable = slot != ShapeAnalysis.UNSHARED;
    long bit = 1L << slot;
    if (sharable && (found[slot >>> 6] & bit) != 0L) {
      System.arraycopy(shared, slot * WIDTH, draft, at, WIDTH);
      return;
    }
    Action discard = hands.discard();
    long river = safety.river(0) | (discard == null ? 0L : Tiles.bit(discard.tile()));
    frontier.analyze(after, river, tiles, o, seatWind);
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
