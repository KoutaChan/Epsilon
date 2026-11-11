package com.epsilon.prism.auxiliary;

import com.epsilon.mahjong.engine.FullView;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.shape.HandShape;
import com.epsilon.mahjong.shape.ShapeAnalyzer;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の時点の全ての状態（commit の前で、全員の手が見える）から Oracle の記録（{@link OracleCodec}）を作る。行に残す判断で呼ぶ。作業用の
 * オブジェクトはスレッドごと・ルールごとに持ち、新しく割り当てるのは返す配列だけ。
 *
 * <p>相手ごとに、最小の向聴数と形だけで見た待ちを求める。待ちの牌ごとに、判断した席が今その牌の赤でない 1 枚を捨てたらロンされるかを、
 * エンジンの判定で確かめる。相手が手牌と面子で 4 枚とも持つ牌は、5 枚目がなく和了れないので確かめない。ロンされる牌では、その和了の点を
 * 基本点から求める。裏ドラ・本場・供託・包は含めない。
 */
public final class OracleCapture {

  /** 赤でない牌の写しの番号。赤は 0 番の写し。 */
  private static final int PLAIN_COPY = Tiles.RED_COPY + 1;

  private static final ThreadLocal<OracleCapture[]> CAPTURES =
      ThreadLocal.withInitial(() -> new OracleCapture[Variant.values().length]);

  private final OracleCodec codec;
  private final ShapeAnalyzer shapes;
  private final HandShape hand = new HandShape();
  private final WinScore score = new WinScore();
  private final long[] draft;

  private OracleCapture(Variant variant) {
    codec = OracleCodec.of(variant);
    shapes = new ShapeAnalyzer(variant);
    draft = new long[codec.maxLongs()];
  }

  /**
   * Oracle の記録を作る。
   *
   * @param full 判断の時点の全ての状態（変えない）
   * @param seat 判断した席（絶対席）
   * @param ws ルールの判定に使う作業領域
   * @return Oracle の記録。長さは {@link OracleCodec#longs} の値
   * @throws IllegalStateException 相手の手が 13 枚相当でないとき
   */
  public static long[] capture(FullView full, int seat, RuleWorkspace ws) {
    Variant variant = full.variant();
    OracleCapture[] all = CAPTURES.get();
    OracleCapture capture = all[variant.ordinal()];
    if (capture == null) {
      capture = new OracleCapture(variant);
      all[variant.ordinal()] = capture;
    }
    return capture.run(full, seat, ws);
  }

  private long[] run(FullView full, int seat, RuleWorkspace ws) {
    int seats = full.variant().seats();
    Objects.checkIndex(seat, seats);
    SeatView table = full.seat(seat);
    Arrays.fill(draft, 0L);
    int dangers = 0;
    for (int relative = 1; relative < seats; relative++) {
      int player = (seat + relative) % seats;
      full.copyHand(player, hand);
      if (hand.size() != HandShape.MAX_SIZE - 1) {
        throw new IllegalStateException("Opponent " + player + " holds " + hand.size() + " tiles at a decision");
      }
      int shanten = shapes.shanten(hand);
      long waits = full.waits(player);
      long danger = 0L;
      boolean dealer = player == table.dealer();
      for (long rest = waits & ~TenpaiProbe.fullyHeld(table, player, hand); rest != 0L; rest &= rest - 1) {
        int kind = Long.numberOfTrailingZeros(rest);
        if (full.ronScore(player, Tiles.physicalId(kind, PLAIN_COPY), seat, ws, score)) {
          danger |= Tiles.bit(kind);
          codec.writePoints(draft, 0, dangers++, ScoreMath.ron(score.base(), dealer));
        }
      }
      codec.writeOpponent(draft, 0, relative, shanten, waits, danger);
    }
    return Arrays.copyOf(draft, codec.longs(draft, 0));
  }
}
