package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.RoundStart;
import com.epsilon.mahjong.rules.Variant;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 局の途中で状態全体をコピーすると、どのフィールドも元と等しく、元と何も共有しない。同じ手で進めれば同じ状態になり、コピーを進めても元は
 * 変わらない。
 */
public class StateCopyTest {

  @Test
  public void aCopyEqualsTheOriginalSharesNothingAndContinuesTheSame() {
    Random random = new Random(64);
    for (Variant variant : Variant.values()) {
      for (int trial = 0; trial < 30; trial++) {
        RandomRound original = new RandomRound(variant);
        original.start(
            RoundStart.first(variant),
            random.nextLong(),
            1 + trial,
            trial % 2 == 0 ? DrawOrder.SEAT_STREAMS : DrawOrder.WALL_ORDER);
        int turns = random.nextInt(25);
        for (int turn = 0; turn < turns; turn++) {
          original.turn(random);
        }
        RandomRound copy = new RandomRound(variant);
        copy.copyFrom(original);
        Fields.assertSameFields(original, copy);
        RandomRound snapshot = new RandomRound(variant);
        snapshot.copyFrom(original);

        long rest = random.nextLong();
        playIntoTheNextRound(copy, new Random(rest));
        Fields.assertSameFields(snapshot, original);
        playIntoTheNextRound(original, new Random(rest));
        Fields.assertSameFields(original, copy);
      }
    }
  }

  @Test
  public void copiesNeedTheSameVariant() {
    RandomRound four = new RandomRound(Variant.TENHOU_4P);
    RandomRound three = new RandomRound(Variant.TENHOU_3P);
    Assert.assertThrows(IllegalArgumentException.class, () -> four.wall.copyFrom(three.wall));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.state.copyFrom(three.state));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.chance.copyFrom(three.chance));
    Assert.assertThrows(IllegalArgumentException.class, () -> four.state.dora().copyFrom(three.state.dora()));
  }

  /** 局を終えて裏ドラ表示牌をめくり、次の牌山を配って数手進める（コピーも牌山の epoch を引き継ぐ）。 */
  private static void playIntoTheNextRound(RandomRound round, Random random) {
    round.finish(random);
    round.revealUra();
    round.deal(RoundStart.first(round.variant));
    for (int turn = 0; turn < 5; turn++) {
      round.turn(random);
    }
  }
}
