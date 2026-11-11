package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;

/** テスト用の半荘を作るヘルパー。ネットワークと卓を作るヘルパーは、com.epsilon.grp の GrpFixtures にある。 */
final class TrainFixtures {

  /** 和了の基本点。安い基本点ほど多く入れて、出やすくしてある。 */
  private static final BaseClass[] BASES = {
    BaseClass.B320, BaseClass.B320, BaseClass.B640, BaseClass.B640, BaseClass.B1280, BaseClass.MANGAN,
    BaseClass.MANGAN, BaseClass.HANEMAN, BaseClass.BAIMAN
  };

  private TrainFixtures() {}

  /**
   * ルール（{@code Settlement} と {@code Progression}）どおりに精算して進めた半荘を作る。局ごとに、荒牌・ロン・ツモを乱数で選ぶ。
   * 立直は、1,000 点以上持つ席が 1/6 の確率で掛ける。
   *
   * @param variant ルール（四人打ち・三人打ち）
   * @param games 半荘の数
   * @param firstId 最初の半荘 ID。以降は順に振る
   * @param seed 乱数の種
   * @return 作った半荘の記録
   */
  static List<GameRecord> simulated(Variant variant, int games, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> records = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      records.add(simulated(variant, firstId + game, random));
    }
    return records;
  }

  private static GameRecord simulated(Variant variant, long gameId, SplittableRandom random) {
    int seats = variant.seats();
    TableState table = new TableState(variant);
    TableState next = new TableState(variant);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[seats];
    List<GrpState> handEnds = new ArrayList<>();
    while (true) {
      int riichi = 0;
      for (int seat = 0; seat < seats; seat++) {
        if (table.score(seat) >= Variant.DEPOSIT && random.nextInt(6) == 0) {
          riichi |= 1 << seat;
        }
      }
      int kind = random.nextInt(20);
      int winner = random.nextInt(seats);
      int base = BASES[random.nextInt(BASES.length)].base();
      if (kind < 3) {
        outcome.setExhaustive(random.nextInt(1 << seats), 0, riichi);
      } else if (kind < 11) {
        int loser = (winner + 1 + random.nextInt(seats - 1)) % seats;
        outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, riichi);
      } else {
        outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, riichi);
      }
      Settlement.apply(variant, table, outcome, delta);
      Progression.next(table, outcome, delta, next);
      if (next.gameOver()) {
        int[] finalScores = new int[seats];
        for (int seat = 0; seat < seats; seat++) {
          finalScores[seat] = next.score(seat);
        }
        return new GameRecord(variant, gameId, handEnds, finalScores);
      }
      handEnds.add(GrpState.of(next));
      table.copyFrom(next);
    }
  }

  /**
   * 最終順位が局末の卓と関係なく一様に決まる、四人打ちの半荘を作る。局末 k の次局は k ＋ 1 で、局末の点数は 25,000 点のまわりに
   * 散らす。最終の点数は {40,000, 30,000, 20,000, 10,000} を一様に並べ替えたものなので、較正された Φ はどの局末でも E[u] = 0 になる。
   *
   * @param games 半荘の数
   * @param hands 半荘ごとの局末の数（6 なら、終わった局は東 1〜南 2）
   * @param random 乱数
   * @return 作った半荘の記録
   */
  static List<GameRecord> uniform(int games, int hands, SplittableRandom random) {
    List<GameRecord> records = new ArrayList<>(games);
    int[] finals = {40_000, 30_000, 20_000, 10_000};
    for (int game = 0; game < games; game++) {
      List<GrpState> handEnds = new ArrayList<>(hands);
      for (int hand = 0; hand < hands; hand++) {
        handEnds.add(new GrpState(Variant.TENHOU_4P, hand + 1, 0, 0, spread(random)));
      }
      int[] order = {0, 1, 2, 3};
      for (int index = 3; index > 0; index--) {
        int other = random.nextInt(index + 1);
        int held = order[index];
        order[index] = order[other];
        order[other] = held;
      }
      int[] finalScores = new int[4];
      for (int seat = 0; seat < 4; seat++) {
        finalScores[seat] = finals[order[seat]];
      }
      records.add(new GameRecord(Variant.TENHOU_4P, game, handEnds, finalScores));
    }
    return records;
  }

  /** 3 席を 25,000 ± 10,000 点の範囲に 100 点刻みで散らし、4 席目で和を 100,000 点にする。 */
  private static int[] spread(SplittableRandom random) {
    int[] scores = new int[4];
    int sum = 0;
    for (int seat = 0; seat < 3; seat++) {
      scores[seat] = 25_000 + 100 * (random.nextInt(201) - 100);
      sum += scores[seat];
    }
    scores[3] = 100_000 - sum;
    return scores;
  }
}
