package com.epsilon.grp.rules;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.Seats;
import com.epsilon.grp.train.CountedHands;
import com.epsilon.grp.train.GameRecord;
import com.epsilon.grp.train.HandResult;
import com.epsilon.mahjong.rules.AbortiveDraw;
import com.epsilon.mahjong.rules.BaseClass;
import com.epsilon.mahjong.rules.Progression;
import com.epsilon.mahjong.rules.RoundOutcome;
import com.epsilon.mahjong.rules.Settlement;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.List;
import java.util.SplittableRandom;
import java.util.function.Consumer;

/** 局の結果のコードと数のファイルのテストで使うヘルパー。卓と結果の作り方と、乱数で打った半荘とその数えを持つ。 */
public final class RulesFixtures {

  private RulesFixtures() {}

  /** 局の始まりの卓を作る。点と供託の点の和が、配給点の和になるように渡す。 */
  public static TableState table(Variant variant, int round, int honba, int kyotaku, int... scores) {
    TableState table = new TableState(variant);
    table.set(round, honba, kyotaku, scores);
    return table;
  }

  /**
   * 局の結果のコードを作る。結果は絶対席で書く。
   *
   * @param dealer その局の親の絶対席
   * @param fill 結果を書く処理
   */
  public static long code(Variant variant, int dealer, Consumer<RoundOutcome> fill) {
    RoundOutcome outcome = new RoundOutcome(variant);
    fill.accept(outcome);
    return HandResult.raw(outcome, dealer);
  }

  /** 全員が不聴の荒牌（立直なし）のコード。どの親でも同じコードになる。 */
  public static long allNoten(Variant variant) {
    return code(variant, 0, outcome -> outcome.setExhaustive(0, 0, 0));
  }

  /**
   * 乱数で打った半荘。局ごとの始まりの卓と結果の数えと、半荘の記録を持つ。
   *
   * @param counts 半荘 ID の昇順の数え
   * @param games 打った順の半荘の記録
   */
  public record Played(CountedHands counts, List<GameRecord> games) {}

  /**
   * 卓に依らない決まった分布で半荘を打つ。包のある役満・ダブロン・流し満貫・途中流局も出し、立直は 1,000 点以上の席だけが掛ける。
   *
   * @param firstId 最初の半荘 ID。以降は順に振る
   */
  public static Played play(Variant variant, int games, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    int seats = variant.seats();
    CountedHands.Builder builder = CountedHands.builder(variant, 1, 0, GrpFixtures.STAMP);
    List<GameRecord> records = new ArrayList<>(games);
    TableState table = new TableState(variant);
    TableState next = new TableState(variant);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[seats];
    BaseClass[] bases = BaseClass.values();
    for (int game = 0; game < games; game++) {
      long gameId = firstId + game;
      table.copyFrom(new TableState(variant));
      List<GrpState> handEnds = new ArrayList<>();
      while (true) {
        int riichi = 0;
        for (int seat = 0; seat < seats; seat++) {
          if (table.score(seat) >= Variant.DEPOSIT && random.nextInt(5) == 0) {
            riichi |= 1 << seat;
          }
        }
        int kind = random.nextInt(100);
        int winner = random.nextInt(seats);
        int loser = Seats.absolute(winner, 1 + random.nextInt(seats - 1), seats);
        int base = bases[random.nextInt(BaseClass.SANBAIMAN.ordinal())].base();
        if (kind < 2) {
          int pao = Seats.absolute(winner, 1 + random.nextInt(seats - 1), seats);
          int yakuman = BaseClass.YAKUMAN.base() * (1 + random.nextInt(2));
          if (pao != loser && random.nextBoolean()) {
            outcome.setRon(yakuman, winner, loser, pao, riichi);
          } else {
            outcome.setTsumo(yakuman, winner, pao, riichi);
          }
        } else if (kind < 32) {
          outcome.setTsumo(base, winner, RoundOutcome.NO_SEAT, riichi);
        } else if (kind < 74) {
          outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, riichi);
        } else if (kind < 78) {
          outcome.setRon(base, winner, loser, RoundOutcome.NO_SEAT, riichi);
          int other = Seats.absolute(loser, 1 + random.nextInt(seats - 1), seats);
          if (other != winner) {
            outcome.addRonWinner(bases[random.nextInt(bases.length)].base(), other, RoundOutcome.NO_SEAT);
          }
        } else if (kind < 96) {
          int tenpai = riichi;
          for (int seat = 0; seat < seats; seat++) {
            if (random.nextInt(5) < 2) {
              tenpai |= 1 << seat;
            }
          }
          int nagashi = random.nextInt(25) == 0 ? 1 << random.nextInt(seats) : 0;
          outcome.setExhaustive(tenpai, nagashi, riichi);
        } else {
          outcome.setAbortive(AbortiveDraw.NINE_TERMINALS, riichi);
        }
        builder.add(gameId, table, HandResult.raw(outcome, table.dealer()));
        Settlement.apply(variant, table, outcome, delta);
        Progression.next(table, outcome, delta, next);
        if (next.gameOver()) {
          records.add(new GameRecord(variant, gameId, handEnds, scores(next)));
          break;
        }
        handEnds.add(GrpState.of(next));
        table.copyFrom(next);
      }
    }
    return new Played(builder.build(), records);
  }

  /** 卓の点（絶対席の順）。 */
  public static int[] scores(TableState table) {
    int[] scores = new int[table.variant().seats()];
    for (int seat = 0; seat < scores.length; seat++) {
      scores[seat] = table.score(seat);
    }
    return scores;
  }
}
