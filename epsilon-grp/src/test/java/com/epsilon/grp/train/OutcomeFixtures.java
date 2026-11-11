package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.HandOutcomes;
import com.epsilon.grp.OutcomeNetwork;
import com.epsilon.grp.OutcomeShape;
import com.epsilon.grp.OutcomeTensor;
import com.epsilon.grp.OutcomeTerm;
import com.epsilon.grp.OutcomeWorkspace;
import com.epsilon.grp.Seats;
import com.epsilon.grp.WinMethod;
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

/**
 * 局の結果のネットワークのテストに使う、半荘とネットワークを作るヘルパー。半荘は、卓に依らない決まった分布で局の結果を選び、
 * Settlement と Progression で進める。結果は局の結果のネットワークが表せるものだけにする（荒牌で立直した席は聴牌、包と流し満貫は
 * 出さない）。立直は、1,000 点以上ある席が 1/5 の確率でかける。終わり方はツモ 30%、ロン 45%、ダブロン 3%、荒牌 18%（立直して
 * いない席は 2/5 で聴牌）、途中流局 4%。
 * 局の結果をネットワーク q から引いて打った半荘（{@link #drawn}）も作れる。
 */
final class OutcomeFixtures {

  static final Variant FOUR = Variant.TENHOU_4P;

  /** 局の結果に出す打点。 */
  private static final BaseClass[] BASES = {
    BaseClass.B240, BaseClass.B320, BaseClass.B400, BaseClass.B480, BaseClass.B640, BaseClass.B800, BaseClass.B960,
    BaseClass.B1280, BaseClass.B1600, BaseClass.B1920, BaseClass.MANGAN, BaseClass.HANEMAN, BaseClass.BAIMAN,
    BaseClass.SANBAIMAN, BaseClass.YAKUMAN
  };

  /** 打点ごとの出やすさの重み。並びは打点と同じで、安い手ほどおおむね出やすい。 */
  private static final int[] WEIGHTS = {12, 12, 6, 12, 12, 3, 10, 8, 2, 4, 12, 4, 2, 1, 1};

  private OutcomeFixtures() {}

  /**
   * 四人打ちの半荘を打つ。
   *
   * @param games 打つ半荘の数
   * @param firstId 最初の半荘 ID。以降は順に振る
   */
  static List<GameRecord> played(int games, long firstId, long seed) {
    SplittableRandom random = new SplittableRandom(seed);
    List<GameRecord> records = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      records.add(play(firstId + game, random));
    }
    return records;
  }

  /**
   * 範囲の各コホートで半荘を打ち、全ての局末を行にする。
   *
   * @param games コホートごとの半荘の数
   */
  static OutcomeRows.Split rows(RecencyWindow window, int games) {
    OutcomeRows.Builder builder = OutcomeRows.builder(FOUR, window, GateCohort.NEXT, state -> true);
    for (int cohort = window.first(); cohort <= window.newest(); cohort++) {
      builder.add(cohort, played(games, cohort * 100_000L, cohort));
    }
    return builder.build();
  }

  /** 重みが全て 0 のネットワークを作る。終わり方と打点は一様で、立直の確率は 1/2 になる。 */
  static OutcomeNetwork uniform(OutcomeShape shape) {
    return OutcomeNetwork.of(shape, new float[OutcomeTensor.starts(shape)[OutcomeTensor.values().length]]);
  }

  /**
   * 各行の最初の結果の項を数えて、卓に依らないネットワークを作る。終わり方と打点のロジットは「数 ＋ 1」の対数。立直のロジットは、
   * 「立直した数 ＋ 1」と「立直していない数 ＋ 1」の比の対数。
   */
  static OutcomeNetwork counted(OutcomeShape shape, OutcomeRows rows) {
    HandOutcomes hands = shape.outcomes();
    double[] counts = new double[hands.terms()];
    for (int row = 0; row < rows.size(); row++) {
      int at = rows.firstOutcome[row] * rows.stride;
      for (int j = 0; j < rows.stride && rows.terms[at + j] >= 0; j++) {
        counts[rows.terms[at + j]]++;
      }
    }
    int riichi = hands.termStart(OutcomeTerm.RIICHI);
    int quiet = hands.termStart(OutcomeTerm.NO_RIICHI);
    double[] logits = new double[hands.logits()];
    for (int index = 0; index < logits.length; index++) {
      logits[index] = index < riichi
          ? Math.log(counts[index] + 1)
          : Math.log((counts[index] + 1) / (counts[quiet + index - riichi] + 1));
    }
    return GrpFixtures.biased(shape, logits);
  }

  /**
   * q から局の結果を引いて打った半荘（打った順）と、同じ局を入れた数のファイルの中身。
   *
   * @param counted コホート 0・探索の温度 0 の数のファイルの中身
   */
  record Drawn(List<GameRecord> games, CountedHands counted) {}

  /**
   * 局の結果をネットワーク q から引いて半荘を打つ。局ごとに、局の始まりの卓で q が出す確率のとおりに終わり方・打点・立直を引き、
   * Settlement と Progression で進める。どの局の結果も q で表せるので、q はこの半荘の結果の分布そのものになる。
   *
   * @param games 打つ半荘の数
   * @param firstId 最初の半荘 ID。以降は順に振る
   */
  static Drawn drawn(OutcomeNetwork q, int games, long firstId, long seed) {
    Variant variant = q.variant();
    int seats = variant.seats();
    HandOutcomes hands = q.shape().outcomes();
    OutcomeWorkspace workspace = OutcomeWorkspace.of(q.shape());
    SplittableRandom random = new SplittableRandom(seed);
    CountedHands.Builder counts = CountedHands.builder(variant, 0, 0, GrpFixtures.STAMP);
    RoundOutcome outcome = new RoundOutcome(variant);
    int[] delta = new int[seats];
    List<GameRecord> records = new ArrayList<>(games);
    for (int game = 0; game < games; game++) {
      long gameId = firstId + game;
      TableState table = new TableState(variant);
      TableState next = new TableState(variant);
      List<GrpState> handEnds = new ArrayList<>();
      while (true) {
        q.forward(table, workspace);
        int dealer = table.dealer();
        int main = pick(random, workspace, hands.mainTerm(0), hands.mainCount());
        int base = HandOutcomes.NONE;
        int other = HandOutcomes.NONE;
        switch (hands.section(main)) {
          case TSUMO -> base = pick(
              random, workspace, hands.gradeTerm(hands.winner(main), WinMethod.TSUMO, 0), HandOutcomes.BASES);
          case RON -> base = pick(
              random, workspace, hands.gradeTerm(hands.winner(main), WinMethod.RON, 0), HandOutcomes.BASES);
          case DOUBLE_RON -> {
            base = pick(random, workspace, hands.gradeTerm(hands.winner(main), WinMethod.RON, 0), HandOutcomes.BASES);
            other = pick(
                random, workspace, hands.gradeTerm(hands.secondWinner(main), WinMethod.RON, 0), HandOutcomes.BASES);
          }
          case DRAW, ABORTIVE -> {}
        }
        int riichi = 0;
        for (int index = 0; index < hands.factorCount(main); index++) {
          int seat = hands.factorSeat(main, index);
          if (random.nextDouble() < workspace.probability(hands.riichiTerm(seat, hands.role(main, seat), true))) {
            riichi |= 1 << Seats.absolute(dealer, seat, seats);
          }
        }
        hands.place(main, base, other, riichi, dealer, outcome);
        counts.add(gameId, table, HandResult.raw(outcome, dealer));
        Settlement.apply(variant, table, outcome, delta);
        Progression.next(table, outcome, delta, next);
        if (next.gameOver()) {
          int[] finalScores = new int[seats];
          for (int seat = 0; seat < seats; seat++) {
            finalScores[seat] = next.score(seat);
          }
          records.add(new GameRecord(variant, gameId, handEnds, finalScores));
          break;
        }
        handEnds.add(GrpState.of(next));
        table.copyFrom(next);
      }
    }
    return new Drawn(records, counts.build());
  }

  /** 項の位置 {@code first} から続く {@code count} 個の確率に従って、番号を 1 つ引く。確率の和の端数は最後の番号に寄せる。 */
  private static int pick(SplittableRandom random, OutcomeWorkspace workspace, int first, int count) {
    double left = random.nextDouble();
    for (int index = 0; index < count - 1; index++) {
      left -= workspace.probability(first + index);
      if (left < 0) {
        return index;
      }
    }
    return count - 1;
  }

  private static GameRecord play(long gameId, SplittableRandom random) {
    TableState table = new TableState(FOUR);
    TableState next = new TableState(FOUR);
    RoundOutcome outcome = new RoundOutcome(FOUR);
    int[] delta = new int[4];
    List<GrpState> handEnds = new ArrayList<>();
    while (true) {
      int riichi = 0;
      for (int seat = 0; seat < 4; seat++) {
        if (table.score(seat) >= Variant.DEPOSIT && random.nextInt(5) == 0) {
          riichi |= 1 << seat;
        }
      }
      int kind = random.nextInt(100);
      int winner = random.nextInt(4);
      int loser = (winner + 1 + random.nextInt(3)) % 4;
      if (kind < 30) {
        outcome.setTsumo(base(random), winner, RoundOutcome.NO_SEAT, riichi);
      } else if (kind < 75) {
        outcome.setRon(base(random), winner, loser, RoundOutcome.NO_SEAT, riichi);
      } else if (kind < 78) {
        outcome.setRon(base(random), winner, loser, RoundOutcome.NO_SEAT, riichi);
        outcome.addRonWinner(base(random), third(winner, loser, random), RoundOutcome.NO_SEAT);
      } else if (kind < 96) {
        int tenpai = riichi;
        for (int seat = 0; seat < 4; seat++) {
          if (random.nextInt(5) < 2) {
            tenpai |= 1 << seat;
          }
        }
        outcome.setExhaustive(tenpai, 0, riichi);
      } else {
        outcome.setAbortive(AbortiveDraw.NINE_TERMINALS, riichi);
      }
      Settlement.apply(FOUR, table, outcome, delta);
      Progression.next(table, outcome, delta, next);
      if (next.gameOver()) {
        int[] finalScores = new int[4];
        for (int seat = 0; seat < 4; seat++) {
          finalScores[seat] = next.score(seat);
        }
        return new GameRecord(FOUR, gameId, handEnds, finalScores);
      }
      handEnds.add(GrpState.of(next));
      table.copyFrom(next);
    }
  }

  private static int base(SplittableRandom random) {
    int total = 0;
    for (int weight : WEIGHTS) {
      total += weight;
    }
    int pick = random.nextInt(total);
    for (int index = 0; ; index++) {
      pick -= WEIGHTS[index];
      if (pick < 0) {
        return BASES[index].base();
      }
    }
  }

  /** 和了者と放銃者を除く 2 席から、1 席を選ぶ。 */
  private static int third(int winner, int loser, SplittableRandom random) {
    int skip = random.nextInt(2);
    for (int seat = 0; ; seat++) {
      if (seat != winner && seat != loser && skip-- == 0) {
        return seat;
      }
    }
  }
}
