package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.AllocationProbe;
import org.testng.annotations.Test;

/** ウォームアップの後は、合法手・他家の応答・和了の判定がメモリを確保しない。 */
public class LegalAllocationTest {

  private static final int WARM_UP = 20_000;
  private static final int MEASURED = 100_000;

  @Test
  public void legalMovesWindowsAndWinChecksAllocateNothingOnceWarm() {
    AllocationProbe.assertNoAllocation(WARM_UP, MEASURED, new Workload()::run);
  }

  /**
   * 測る処理の 1 回分。ツモ和了・暗槓・立直のある手番と、立直中に暗槓できる手番で合法手を作る。打牌への応答（ロン・ポン・チー）と
   * 加槓への応答は、求めてから締める。ほかに、鳴いた後の打牌、ロンの点数、包、聴牌、流し満貫を調べる。応答を締めると状態が変わる
   * ので、応答を求める前に毎回元の局面をコピーする。
   */
  private static final class Workload {

    private final Scene turn = Scene.first().river(0, "9s").hand(0, "1111m23m2222p34p5s").draw(0, "5s");
    private final Scene riichi =
        Scene.first().hand(0, "111m456p789s2355s").river(0, "9p").riichi(0).draw(0, "1m");
    private final Scene response =
        Scene.first()
            .hand(1, "3467p123m789s11z9m")
            .hand(2, "055p123s456s789s2z")
            .hand(3, "123m456m789m46p11z")
            .riichi(3)
            .indicator("1m")
            .ura(0, "1s")
            .river(0, "5p");
    private final Scene chankan =
        Scene.first()
            .pon(1, 3, "4s")
            .hand(1, "4s123m456m789p")
            .hand(2, "123p456p789p35s11z")
            .draw(1, "5z")
            .kakan(1, "4s");
    private final Scene postCall = Scene.first().hand(1, "2345m6899p789s11z").river(0, "2m").chi(1, "234m");
    private final Scene work = Scene.first();
    private final int[] responseChoices;
    private final int[] chankanChoices;
    private final WinScore score = new WinScore();

    Workload() {
      responseChoices = response.respond().choices("CHI(4p5p6p 5p)", "PON(5p)", "PASS");
      chankanChoices = chankan.respond().choices("PASS");
    }

    long run(int rounds) {
      long checksum = 0;
      for (int round = 0; round < rounds; round++) {
        checksum += Legal.turn(turn.state, turn.hands[0], turn.wall, turn.decisions, round, turn.ws);
        checksum += Legal.turn(riichi.state, riichi.hands[0], riichi.wall, riichi.decisions, round, riichi.ws);
        checksum += turn.decisions.win(0).base();
        checksum += close(response, responseChoices, round);
        checksum += close(chankan, chankanChoices, round);
        checksum += Legal.postCall(postCall.hands[1], 1, postCall.decisions, round);
        if (response.full.ronScore(3, Tiles.physicalId(Tiles.P5, 3), 1, response.ws, score)) {
          checksum += score.han();
        }
        checksum += WinLegality.pao(response.full.seat(3), score, 1);
        checksum += WinLegality.tenpai(response.hands[3]) ? 1 : 0;
        checksum += WinLegality.nagashi(postCall.state.events(), 0) ? 1 : 0;
      }
      return checksum;
    }

    /**
     * 局面をコピーして他家の応答を求め、選んだ手で応答を締める。
     *
     * @param scene コピーする局面
     * @param choices 席ごとに選んだ手（合法手の何番目か）
     * @param id 次に使う判断の番号
     */
    private long close(Scene scene, int[] choices, long id) {
      work.copyFrom(scene);
      long next = work.window.open(work.state, work.hands, work.wall, work.decisions, id, work.ws, true);
      return next + work.window.close(work.state, work.hands, work.decisions, choices, work.resolution).ordinal();
    }
  }
}
