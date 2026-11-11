package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.RoundEnd;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.ReplayDriver;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.record.SceneGames;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.ScoringYaku;
import com.epsilon.reviewer.engine.view.OutcomeProjector;
import com.epsilon.reviewer.engine.view.PlayerState;
import com.epsilon.reviewer.engine.view.RecordTable;
import com.epsilon.reviewer.engine.view.RoundOutcome;
import com.epsilon.reviewer.engine.view.TableState;
import com.epsilon.reviewer.engine.view.TileCode;
import com.epsilon.reviewer.engine.view.WinDetails;
import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.testng.Assert;
import org.testng.annotations.BeforeClass;
import org.testng.annotations.Test;

/**
 * 牌譜の局を表示の卓で進め、エンジンの局末から局の結果を作るテスト。確かめるのは、ダブロンの和了ごとの点、赤の牌のツモ・ロンや槍槓で
 * 和了った牌、流局、採点の内訳（役のコード・翻・符・点・役満・裏ドラ）。牌譜は、ランダムな手で打った自己対局の半荘（四人打ちと
 * 三人打ち）と、牌山を手で並べたダブロンと包の局を使う。
 */
public class RoundOutcomeTest {
  private static final String WINDS = "ESWN";

  /** 自己対局の半荘と手で並べた局の和了。点の内訳は記録にある。 */
  private final List<Win> played = new ArrayList<>();

  private final List<Draw> draws = new ArrayList<>();

  /** 和了 1 つと、その和了を精算する前と後の卓。 */
  private record Win(
      Variant variant,
      RecordedRound round,
      int index,
      RoundOutcome.Hora hora,
      RecordEvent last,
      TableState before,
      TableState after,
      int pao) {
    RoundResult.Win result() {
      return round.result().wins().get(index);
    }

    int gain(int seat) {
      return after.players().get(seat).score() - before.players().get(seat).score();
    }
  }

  private record Draw(TableState before, TableState after, int[] next, boolean gameOver) {}

  @BeforeClass
  public void replay() {
    for (long seed = 0; seed < 300; seed++)
      walk(RecordedGames.play(seed % 3 == 0 ? Variant.TENHOU_3P : Variant.TENHOU_4P, seed, "g" + seed), played);
    // 自己対局では起きにくいダブロン（本場と供託のある局）と包の和了は、牌山を手で並べた局で足す。
    for (RecordedGame game :
        List.of(
            SceneGames.doubleRon(2),
            SceneGames.doubleRon(0),
            SceneGames.threePlayerDoubleRon(),
            SceneGames.paoTsumo(),
            SceneGames.paoFeederRon(),
            SceneGames.paoSplitRon()))
      walk(game, played);
  }

  @Test
  public void eachRonOfADoubleRonKeepsItsOwnScore() {
    int doubles = 0;
    for (int i = 0; i + 1 < played.size(); i++) {
      Win first = played.get(i);
      Win second = played.get(i + 1);
      if (first.round() != second.round() || first.index() != 0) continue;
      doubles++;
      int loser = first.round().result().loser();
      Assert.assertEquals(first.hora().target(), loser);
      Assert.assertEquals(second.hora().target(), loser);
      Assert.assertEquals(first.hora().winningTile(), second.hora().winningTile());
      Assert.assertEquals(first.after().kyotaku(), 0, "the first winner takes the deposits");
      Assert.assertEquals(
          first.gain(second.hora().actor()), 0, "the second winner is paid by its own ron");
      int bonus = first.gain(first.hora().actor()) - first.hora().details().points();
      Assert.assertEquals(
          bonus >= first.before().kyotaku() * Variant.DEPOSIT, true, "honba and deposits");
      if (first.before().honba() == 0)
        Assert.assertEquals(bonus, first.before().kyotaku() * Variant.DEPOSIT);
      if (second.pao() < 0) {
        Assert.assertEquals(second.gain(second.hora().actor()), second.hora().details().points());
        Assert.assertEquals(second.gain(loser), -second.hora().details().points());
      }
    }
    Assert.assertTrue(doubles >= 3, "double rons: " + doubles);
  }

  @Test
  public void theWinningTileIsTheDrawTheDiscardOrTheRobbedKan() {
    int redTsumo = 0;
    int redRon = 0;
    int robbed = 0;
    for (Win win : played) {
      RoundOutcome.Hora hora = win.hora();
      String tile = hora.winningTile();
      PlayerState winner = win.before().players().get(hora.actor());
      boolean tsumo = win.round().result().kind() == EndKind.TSUMO;
      if (tsumo) {
        Assert.assertEquals(hora.target(), hora.actor());
        Assert.assertTrue(win.last() instanceof RecordEvent.Draw, win.last().toString());
        Assert.assertEquals(winner.draw(), tile);
        long sets = winner.melds().stream().filter(meld -> !meld.type().equals("NUKI")).count();
        Assert.assertEquals(winner.hand().size() + 3 * sets, 13);
        if (TileCode.red(tile)) {
          redTsumo++;
          Assert.assertFalse(winner.hand().contains(tile), "the red draw stays apart");
        }
        continue;
      }
      Assert.assertNull(winner.draw());
      switch (win.last()) {
        case RecordEvent.Discard discard -> {
          Assert.assertEquals(discard.seat(), hora.target());
          Assert.assertEquals(TileCode.of(win.variant().tiles(), discard.tile()), tile);
          Assert.assertEquals(win.before().players().get(hora.target()).river().getLast().tile(), tile);
          if (TileCode.red(tile)) redRon++;
        }
        case RecordEvent.AddedKan kan -> {
          robbed++;
          Assert.assertEquals(kan.seat(), hora.target());
          Assert.assertEquals(TileCode.kind(tile), kan.kind());
          Assert.assertTrue(codes(hora.details()).contains(ScoringYaku.CHANKAN.name()));
        }
        case RecordEvent.Nuki nuki -> {
          Assert.assertEquals(nuki.seat(), hora.target());
          Assert.assertEquals(TileCode.of(win.variant().tiles(), nuki.tile()), tile);
        }
        default -> Assert.fail("A ron after " + win.last());
      }
    }
    Assert.assertTrue(redTsumo > 0, "red tsumo wins");
    Assert.assertTrue(redRon > 0, "red ron wins");
    Assert.assertTrue(robbed > 0, "robbed kans");
  }

  @Test
  public void drawsCarryNoReasonOrPaymentAndMoveTheScores() {
    Assert.assertEquals(RoundOutcome.Ryukyoku.class.getRecordComponents().length, 0);
    Assert.assertEquals(OutcomeProjector.draw(), new RoundOutcome.Ryukyoku());
    Assert.assertFalse(draws.isEmpty());
    for (Draw draw : draws) {
      if (draw.gameOver()) continue;
      for (int seat = 0; seat < draw.next().length; seat++)
        Assert.assertEquals(draw.after().players().get(seat).score(), draw.next()[seat]);
      Assert.assertEquals(draw.after().kyotaku(), draw.before().kyotaku());
    }
  }

  @Test
  public void winDetailsMatchTheRecordedScoring() {
    int riichi = 0;
    int seatWinds = 0;
    for (Win win : played) {
      WinDetails details = win.hora().details();
      RoundResult.Win result = win.result();
      Set<String> codes = codes(details);
      Assert.assertEquals(codes.size(), details.yaku().size(), "codes are distinct");
      if (details.yakuman() > 0) {
        Assert.assertEquals(details.han(), 0);
        Assert.assertEquals(details.fu(), 0);
        for (WinDetails.Yaku yaku : details.yaku()) {
          Assert.assertEquals(yaku.han(), 0, yaku.code());
          Assert.assertTrue(yaku.yakuman() > 0, yaku.code());
        }
      } else {
        Assert.assertEquals(details.han(), result.han());
        Assert.assertEquals(details.fu(), result.fu());
        Assert.assertEquals(count(details, "AKADORA"), result.aka());
        Assert.assertEquals(count(details, "URADORA"), result.ura());
        if (win.variant() == Variant.TENHOU_4P) {
          Assert.assertEquals(
              details.yaku().stream().mapToInt(WinDetails.Yaku::han).sum(), details.han());
          Assert.assertEquals(count(details, "DORA"), result.dora());
        }
      }
      for (ScoringYaku yaku : ScoringYaku.values()) {
        boolean has = (result.yaku() & yaku.bit()) != 0;
        String code = switch (yaku) {
          case YAKUHAI_SEAT -> "YAKUHAI_SEAT_" + seatWind(win);
          case YAKUHAI_ROUND -> "YAKUHAI_ROUND_" + WINDS.charAt(win.round().start().round() / win.variant().seats());
          default -> yaku.name();
        };
        Assert.assertEquals(codes.contains(code), has, code);
        if (has && yaku == ScoringYaku.YAKUHAI_SEAT) seatWinds++;
      }
      boolean declared = (result.yaku() & (ScoringYaku.RIICHI.bit() | ScoringYaku.DOUBLE_RIICHI.bit())) != 0;
      if (declared) {
        riichi++;
        Assert.assertEquals(details.uraIndicators().size(), win.round().result().uraCount());
        for (String tile : details.uraIndicators()) Assert.assertTrue(TileCode.isValid(tile), tile);
      } else Assert.assertTrue(details.uraIndicators().isEmpty());
      boolean plain =
          win.before().honba() == 0 && win.before().kyotaku() == 0 && win.pao() < 0;
      if (plain && win.round().result().wins().size() == 1)
        Assert.assertEquals(win.gain(win.hora().actor()), details.points());
    }
    Assert.assertTrue(riichi > 0, "riichi wins");
    Assert.assertTrue(seatWinds > 0, "seat wind wins");
  }

  /** 局の最後の和了を精算した後の点は、局の始めの点に、記録にある点の移動を足した値（次の局の点）になる。 */
  @Test
  public void settledScoresAreTheNextRoundsScores() {
    for (int i = 0; i < played.size(); i++) {
      Win win = played.get(i);
      boolean last = i + 1 == played.size() || played.get(i + 1).round() != win.round();
      if (!last) continue;
      int[] deltas = win.round().result().deltas();
      TableState start = start(win);
      for (int seat = 0; seat < deltas.length; seat++)
        Assert.assertEquals(win.after().players().get(seat).score(), start.players().get(seat).score() + deltas[seat]);
    }
  }

  private static String seatWind(Win win) {
    int seats = win.variant().seats();
    int dealer = win.round().start().round() % seats;
    return String.valueOf(WINDS.charAt((win.hora().actor() - dealer + seats) % seats));
  }

  /** 局の始めの卓（配牌だけを置いた状態）。 */
  private static TableState start(Win win) {
    return new RecordTable(win.round()).state();
  }

  private static Set<String> codes(WinDetails details) {
    Set<String> codes = new HashSet<>();
    for (WinDetails.Yaku yaku : details.yaku()) codes.add(yaku.code());
    return codes;
  }

  private static int count(WinDetails details, String code) {
    for (WinDetails.Yaku yaku : details.yaku()) if (yaku.code().equals(code)) return yaku.han();
    return 0;
  }

  /** 半荘の全ての局を表示の卓とエンジンで進め、和了と流局を集める。 */
  private void walk(RecordedGame game, List<Win> into) {
    ReplayDriver driver = new ReplayDriver(game.variant());
    RuleWorkspace ws = new RuleWorkspace(game.variant());
    for (RecordedRound round : game.rounds()) {
      RecordTable table = new RecordTable(round);
      RecordEvent last = null;
      for (RecordEvent event : round.events()) {
        table.apply(event);
        if (!(event instanceof RecordEvent.DoraReveal) && !(event instanceof RecordEvent.RiichiAccepted))
          last = event;
      }
      Boundary boundary = driver.start(round, ws);
      while (boundary == Boundary.DECISIONS) boundary = driver.step(ws);
      RoundEnd end = driver.engine().roundEnd();
      RoundResult result = round.result();
      if (result.kind().win()) {
        for (int index = 0; index < result.wins().size(); index++) {
          TableState before = table.state();
          RoundOutcome.Hora hora = OutcomeProjector.win(table, result, end, index);
          table.settleWin(end, index);
          into.add(
              new Win(game.variant(), round, index, hora, last, before, table.state(), end.outcome().pao(index)));
        }
      } else {
        TableState before = table.state();
        table.settleDraw(end);
        int[] next = new int[game.variant().seats()];
        for (int seat = 0; seat < next.length; seat++) next[seat] = end.next().score(seat);
        draws.add(new Draw(before, table.state(), next, end.next().gameOver()));
      }
    }
  }
}
