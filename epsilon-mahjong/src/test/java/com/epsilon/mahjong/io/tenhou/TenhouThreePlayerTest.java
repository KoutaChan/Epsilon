package com.epsilon.mahjong.io.tenhou;

import com.epsilon.mahjong.engine.Scenes;
import com.epsilon.mahjong.record.RecordEvent;
import com.epsilon.mahjong.record.RecordedGame;
import com.epsilon.mahjong.record.RecordedGames;
import com.epsilon.mahjong.record.RecordedRound;
import com.epsilon.mahjong.record.RoundResult;
import com.epsilon.mahjong.record.SceneGames;
import com.epsilon.mahjong.rules.EndKind;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import java.util.Map;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * 天鳳の三人打ちの XML を書いて読むテスト。見本は牌山を手で並べてエンジンで打った局。北抜きと補充のツモ、保留の槓ドラ、抜いた北への
 * ロンを、書いたタグの並びと読み直した記録で確かめる。
 */
public class TenhouThreePlayerTest {

  private static final Variant VARIANT = Variant.TENHOU_3P;

  /**
   * 大明槓の嶺上牌の後に北を抜くと、大明槓の表示牌は補充のツモの前にめくる。書いた XML のタグも、北抜きの N の直後に DORA、その後に
   * 補充のツモの順に並び、読み直すと同じ記録になる。
   */
  @Test
  public void aNukiAfterAKanRevealsThePendingDoraBeforeItsReplacement() {
    RecordedGame game =
        RecordedGames.scene(
            Scenes.first(VARIANT)
                .hand(0, "9s19m258p147s36s5z6z")
                .hand(1, "999s4z123p456p78p1m")
                .hand(2, "19m689p258s147s37z")
                .draws("2z")
                .rinshan("5s1z"),
            "kan-then-nuki",
            "0:DAHAI(9s)",
            "1:DAIMINKAN(9s)",
            "1:NUKI");
    List<RecordEvent> events = game.rounds().get(0).events();
    int call = indexOf(events, RecordEvent.Call.class);
    Assert.assertEquals(
        events.subList(call, call + 5).stream().map(Object::getClass).toList(),
        List.of(
            RecordEvent.Call.class,
            RecordEvent.Draw.class,
            RecordEvent.Nuki.class,
            RecordEvent.DoraReveal.class,
            RecordEvent.Draw.class),
        events.toString());
    RecordEvent.Nuki nuki = (RecordEvent.Nuki) events.get(call + 2);

    String xml = TenhouXml.write(game);
    List<TenhouTag> tags = TenhouTag.parse(xml);
    String code = Integer.toString(TenhouMeldCode.encodeNuki(nuki.tile()));
    int at = tags.indexOf(new TenhouTag("N", Map.of("who", Integer.toString(nuki.seat()), "m", code)));
    Assert.assertTrue(at >= 0, xml);
    Assert.assertEquals(tags.get(at + 1).name(), "DORA", xml);
    TenhouTiles.TileTag draw = TenhouTiles.TileTag.parse(tags.get(at + 2).name(), VARIANT);
    Assert.assertEquals(draw.move(), TenhouTiles.Move.DRAW, xml);
    Assert.assertEquals(draw.seat(), nuki.seat(), xml);
    Assert.assertEquals(TenhouXml.read(xml, "kan-then-nuki"), game);
  }

  /** 抜いた北へのロンは、北を抜いた席が放銃したロンになる。書いて読み直すと同じ記録になり、局の最後のイベントは北抜き。 */
  @Test
  public void aRonOnANukiReadsAsRecorded() {
    RecordedGame game = SceneGames.ronOnNuki(35_000, 35_000, 35_000);
    RecordedGame read = TenhouXml.read(TenhouXml.write(game), "ron-on-nuki");
    Assert.assertEquals(read, game);
    RecordedRound round = read.rounds().get(0);
    List<RecordEvent> events = round.events();
    Assert.assertTrue(
        events.get(events.size() - 1) instanceof RecordEvent.Nuki last && last.seat() == 0, events.toString());
    RoundResult result = round.result();
    Assert.assertEquals(result.kind(), EndKind.RON);
    Assert.assertEquals(result.loser(), 0);
    Assert.assertEquals(result.wins().size(), 1);
    Assert.assertEquals(result.wins().get(0).seat(), 1);
  }

  /** 指定した型の最初のイベントの位置。 */
  private static int indexOf(List<RecordEvent> events, Class<? extends RecordEvent> type) {
    for (int index = 0; index < events.size(); index++) {
      if (type.isInstance(events.get(index))) {
        return index;
      }
    }
    throw new AssertionError("No " + type.getSimpleName() + " in " + events);
  }
}
