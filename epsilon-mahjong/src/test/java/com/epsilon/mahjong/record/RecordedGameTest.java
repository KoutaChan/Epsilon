package com.epsilon.mahjong.record;

import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 半荘の出どころの情報（{@code RecordedGame.Meta}）は、ルールの席の数と同じ数の打ち手の名前を持つ。 */
public class RecordedGameTest {

  @Test
  public void aGameNamesEverySeat() {
    RecordedGame game = RecordedGames.play(Variant.TENHOU_4P, 3, "four");
    RecordedGame.Meta three = new RecordedGame.Meta("three", RecordedGame.Source.MJAI, List.of("A", "B", "C"));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () -> new RecordedGame(game.variant(), game.rounds(), game.finalScores(), three));
    RecordedGame.Meta four = new RecordedGame.Meta("four", RecordedGame.Source.MJAI, List.of("A", "B", "C", ""));
    Assert.assertEquals(new RecordedGame(game.variant(), game.rounds(), game.finalScores(), four).meta(), four);
  }
}
