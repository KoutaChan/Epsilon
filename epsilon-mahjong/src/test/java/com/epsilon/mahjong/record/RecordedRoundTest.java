package com.epsilon.mahjong.record;

import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.tile.Tiles;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 牌譜の形式ごとに違うイベントの並びを、{@code RecordedRound} が決めた並び（表示牌と立直の成立の位置）に直すテスト。 */
public class RecordedRoundTest {

  private static final Meld DAIMINKAN =
      Meld.daiminkan(Tiles.S9, Meld.RelativeSource.KAMICHA, Meld.AkaSource.NONE);

  /** 9s（104）を大明槓するときに手から出す 3 枚。 */
  private static final List<Integer> DAIMINKAN_HAND = List.of(105, 106, 107);

  /**
   * 天鳳の並びでは、大明槓の表示牌は嶺上牌の後・打牌の前に、続く暗槓では保留の表示牌と暗槓の表示牌が嶺上牌の前に並び、立直の成立は
   * 宣言牌への鳴きの後に並ぶ。{@code RecordedRound} が決めた並びでは、大明槓の表示牌は続く暗槓の直後（保留、暗槓の順）に並び、立直の成立は
   * 宣言牌の直後で鳴きの前に並ぶ。
   */
  @Test
  public void revealsAndAcceptancesMoveToTheirContractPlaces() {
    List<RecordEvent> read =
        List.of(
            new RecordEvent.Draw(0, 4),
            new RecordEvent.Discard(0, 104, false, true),
            new RecordEvent.Call(1, DAIMINKAN.id(), DAIMINKAN_HAND),
            new RecordEvent.RiichiAccepted(0),
            new RecordEvent.Draw(1, 132),
            new RecordEvent.ClosedKan(1, Tiles.CHUN),
            new RecordEvent.DoraReveal(41),
            new RecordEvent.DoraReveal(45),
            new RecordEvent.Draw(1, 133),
            new RecordEvent.Discard(1, 133, true, false));
    Assert.assertEquals(
        RecordedRound.contractOrder(read),
        List.of(
            new RecordEvent.Draw(0, 4),
            new RecordEvent.Discard(0, 104, false, true),
            new RecordEvent.RiichiAccepted(0),
            new RecordEvent.Call(1, DAIMINKAN.id(), DAIMINKAN_HAND),
            new RecordEvent.Draw(1, 132),
            new RecordEvent.ClosedKan(1, Tiles.CHUN),
            new RecordEvent.DoraReveal(41),
            new RecordEvent.DoraReveal(45),
            new RecordEvent.Draw(1, 133),
            new RecordEvent.Discard(1, 133, true, false)));
  }

  /** 加槓の表示牌は、嶺上牌を引いた後の打牌の直後に並ぶ。 */
  @Test
  public void anAddedKanRevealsAfterTheNextDiscard() {
    List<RecordEvent> read =
        List.of(
            new RecordEvent.AddedKan(3, Tiles.CHUN),
            new RecordEvent.Draw(3, 35),
            new RecordEvent.DoraReveal(41),
            new RecordEvent.Discard(3, 18, false, false),
            new RecordEvent.Draw(0, 79));
    Assert.assertEquals(
        RecordedRound.contractOrder(read),
        List.of(
            new RecordEvent.AddedKan(3, Tiles.CHUN),
            new RecordEvent.Draw(3, 35),
            new RecordEvent.Discard(3, 18, false, false),
            new RecordEvent.DoraReveal(41),
            new RecordEvent.Draw(0, 79)));
  }

  /**
   * 大明槓の嶺上牌で引いた北を抜くと、保留の表示牌は北抜きへの応答が出そろった直後（補充のツモの前）にめくる。表示牌が補充のツモの後に
   * 並ぶ記録（次の打牌のときにめくった記録）は、エンジンより遅くめくっているので直せない。保留の表示牌が無ければ、北抜きがあっても並びは
   * 変えない。
   */
  @Test
  public void aNukiRevealsThePendingDoraBeforeItsReplacementDraw() {
    int north = Tiles.physicalId(Tiles.PEI, 1);
    List<RecordEvent> contract =
        List.of(
            new RecordEvent.Call(1, DAIMINKAN.id(), DAIMINKAN_HAND),
            new RecordEvent.Draw(1, north),
            new RecordEvent.Nuki(1, north),
            new RecordEvent.DoraReveal(73),
            new RecordEvent.Draw(1, 41),
            new RecordEvent.Discard(1, 41, true, false));
    Assert.assertEquals(RecordedRound.contractOrder(contract), contract);
    List<RecordEvent> late =
        List.of(
            new RecordEvent.Call(1, DAIMINKAN.id(), DAIMINKAN_HAND),
            new RecordEvent.Draw(1, north),
            new RecordEvent.Nuki(1, north),
            new RecordEvent.Draw(1, 41),
            new RecordEvent.DoraReveal(73),
            new RecordEvent.Discard(1, 41, true, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> RecordedRound.contractOrder(late));
    List<RecordEvent> plain =
        List.of(
            new RecordEvent.Draw(1, north),
            new RecordEvent.Nuki(1, north),
            new RecordEvent.Draw(1, 41),
            new RecordEvent.Discard(1, 41, true, false));
    Assert.assertEquals(RecordedRound.contractOrder(plain), plain);
  }

  /** 抜けるのは北だけ。 */
  @Test
  public void onlyANorthIsTakenOut() {
    Assert.assertThrows(IllegalArgumentException.class, () -> new RecordEvent.Nuki(0, Tiles.physicalId(Tiles.SHA, 0)));
    Assert.assertEquals(new RecordEvent.Nuki(2, 123).tile(), 123);
  }

  /** 鳴きが手から出す牌は、面子の鳴いた牌のほかの牌と 1 枚ずつ同じ種類の、重ならない物理牌。昇順に並べ直して持つ。 */
  @Test
  public void aCallTakesTheOtherTilesOfItsMeldFromTheHand() {
    Assert.assertEquals(new RecordEvent.Call(1, DAIMINKAN.id(), List.of(107, 105, 106)).consumed(), DAIMINKAN_HAND);
    Meld chi = Meld.chi(Tiles.P6, Tiles.P6, Meld.AkaSource.NONE);
    Assert.assertEquals(new RecordEvent.Call(1, chi.id(), List.of(66, 63)).consumed(), List.of(63, 66));
    for (List<Integer> consumed :
        List.of(List.of(105, 106), List.of(105, 106, 107, 104), List.of(105, 106, 103), List.of(105, 105, 106))) {
      Assert.assertThrows(IllegalArgumentException.class, () -> new RecordEvent.Call(1, DAIMINKAN.id(), consumed));
    }
    Assert.assertThrows(IllegalArgumentException.class, () -> new RecordEvent.Call(1, chi.id(), List.of(59, 63)));
  }

  /** 槓が無いのにめくった表示牌と、宣言牌の無い立直の成立は直せない。 */
  @Test
  public void revealsWithoutKansAndAcceptancesWithoutRiichiAreRejected() {
    List<RecordEvent> reveal =
        List.of(new RecordEvent.Draw(0, 4), new RecordEvent.DoraReveal(41), new RecordEvent.Discard(0, 4, true, false));
    Assert.assertThrows(IllegalArgumentException.class, () -> RecordedRound.contractOrder(reveal));
    Assert.assertThrows(
        IllegalArgumentException.class,
        () ->
            RecordedRound.contractOrder(
                List.of(new RecordEvent.Discard(0, 4, true, false), new RecordEvent.RiichiAccepted(0))));
  }
}
