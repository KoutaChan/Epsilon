package com.epsilon.prism.policy;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Action.TileSelection;
import com.epsilon.mahjong.action.Action.Type;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.ObservedDecisions;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.Threats;
import com.epsilon.prism.core.Twins;
import java.util.Random;
import org.testng.Assert;
import org.testng.annotations.Test;

/** 公開情報だけで測って最も安全な打牌と、方針の分かれ目ごとの代表の手をテストする。当てはまる分かれ目では、両側の手を必ず返す。 */
public class RegimeRepresentativesTest {
  private static final int M1 = 0, M2 = 1, M4 = 3, M5 = 4, M6 = 5, P2 = 10, TON = 27;

  private static int dahai(int kind) {
    return Action.discard(kind, false, false, false);
  }

  /** 脅威が無ければ O を読まないので、鳴き・立直・槓の分かれ目は、合法手と s・p0 だけで決まる。 */
  @Test
  public void callRiichiAndKanRegimes() {
    int[] calls = {
      Action.chi(M1, M2, false), Action.code(Type.PON, M2, TileSelection.NON_AKA), Action.code(Type.PASS)
    };
    Assert.assertEquals(
        RegimeRepresentatives.of(null, calls, new float[] {0.2f, 0.5f, 0.1f}, null, 0), new int[] {1, 2});
    int[] riichi = {dahai(M5), dahai(TON), dahai(P2), Action.discard(M1, false, false, true), dahai(M1)};
    Assert.assertEquals(
        RegimeRepresentatives.of(null, riichi, new float[] {0.1f, 0f, -0.2f, 0.3f, 0.05f}, null, 0),
        new int[] {3, 0});
    int[] kans = {
      Action.code(Type.ANKAN, M4, TileSelection.NOT_APPLICABLE), dahai(M4), dahai(M6), Action.code(Type.TSUMO_AGARI)
    };
    Assert.assertEquals(
        RegimeRepresentatives.of(null, kans, new float[] {0.1f, 0.4f, 0.2f, 0.9f}, null, 0), new int[] {0, 3});
    int[] plain = {dahai(M1), dahai(M2)};
    Assert.assertEquals(RegimeRepresentatives.of(null, plain, new float[] {0f, 1f}, null, 0), new int[0]);
  }

  /**
   * エンジンで打った判断で確かめる。最も安全な手は打牌から選び、脅威が無ければ p0 が最大の打牌になる。脅威があれば、(現物でない立直者の
   * 数, 脅威の危険度の最大) が全ての打牌の中で辞書式に最小の打牌になる。脅威の分かれ目では、代表の手の最後に、牌効率が最善の打牌と
   * 最も安全な打牌をこの順に足す。打牌が無ければ、最も安全な手は -1。
   */
  @Test
  public void safestIsTheLexicographicMinimumOverDiscards() {
    SeatFacts facts = new SeatFacts(Variant.TENHOU_4P);
    TileCounts counts = new TileCounts(Variant.TENHOU_4P);
    Random random = new Random(5);
    int riichiChecked = 0, quiet = 0;
    for (ObservedDecisions.Decision decision : ObservedDecisions.collect(13, 4, TableWalk.Policy.GREEDY, 20_000)) {
      Observation o = decision.observation();
      int[] legal = decision.legal();
      float[] prior = new float[legal.length];
      float[] scores = new float[legal.length];
      for (int a = 0; a < legal.length; a++) {
        prior[a] = random.nextFloat();
        scores[a] = random.nextFloat();
      }
      int safest = RegimeRepresentatives.safest(o, legal, prior);
      int efficient = -1;
      for (int a = 0; a < legal.length; a++)
        if (Twins.discard(Action.of(legal[a])) && (efficient < 0 || prior[a] > prior[efficient])) efficient = a;
      if (efficient < 0) {
        Assert.assertEquals(safest, -1);
        continue;
      }
      Assert.assertTrue(Twins.discard(Action.of(legal[safest])));
      facts.load(o);
      counts.load(o);
      int threats = Threats.count(o, facts);
      int[] chosen = RegimeRepresentatives.of(o, legal, scores, prior, threats);
      if (threats == 0) {
        Assert.assertEquals(safest, efficient);
        quiet++;
        continue;
      }
      Assert.assertEquals(chosen[chosen.length - 2], efficient);
      Assert.assertEquals(chosen[chosen.length - 1], safest);
      long best = Long.MAX_VALUE;
      for (int a = 0; a < legal.length; a++)
        if (Twins.discard(Action.of(legal[a])))
          best = Math.min(best, risk(o, facts, counts, Action.of(legal[a]).tile()));
      Assert.assertEquals(risk(o, facts, counts, Action.of(legal[safest]).tile()), best);
      for (int seat = 1; seat < 4; seat++) if (o.riichi(seat) != RiichiState.NONE) riichiChecked++;
    }
    Assert.assertTrue(riichiChecked > 50, "riichi " + riichiChecked);
    Assert.assertTrue(quiet > 100, "quiet " + quiet);
  }

  /** (現物でない立直者の数, 脅威の危険度の最大) を、辞書式の順を保ったまま 1 つの数にする。 */
  private static long risk(Observation o, SeatFacts facts, TileCounts counts, int kind) {
    int unsafe = 0, danger = 0;
    for (int seat = 1; seat < 4; seat++) {
      if (!Threats.isThreat(o, facts, seat)) continue;
      if (o.riichi(seat) != RiichiState.NONE && !Tiles.contains(facts.safe(seat), kind)) unsafe++;
      danger = Math.max(danger, facts.danger(seat, kind, counts));
    }
    return (long) unsafe << 32 | danger;
  }
}
