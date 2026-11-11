package com.epsilon.grp.train;

import com.epsilon.grp.GrpFixtures;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.grp.GrpState;
import com.epsilon.grp.GrpWorkspace;
import com.epsilon.grp.GruShape;
import com.epsilon.grp.GruSinkhorn;
import com.epsilon.grp.Phi;
import com.epsilon.grp.PhiTable;
import com.epsilon.grp.RoundHistory;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.TableState;
import com.epsilon.mahjong.rules.Variant;
import java.util.List;
import org.testng.Assert;
import org.testng.annotations.Test;

/**
 * Φ の項目から作る局末の値と、その値で単調性を調べる処理を確かめる。ネットワークの項目で作った値は、ネットワークを直接使って
 * 求めた値とビット単位で同じになる。GruSinkhorn の項目で作った値は、収集と同じく半荘の始めから局の履歴を積んで求めた Φ の値と、
 * ビット単位で同じになる。点数だけの項目で作った値も、Φ の値と同じになる。
 */
public class HandEndValuesTest {
  private static final Variant FOUR = Variant.TENHOU_4P;

  @Test
  public void networkEntriesTakeTheNetworkPath() {
    HandEnds ends = HandEnds.of(FOUR, RankUtility.TENHOU, TrainFixtures.simulated(FOUR, 200, 0, 41));
    GrpNetwork network = GrpFixtures.network(GrpFixtures.shape(FOUR, 8, false), 43, 0.5);
    HandEndValues direct = HandEndValues.of(ends, network, GrpWorkspace.of(network.shape()));
    HandEndValues fromPhi = HandEndValues.of(ends, Phi.of(network, RankUtility.TENHOU));
    Assert.assertTrue(fromPhi.joint());
    for (int h = 0; h < ends.handEnds(); h++) {
      Assert.assertEquals(Double.doubleToRawLongBits(fromPhi.jointLog(h)), Double.doubleToRawLongBits(direct.jointLog(h)));
      for (int seat = 0; seat < 4; seat++) {
        assertBits(fromPhi.phi(h, seat), direct.phi(h, seat), ends.state(h));
        for (int place = 0; place < 4; place++) {
          assertBits(fromPhi.marginal(h, seat, place), direct.marginal(h, seat, place), ends.state(h));
        }
      }
    }
    Assert.assertEquals(
        MonotoneProbe.of(ends).violationRate(Phi.of(network, RankUtility.TENHOU)),
        MonotoneProbe.of(ends).violationRate(network, GrpWorkspace.of(network.shape())));
  }

  @Test
  public void gruEntriesMatchTheCollectionHistory() {
    List<GameRecord> games = TrainFixtures.simulated(FOUR, 120, 0, 47);
    HandEnds ends = HandEnds.of(FOUR, RankUtility.TENHOU, games);
    GruSinkhorn gru = GrpFixtures.gru(new GruShape(FOUR, 8, 2), 53, 0.5);
    PhiTable phiTable =
        PhiTable.of(FOUR, new double[] {1, 1, 1, 1, 1.3, 1.3, 0.8, 0.8}, -1, 2.0 / 3, new double[][] {{0.01, -0.02, 0}});
    Phi phi = Phi.of(gru, RankUtility.TENHOU, phiTable);
    HandEndValues values = HandEndValues.of(ends, phi);
    Assert.assertFalse(values.joint());
    // 収集と同じく、半荘の始めの卓から始めて、局末ごとに値を求めてから次局の卓を積む
    Phi live = Phi.of(gru, RankUtility.TENHOU, phiTable);
    double[] expected = new double[4];
    int h = 0;
    for (GameRecord game : games) {
      RoundHistory history = new RoundHistory(FOUR);
      history.add(new TableState(FOUR));
      for (GrpState state : game.handEnds()) {
        live.values(history, state, expected);
        for (int seat = 0; seat < 4; seat++) {
          assertBits(values.phi(h, seat), expected[seat], state);
        }
        history.add(state);
        h++;
      }
    }
    Assert.assertEquals(h, ends.handEnds());

    // 選んだ局末の列でも、元の半荘の履歴を使う
    HandEnds selected = ends.select(end -> end % 3 == 1);
    HandEndValues chosen = HandEndValues.of(selected, phi);
    for (int index = 0; index < selected.handEnds(); index++) {
      for (int seat = 0; seat < 4; seat++) {
        assertBits(chosen.phi(index, seat), values.phi(3 * index + 1, seat), selected.state(index));
      }
    }

    // 単調性を調べるときも、元の局末までの履歴を積んで同じ値を測る
    MonotoneProbe probe = MonotoneProbe.of(ends);
    double[][] moved = new double[probe.moves().size()][4];
    RoundHistory history = new RoundHistory(FOUR);
    for (int index = 0; index < moved.length; index++) {
      MonotoneProbe.Move move = probe.moves().get(index);
      ends.history(move.handEnd(), history);
      live.values(history, move.state(), moved[index]);
    }
    Assert.assertEquals(probe.violationRate(phi), probe.violationRate(moved));
  }

  @Test
  public void scoreEntriesMatchThePhiValues() {
    HandEnds ends = HandEnds.of(FOUR, RankUtility.TENHOU, TrainFixtures.simulated(FOUR, 100, 0, 59));
    for (Phi phi : new Phi[] {Phi.scoreRank(FOUR, RankUtility.TENHOU), Phi.scoreSmooth(FOUR, RankUtility.TENHOU)}) {
      HandEndValues values = HandEndValues.of(ends, phi);
      double[] expected = new double[4];
      for (int h = 0; h < ends.handEnds(); h++) {
        phi.values(ends.state(h), expected);
        for (int seat = 0; seat < 4; seat++) {
          assertBits(values.phi(h, seat), expected[seat], ends.state(h));
        }
      }
    }
    Assert.assertThrows(
        IllegalArgumentException.class, () -> HandEndValues.of(ends, Phi.scoreRank(FOUR, RankUtility.AVERAGE_RANK)));
  }

  private static void assertBits(double actual, double expected, GrpState state) {
    Assert.assertEquals(Double.doubleToRawLongBits(actual), Double.doubleToRawLongBits(expected), state.toString());
  }
}
