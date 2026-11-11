package com.epsilon.prism.policy;

import static com.epsilon.mahjong.rules.Variant.TENHOU_4P;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.prism.TableWalk;
import org.testng.Assert;
import org.testng.annotations.Test;

/** ネットワークの方策の答え（{@link GreedyAnswer}）: s が NaN の手が先頭にあっても、σ は NaN でない手の貪欲の one-hot になる。 */
public class GreedyAnswerTest {

  @Test
  public void nanScoreInFrontIsNotTheGreedyAction() {
    PolicyRequest request = request(4);
    int n = request.legalCount();
    float[] scores = new float[n];
    scores[0] = Float.NaN;
    scores[1] = 0.3f;
    scores[2] = 0.3f;
    scores[3] = 0.1f;
    for (int a = 4; a < n; a++) scores[a] = -1f;
    // s の最大は手 1 と手 2 の同点なので、p0 を作る
    Assert.assertTrue(GreedyAnswer.needsPrior(request, scores));
    float[] prior = new float[n];
    prior[0] = 0.5f;
    prior[1] = 0.1f;
    prior[2] = 0.4f;
    GreedyAnswer.write(request, scores, prior);
    float[] sigma = new float[n];
    sigma[2] = 1f;
    Assert.assertEquals(request.sigma, sigma);

    PolicyRequest untied = request(4);
    float[] distinct = scores.clone();
    distinct[2] = 0.2f;
    Assert.assertFalse(GreedyAnswer.needsPrior(untied, distinct));
    GreedyAnswer.write(untied, distinct, null);
    sigma = new float[n];
    sigma[1] = 1f;
    Assert.assertEquals(untied.sigma, sigma);
  }

  /** エンジンで打った判断から、合法手が {@code actions} 個以上ある最初の判断の、retain でない要求を作る。 */
  private static PolicyRequest request(int actions) {
    ObservationBatch batch = new ObservationBatch(TENHOU_4P, 1 << 12);
    Observation reader = new Observation(TENHOU_4P);
    for (long[] words : PolicyRequests.observations(3, TableWalk.Policy.GREEDY, 64)) {
      if (reader.bind(words, 0).legalActions() >= actions) {
        int index = batch.add(reader.bind(words, 0));
        return new PolicyRequest(null, 0, index, null, null, false, 0, batch, index, reader.bind(batch, index));
      }
    }
    throw new IllegalStateException("no decision with " + actions + " legal actions");
  }
}
