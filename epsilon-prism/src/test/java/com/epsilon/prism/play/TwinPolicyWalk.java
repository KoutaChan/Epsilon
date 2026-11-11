package com.epsilon.prism.play;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.agent.play.PolicySeat;
import com.epsilon.agent.play.Query;
import com.epsilon.agent.policy.ClientSeat;
import com.epsilon.agent.policy.Evaluation;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.TableWalk;
import com.epsilon.prism.core.Greedy;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.eval.EvaluationTable;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.testng.Assert;

/**
 * テスト用のヘルパー。PRISM の方策で新しいエンジンの判断を評価し、同値な打牌の切り方が 1 判断の評価（Reviewer）・クライアントの席（天鳳・
 * RiichiLab）・評価の卓の席（評価の卓・決闘・牌譜）でそろうかを確かめる。
 */
public final class TwinPolicyWalk {

  /** CPU の FP32 の eager で、PRISM のバージョンを評価の卓の入口から開く推論の節。 */
  private static final InferenceSettings CPU =
      new InferenceSettings(List.of("cpu"), Precision.FLOAT32, Kernels.EAGER, 8, 1, 1, 1);

  /**
   * 数えた判断。
   *
   * @param pairs 手出しとツモ切りの両方で切れる打牌がある判断
   * @param plain そのような打牌の無い判断
   * @param twins s の最大の手を、手出しでもツモ切りでも切れた判断
   */
  public record Counts(int pairs, int plain, int twins) {}

  private TwinPolicyWalk() {}

  /**
   * バージョンを評価の卓の入口（{@link EvaluationTable#open}）から開く。
   *
   * @param snapshot バージョンのディレクトリ
   * @param twinDiscard 同値な打牌のうち切る方
   */
  public static Policy open(Path snapshot, Variant variant, TwinDiscard twinDiscard) throws IOException {
    return EvaluationTable.open(
        snapshot.toAbsolutePath().toString(), variant, CPU, twinDiscard, PolicyOptions.defaults(), null);
  }

  /**
   * 種の決まった半荘の、合法手が 2 つ以上ある判断を評価する。手出しとツモ切りの両方で切れる打牌がある判断を pairs 個まで、無い判断を plain 個
   * まで。どの判断でも、評価の貪欲は s の最大を切る方の手に替えた手で、同値な打牌の s はビットまで同じ。s の最大を手出しでもツモ切りでも切れる
   * なら、貪欲は TSUMOGIRI ならツモ牌、TEDASHI なら手牌を切る手。クライアントの席は貪欲の手を返し、評価の卓の席の σ は貪欲の手の one-hot。
   *
   * @param policy PRISM の方策（閉じない）
   * @param twinDiscard 方策が同値な打牌のうち切る方
   * @param play 半荘の打ち方
   */
  public static Counts walk(
      Policy policy, TwinDiscard twinDiscard, TableWalk.Policy play, long seed, int games, int pairs, int plain) {
    Variant variant = policy.variant();
    ObservationWriter writer = new ObservationWriter(variant);
    ObservationBatch batch = new ObservationBatch(variant, 1 << 12);
    Observation reader = new Observation(variant);
    ClientSeat client = new ClientSeat(policy);
    PolicySeat table = new PolicySeat(policy);
    int[] counts = new int[3];
    new TableWalk(variant, play)
        .run(
            seed,
            games,
            (engine, slot) -> {
              Decisions decisions = engine.decisions();
              int n = decisions.legalCount(slot);
              if (n < 2) return;
              int[] legal = new int[n];
              for (int a = 0; a < n; a++) legal[a] = decisions.legal(slot, a);
              boolean pair = false;
              for (int a = 0; a < n && !pair; a++)
                pair = TwinDiscard.TSUMOGIRI.play(legal, a) != TwinDiscard.TEDASHI.play(legal, a);
              if (pair ? counts[0] >= pairs : counts[1] >= plain) return;
              counts[pair ? 0 : 1]++;
              batch.clear();
              int index = writer.capture(engine.view(decisions.seat(slot)), decisions, slot, batch);
              Observation o = reader.bind(batch, index);
              Evaluation evaluation = policy.evaluate(o);
              float[] values = evaluation.values();
              for (int a = 0; a < n; a++)
                Assert.assertEquals(
                    Float.floatToRawIntBits(values[TwinDiscard.TSUMOGIRI.play(legal, a)]),
                    Float.floatToRawIntBits(values[TwinDiscard.TEDASHI.play(legal, a)]),
                    "the twins of " + Action.of(legal[a]));
              int best = Greedy.argmax(values, null, n);
              int greedy = evaluation.greedy();
              Assert.assertEquals(greedy, twinDiscard.play(legal, best));
              int fromHand = TwinDiscard.TEDASHI.play(legal, best), fromDraw = TwinDiscard.TSUMOGIRI.play(legal, best);
              if (fromHand != fromDraw) {
                Assert.assertFalse(Action.of(legal[fromHand]).selection().tsumogiri());
                Assert.assertTrue(Action.of(legal[fromDraw]).selection().tsumogiri());
                Assert.assertEquals(Action.of(legal[fromDraw]).tile(), o.drawnTile());
                Assert.assertEquals(greedy, twinDiscard == TwinDiscard.TSUMOGIRI ? fromDraw : fromHand);
                counts[2]++;
              } else {
                Assert.assertEquals(greedy, best);
              }
              Assert.assertEquals(client.decide(engine.view(decisions.seat(slot)), decisions, slot), greedy);
              Query query = new Query(null, 0, slot, 0, batch, index, reader.bind(batch, index));
              var attempt = table.tryAcquireIngress(0);
              Assert.assertTrue(attempt.acquired());
              table.submit(List.of(query), attempt.ingress()).join();
              float[] sigma = new float[n];
              sigma[greedy] = 1f;
              Assert.assertEquals(query.sigma, sigma);
            });
    return new Counts(counts[0], counts[1], counts[2]);
  }
}
