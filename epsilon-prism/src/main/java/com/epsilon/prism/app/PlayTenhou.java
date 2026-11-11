package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.policy.ClientSeat;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.mahjong.client.tenhou.TenhouClient;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.settings.PrismSettings;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 天鳳の個室で打つ（{@link TenhouClient}）。手は方策で貪欲に選ぶ（{@link ClientSeat}）。半荘ごとに接続し直す。打ち手の指定の仕方は
 * evaluate と同じ。
 *
 * @param model 打ち手（バージョンのディレクトリ、prism:DIR、nano:DIR、major:DIR、heuristic）
 * @param room 個室の番号
 * @param name 天鳳で名乗る名前
 * @param games 打つ半荘の数
 * @param variant ルール（四人打ちか三人打ちか）
 * @param policy 方策を開く設定
 */
@Named(value = "play-tenhou", doc = "Play in a Tenhou private lobby")
public record PlayTenhou(
    @Key(doc = "Player (version directory, prism:DIR, nano:DIR, major:DIR, heuristic)", presence = Presence.REQUIRED)
        String model,
    @Key(doc = "Lobby number", presence = Presence.REQUIRED) String room,
    @Key(doc = "Name shown on Tenhou", value = "NoName") String name,
    @Key(doc = "Number of games to play (reconnects for each game)", value = "1") @Range(min = 1) int games,
    @Key(doc = "Rules (four-player or three-player)", value = "TENHOU_4P") Variant variant,
    @Key(doc = "Settings for opening the policy") PolicyOptions policy)
    implements PrismCommand {

  private static final Logger LOG = LoggerFactory.getLogger(PlayTenhou.class);

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    Series.Spec spec = EvaluationTable.spec(model);
    long mismatched = 0;
    try (Policy opened = ModelFamily.of(spec.series()).open(spec.checkpoint(), variant, policy)) {
      LOG.info("Tenhou room {} as {}: {} with {}", room, name, opened.label(), policy);
      for (int game = 0; game < games; game++) {
        int rounds = TenhouClient.play(TenhouClient.SERVER, variant, room, name, new ClientSeat(opened));
        mismatched += rounds;
        LOG.info("Tenhou game {} finished: mismatchedRounds={}", game + 1, rounds);
      }
    }
    return Outcome.ok("games=" + games + " mismatchedRounds=" + mismatched);
  }
}
