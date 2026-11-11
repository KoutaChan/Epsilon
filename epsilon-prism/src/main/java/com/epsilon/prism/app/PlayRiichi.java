package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.app.RunFiles;
import com.epsilon.agent.policy.ClientSeat;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.mahjong.client.riichi.RiichiClient;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.DrawOrder;
import com.epsilon.mahjong.engine.Engine;
import com.epsilon.mahjong.engine.RuleWorkspace;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.model.PrismNetwork;
import com.epsilon.prism.run.Snapshot;
import com.epsilon.prism.settings.PrismSettings;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * 公開したバージョンで RiichiLab のランク戦を打つ（{@link RiichiClient}）。手は、バージョンの s が最も大きい手を選ぶ
 * （{@link ClientSeat}）。ルール（四人打ちか三人打ちか）は、バージョンのネットワークが読む入力の形式で決まる。接続の前に新しい半荘の
 * 最初の判断を 1 回打っておき、最初の手の遅れが対局に入らないようにする。
 *
 * @param model 打つバージョンのディレクトリか、run のディレクトリ（その最新のバージョンで打つ）
 * @param games 打ち終える対局の数（0 なら止めるまで打つ）
 * @param policy 方策を開く設定
 * @param token 接続のトークン（設定のファイルからだけ受け付ける）
 */
@Named(value = "play-riichi", doc = "Play RiichiLab ranked games with a published version")
public record PlayRiichi(
    @Key(doc = "Version directory to play, or a run directory (plays its latest version)", presence = Presence.REQUIRED) Path model,
    @Key(doc = "Number of games to finish (0: play until stopped)", value = "0") @Range(min = 0) int games,
    @Key(doc = "Settings for opening the policy") PolicyOptions policy,
    @Key(doc = "Connection token (settings file only)", presence = Presence.ABSENT, secret = true) String token)
    implements PrismCommand {

  private static final Logger LOG = LoggerFactory.getLogger(PlayRiichi.class);

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    if (token == null || token.isBlank()) {
      throw new SettingsException(
          List.of("Set play-riichi.token in the settings file before play-riichi"));
    }
    Path snapshot = snapshot(model);
    Variant variant = variant(snapshot);
    try (Policy opened = ModelFamily.of(Series.PRISM).open(snapshot, variant, policy)) {
      warmUp(opened);
      LOG.info("RiichiLab snapshot {} ready: variant={} games={}", snapshot, variant, games);
      RiichiClient.Result result = RiichiClient.play(token, variant, games, () -> new ClientSeat(opened));
      return Outcome.ok(
          "snapshot=" + snapshot + " completedGames=" + result.completedGames() + " interruptedGames="
              + result.interruptedGames());
    }
  }

  /** 打つバージョンのディレクトリを返す。{@value Snapshot#POLICY} があればそのまま返し、無ければ run のディレクトリとみて最新のバージョンを返す。 */
  static Path snapshot(Path directory) throws IOException {
    if (Files.isRegularFile(directory.resolve(Snapshot.POLICY))) {
      return directory;
    }
    return RunFiles.latestSnapshot(directory, Snapshot.POLICY);
  }

  /** バージョンの打つネットワークが読む入力の形式のルール（四人打ちか三人打ちか）を返す。 */
  static Variant variant(Path snapshot) throws IOException {
    return PrismNetwork.readFormat(Snapshot.model(snapshot)).variant();
  }

  /** 新しい半荘の最初の判断（親の第一打）を 1 回打つ。 */
  static void warmUp(Policy policy) {
    var engine = new Engine(policy.variant());
    var workspace = new RuleWorkspace(policy.variant());
    engine.startGame(0L, DrawOrder.SEAT_STREAMS);
    if (engine.advance(workspace) != Boundary.DECISIONS) {
      throw new IllegalStateException("A new game starts without a decision");
    }
    Decisions decisions = engine.decisions();
    new ClientSeat(policy).decide(engine.view(decisions.seat(0)), decisions, 0);
  }
}
