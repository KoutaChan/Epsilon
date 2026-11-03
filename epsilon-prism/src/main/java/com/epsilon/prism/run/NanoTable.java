package com.epsilon.prism.run;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.collect.CollectionDriver;
import com.epsilon.prism.eval.Arena;
import com.epsilon.prism.eval.NanoService;
import com.epsilon.prism.eval.PlayStats;
import com.epsilon.prism.policy.PlayerPolicy;
import com.epsilon.prism.policy.PolicyService;
import com.epsilon.spi.PolicyExecutionContext;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 候補の打ち手を 1 席、相手を 3 席に置く卓（{@link Arena#detailed}。候補の席は半荘 ID で回し、全員貪欲。牌山の種は seed ^ {@link
 * Arena#EVALUATION_SALT}、半荘 ID は {@link Arena#HEURISTIC_BASE} から）。evaluate と系譜の x1n3 が、打ち手の書き方・半荘ごとの記録・集計の行を
 * ここで共有する。
 *
 * <p>打ち手の書き方: 版のディレクトリ（PRISM の打つ網。{@link Snapshot#open}）、{@value #HEURISTIC}（規則の打ち手）、{@value #NANO}DIR
 * （Epsilon-Nano の Decision チェックポイント。測定専用）。
 */
public final class NanoTable {

  /** 規則の打ち手。 */
  public static final String HEURISTIC = "heuristic";

  /** Nano の打ち手の接頭辞。 */
  public static final String NANO = "nano:";

  /** 系譜の x1n3 の Nano のデバイス（Nano は GPU 1 枚で打つ）。 */
  public static final String NANO_DEVICE = "gpu:0";

  /** 内訳の行は 1 行（NaN もそのまま書く）。 */
  private static final Gson LINES = new GsonBuilder().serializeSpecialFloatingPointValues().create();

  private NanoTable() {}

  /**
   * 版 snapshot の打つ網 1 席対 opponent 3 席で games 局打つ。網の推論は config の収集のデバイス・バッチ・推論の道。
   *
   * @param stats 局と判断の内訳を足す先
   */
  public static Arena.Detailed play(
      PrismConfig config,
      Path snapshot,
      String opponent,
      int games,
      CollectionDriver driver,
      PolicyExecutionContext execution,
      String nanoDevice,
      PlayStats stats)
      throws Exception {
    try (var candidate = open(snapshot.toString(), config, execution, nanoDevice);
        var rivals = open(opponent, config, execution, nanoDevice)) {
      return Arena.detailed(
          driver,
          candidate,
          rivals,
          label(opponent),
          games,
          config.seed() ^ Arena.EVALUATION_SALT,
          Arena.HEURISTIC_BASE,
          EpsilonUtilityProfile.valueOf(config.utility()),
          stats);
    }
  }

  /** 打ち手の書き方から方策を開く（{@value #HEURISTIC}・{@value #NANO}DIR・版のディレクトリ）。 */
  public static PolicyService open(
      String name, PrismConfig config, PolicyExecutionContext execution, String nanoDevice)
      throws IOException {
    if (name.equals(HEURISTIC)) return PlayerPolicy.heuristic();
    if (name.startsWith(NANO))
      return NanoService.open(Path.of(name.substring(NANO.length())), nanoDevice, execution);
    return Snapshot.open(
        Path.of(name),
        PrismConfig.devices(config.collectDevices()),
        execution,
        config.inferBatch(),
        config.priorSettings(),
        config.inference());
  }

  /** 系譜の x1n3 の相手に書ける打ち手（{@value #HEURISTIC} か {@value #NANO}&lt;絶対パス&gt;）なら true。 */
  public static boolean validOpponent(String name) {
    return name.equals(HEURISTIC)
        || (name.startsWith(NANO) && Path.of(name.substring(NANO.length())).isAbsolute());
  }

  /** 結果の相手の名前（heuristic、nano:名前、snapshot:名前）。 */
  public static String label(String name) {
    if (name.equals(HEURISTIC)) return name;
    if (name.startsWith(NANO)) return NANO + Path.of(name.substring(NANO.length())).getFileName();
    return "snapshot:" + Path.of(name).getFileName();
  }

  /** 半荘ごとの候補の結果（1 行 {@code 半荘 ID\t着順\t効用}、半荘 ID の順）。 */
  public static String perGame(Arena.Game[] games) {
    StringBuilder text = new StringBuilder(games.length * 32);
    for (Arena.Game game : games)
      text.append(game.gameId()).append('\t').append(game.rank()).append('\t').append(game.utility()).append('\n');
    return text.toString();
  }

  /** 内訳の行（{@code evaluate play {…}}）。 */
  public static String playLine(PlayStats stats) {
    return "evaluate play " + LINES.toJson(stats.report());
  }

  /** 集計の行（{@code opponent=… games=… rank=R±E utility=U±E shares=…}）。 */
  public static String summary(Arena.Result result) {
    return String.format(
        "opponent=%s games=%d rank=%.3f±%.3f utility=%.3f±%.3f shares=%.3f/%.3f/%.3f/%.3f",
        result.opponent(),
        result.games(),
        result.meanRank(),
        result.rankError(),
        result.meanUtility(),
        result.utilityError(),
        result.rankShares()[0],
        result.rankShares()[1],
        result.rankShares()[2],
        result.rankShares()[3]);
  }
}
