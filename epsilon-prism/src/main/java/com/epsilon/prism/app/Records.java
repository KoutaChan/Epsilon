package com.epsilon.prism.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.play.TableLoop;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.Kernels;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.eval.EvaluationTable;
import com.epsilon.prism.play.SelfPlayRecords;
import com.epsilon.prism.settings.PrismSettings;
import java.nio.file.Path;
import java.util.Locale;

/**
 * 自己対局の半荘を牌譜に書く（{@link SelfPlayRecords}）。候補の打ち手を指定した数の席に座らせ、残りの席には相手の打ち手を座らせる。打ち手の
 * 指定の仕方は evaluate と同じで、heuristic も候補にできる。牌山の種は run の種とソルトの排他的論理和にし、半荘 ID は指定した最初の値から
 * 振る。推論と卓の設定、ルールは設定の層から読む。
 *
 * @param model 候補の打ち手
 * @param out 書き出すディレクトリ
 * @param opponent 相手の打ち手
 * @param seats 候補を座らせる席の数（無ければ全ての席）
 * @param games 半荘の数
 * @param firstGame 最初の半荘 ID
 * @param nanoDevice classic の打ち手のデバイス（無ければルールごとの既定値）
 */
@Named(value = "records", doc = "Write self-play games as game records")
public record Records(
    @Key(doc = "Candidate player (same forms as evaluate; heuristic also works)", presence = Presence.REQUIRED) String model,
    @Key(doc = "Output directory", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Opponent player", value = EvaluationTable.HEURISTIC) String opponent,
    @Key(doc = "Number of seats for the candidate (default: every seat)", presence = Presence.ABSENT) @Range(min = 1) Integer seats,
    @Key(doc = "Number of games", value = "1024") @Range(min = 1) int games,
    @Key(doc = "First game ID", value = "0") @Range(min = 0) long firstGame,
    @Key(doc = "Device of the classic players (cpu or gpuN; default: the per-rule default)", presence = Presence.ABSENT) String nanoDevice)
    implements PrismCommand {

  /** 牌山の種に混ぜるソルト。評価の牌山と重ならないようにする。 */
  public static final long SALT = 0x5245_434F_5244_5331L;

  @Override
  public Outcome run(Context<PrismSettings> context) throws Exception {
    PrismSettings settings = context.settings().value();
    settings.runtime().apply();
    settings.diagnostics().apply();
    Variant variant = settings.run().variant();
    InferenceSettings inference = settings.inference();
    PolicyOptions options =
        new PolicyOptions(
            nanoDevice != null ? nanoDevice : settings.module().opponents().defaultDevice(),
            inference.precision(),
            inference.maxBatch(),
            Kernels.EAGER);
    TwinDiscard twinDiscard = settings.behaviour().twinDiscard();
    try (var runtime = new NetworkRuntime(inference.encoders(), inference.slots(), inference.readyBatches());
        var table = new TableLoop(settings.table());
        var candidate = EvaluationTable.open(model, variant, inference, twinDiscard, options, runtime);
        var rival = EvaluationTable.open(opponent, variant, inference, twinDiscard, options, runtime)) {
      SelfPlayRecords.Result result =
          SelfPlayRecords.write(
              table,
              new SelfPlayRecords.Player(candidate, EvaluationTable.label(model)),
              new SelfPlayRecords.Player(rival, EvaluationTable.label(opponent)),
              seats == null ? variant.seats() : seats,
              games,
              settings.run().seed() ^ SALT,
              firstGame,
              out);
      return Outcome.ok(
          String.format(
              Locale.ROOT,
              "games=%d rounds=%d seconds=%.1f games/s=%.2f",
              result.games(),
              result.rounds(),
              result.seconds(),
              result.games() / result.seconds()));
    }
  }
}
