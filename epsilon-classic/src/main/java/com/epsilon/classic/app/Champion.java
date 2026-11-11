package com.epsilon.classic.app;

import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.StoreEvent;
import com.epsilon.agent.run.RunDirectory;
import com.epsilon.agent.run.SettingsRows;
import com.epsilon.agent.run.TrainLock;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.io.PrintStream;
import java.nio.file.Path;
import java.util.List;
import java.util.Optional;

/**
 * campaign の今の王者と過去の王者を出す。初めの王者を置くこともできる: 王者がまだいない run だけで、campaign を回していない間に限る。
 * 初めの王者を置かずに campaign を始めると、バージョン 1（学習器の始めのネットワーク）が初めの王者になる。王者を出すだけのときは王者の
 * 保存先に何も書かないので、campaign を回している間でも使える。
 *
 * <p>置いた王者のチェックポイントはそのパスのまま参照する。run の外のチェックポイントなので、自己対局の相手や決闘の打ち手として開く
 * ときに、置いたときの SHA-256 と照合する。run のバージョンでない王者のバージョンは 0。
 *
 * @param run campaign の run のディレクトリ
 * @param initial 初めの王者にする classic のチェックポイント（無ければ王者を出すだけ）
 */
@Named(value = "champion", doc = "Show the current and past champions of a campaign; can set the first champion when there is none yet")
public record Champion(
    @Key(doc = "Campaign run directory", presence = Presence.REQUIRED) Path run,
    @Key(doc = "Classic checkpoint to make the first champion (only for a run with no champion yet)", presence = Presence.ABSENT) Path initial)
    implements ClassicCommand {

  @Override
  public Outcome run(Context<ClassicSettings> context) throws IOException {
    context.requireNoLayers();
    RunDirectory directory = new RunDirectory(run);
    ClassicSettings settings = new SettingsRows(directory).latest().bind(ClassicSettings.class).value();
    if (initial != null) {
      place(directory, settings.run().variant());
    }
    ChampionStore store = ChampionStore.view(directory.champion());
    Optional<ChampionStore.Champion> current = store.current();
    PrintStream out = context.out();
    if (current.isEmpty()) {
      out.println("No champion yet: the campaign makes version 1 the first champion when it starts");
      return Outcome.ok("no champion");
    }
    ChampionStore.Entry crowned = null;
    for (ChampionStore.Entry entry : store.history()) {
      if (entry.event() instanceof StoreEvent.Initial || entry.event() instanceof StoreEvent.Promoted) {
        crowned = entry;
      }
    }
    String by = crowned.event() instanceof StoreEvent.Promoted promoted ? "promoted by " + promoted.duel() : "initial";
    ChampionStore.Champion champion = current.get();
    out.println(
        "champion " + champion.label() + " version " + champion.version() + " since " + crowned.at() + " (" + by + ") "
            + champion.checkpoint());
    List<ChampionStore.Champion> past = store.past();
    for (int index = past.size() - 1; index >= 0; index--) {
      ChampionStore.Champion earlier = past.get(index);
      out.println("past " + earlier.label() + " version " + earlier.version() + " " + earlier.checkpoint());
    }
    return Outcome.ok("champion=" + champion.label() + " past=" + past.size());
  }

  /**
   * 初めの王者を置く。学習のロックを取って置くので、campaign と同時には書かない。
   *
   * @throws SettingsException campaign が回っている、チェックポイントが run のルールの classic のチェックポイントでない、王者がもういる
   */
  private void place(RunDirectory directory, Variant variant) throws IOException {
    Inputs.requireDirectory(initial, "initial");
    try {
      ClassicCheckpoint.verify(initial, variant);
    } catch (IllegalStateException | IOException mismatch) {
      throw new SettingsException(
          List.of("The first champion " + initial + " is not a classic checkpoint for " + variant + ": "
              + mismatch.getMessage()));
    }
    if (TrainLock.running(directory.trainLock())) {
      throw new SettingsException(
          List.of("The campaign " + directory.root() + " is running: stop it before placing the first champion"));
    }
    try (TrainLock lock = TrainLock.hold(directory.trainLock())) {
      ChampionStore store = ChampionStore.open(directory.champion());
      Optional<ChampionStore.Champion> current = store.current();
      if (current.isPresent()) {
        throw new SettingsException(
            List.of("The campaign " + directory.root() + " already has the champion " + current.get().label()));
      }
      Path checkpoint = initial.toAbsolutePath().normalize();
      ChampionStore.Champion first = ChampionStore.Champion.placed(checkpoint.getFileName().toString(), checkpoint, 0);
      store.append(new StoreEvent.Initial(first));
    }
  }
}
