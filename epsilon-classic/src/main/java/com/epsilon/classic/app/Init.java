package com.epsilon.classic.app;

import ai.djl.Device;
import ai.djl.ndarray.NDManager;
import ai.djl.pytorch.engine.PtEngine;
import com.epsilon.agent.app.Context;
import com.epsilon.agent.app.Named;
import com.epsilon.agent.app.Outcome;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.ClassicHeader;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.classic.model.ClassicNetwork;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Path;

/**
 * 初期化したままのネットワークの checkpoint を書く。初期値は乱数の種で決まり、学習のステップ数は 0。ネットワークは CPU で作る。
 *
 * @param out 書く checkpoint のディレクトリ。既にあれば何も書かず、使い方の誤りとして扱う
 */
@Named(value = "init", doc = "Write a checkpoint of a fresh network")
public record Init(
    @Key(doc = "Network architecture", presence = Presence.REQUIRED) Architecture architecture,
    @Key(doc = "Checkpoint directory to write (must not exist yet)", presence = Presence.REQUIRED) Path out,
    @Key(doc = "Rules", value = "TENHOU_4P") Variant variant,
    @Key(doc = "Random seed for the initial weights", value = "1") long seed)
    implements ClassicCommand {

  @Override
  public Outcome run(Context<ClassicSettings> context) throws IOException {
    return Outcome.ok(summary(write()));
  }

  /**
   * checkpoint を書き、そのヘッダーを返す。
   *
   * @throws SettingsException 書く checkpoint が既にあるとき
   */
  public ClassicHeader write() throws IOException {
    Inputs.requireNew(out, "output checkpoint");
    Networks.device("cpu", 0);
    try (NDManager manager = NDManager.newBaseManager(Device.cpu(), PtEngine.ENGINE_NAME)) {
      ClassicNetwork network = Networks.create(architecture, variant, manager, seed);
      ClassicCheckpoint.write(out, network, 0);
      return network.header(0);
    }
  }

  private static String summary(ClassicHeader header) {
    return "architecture=" + header.architecture() + " variant=" + header.variant() + " utility=" + header.utility();
  }
}
