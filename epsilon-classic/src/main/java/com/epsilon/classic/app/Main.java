package com.epsilon.classic.app;

import com.epsilon.agent.app.App;
import com.epsilon.agent.app.Application;
import com.epsilon.agent.eval.TableSpec;
import com.epsilon.agent.policy.Series;
import com.epsilon.classic.model.ClassicCheckpoint;
import com.epsilon.grp.GrpCheckpoint;
import com.epsilon.grp.GrpModel;
import com.epsilon.grp.GrpNetwork;
import com.epsilon.mahjong.rules.Variant;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;

/** classic の入口。初期化したままのチェックポイントを書く、牌譜から行を作る、模倣と価値を事前学習する、GRP を学習する、自己対局の campaign を回す、王者を出す・置く、のコマンドを持つ。 */
public final class Main {

  private Main() {}

  /** classic のアプリの宣言。 */
  public static Application<ClassicSettings> application() {
    return Application.of("classic", ClassicSettings.class, List.of(ClassicCommand.class)).withCheck(Main::problems);
  }

  /** コマンドを 1 つ実行し、終了コードで JVM を終える。残ったスレッドのせいで JVM が終われなくなるのを防ぐため。 */
  public static void main(String[] arguments) {
    App.main(application(), arguments);
  }

  /**
   * 設定が参照するファイルを確かめる: campaign の始めのネットワークと GRP、評価の卓の相手のチェックポイント。GRP は、ルールが同じで、自己対局が
   * 読める種類（GrpNetwork）であること。
   */
  private static List<String> problems(ClassicSettings settings) {
    List<String> problems = new ArrayList<>();
    Variant variant = settings.run().variant();
    Path start = settings.run().start();
    if (start != null) {
      try {
        ClassicCheckpoint.verify(start, variant);
      } catch (IOException | IllegalStateException unusable) {
        problems.add(
            "The start network run.start " + start + " is not a usable classic checkpoint: " + unusable.getMessage());
      }
    }
    Path grp = settings.run().grp();
    if (grp != null) {
      try {
        GrpModel model = GrpCheckpoint.read(grp);
        if (model.variant() != variant) {
          problems.add("The GRP run.grp " + grp + " is for " + model.variant() + ", not " + variant);
        }
        if (!(model instanceof GrpNetwork)) {
          problems.add(
              "The GRP run.grp " + grp + " must be a network GRP, not " + model.getClass().getSimpleName()
                  + ": the self-play reads only a network GRP");
        }
      } catch (IOException unusable) {
        problems.add("The GRP run.grp " + grp + " is not a usable GRP checkpoint: " + unusable.getMessage());
      }
    }
    for (Map.Entry<String, TableSpec> entry : settings.evaluation().tables().entrySet()) {
      if (TableSpec.kind(entry.getValue().opponent()) == TableSpec.Opponent.CHECKPOINT) {
        Series.Spec spec = Series.parse(entry.getValue().opponent());
        if (!Files.isDirectory(spec.checkpoint())) {
          problems.add(
              "The opponent checkpoint of evaluation table " + entry.getKey() + " does not exist: " + spec.checkpoint());
        }
      }
    }
    return problems;
  }
}
