package com.epsilon.prism.grp;

import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.prism.collect.PhiCalibration;
import com.epsilon.prism.run.PrismConfig;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * 記録した局末の Φ を CPU でまとめて評価するもの（裏の採点・関門・段 1 の調べ。学習の JVM のラベルは {@code HandValue}）。GRP の順伝播の行の
 * 数は常に GRP の maxBatch で、学習の JVM と同じスレッド数なら同じ入力に同じ値を返す。
 */
public interface HandEndPhi extends AutoCloseable {

  /** 局末の列の Φ（入力の順）。 */
  PhiValues values(List<HandEndInput> inputs) throws IOException;

  /** 半荘の列の、半荘が続く全ての局末の Φ（半荘の順、半荘の中は局末の順）。 */
  default PhiValues gameValues(List<RecordedGame> games) throws IOException {
    List<HandEndInput> inputs = new ArrayList<>();
    for (RecordedGame game : games)
      for (int k = 0; k < game.continuingHandEnds(); k++) inputs.add(game.input(k));
    return values(inputs);
  }

  @Override
  default void close() {}

  /**
   * phi.jsonl の項目の Φ を開く。GRP の項目は正確なパスの checkpoint を読み（{@link CheckpointPhi}）、元の GRP の項目だけ温度と較正の表を
   * 重ねる。点数だけの項目は {@link ScorePhi}。
   *
   * @throws IOException checkpoint か較正の表が読めないとき
   */
  static HandEndPhi of(PhiEntry entry, EpsilonUtilityProfile profile) throws IOException {
    if (entry.kind().scoreBased()) return ScorePhi.offline(entry.kind(), profile);
    boolean fixed = entry.kind() == PhiEntryKind.FIXED_CHECKPOINT;
    return CheckpointPhi.open(
        Path.of(entry.path()),
        fixed ? PrismConfig.temperatures(entry.temperatures()) : new double[0],
        fixed && !entry.calibration().isBlank() ? PhiCalibration.load(Path.of(entry.calibration())) : null,
        profile);
  }
}
