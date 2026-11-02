package com.epsilon.prism.grp;

import ai.djl.Device;
import ai.djl.Model;
import com.epsilon.ai.decision.EpsilonUtilityProfile;
import com.epsilon.ai.decision.EpsilonUtilityTargets;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.config.settings.GrpInferenceSettings;
import com.epsilon.nano.ai.grp.EpsilonGrpCheckpointManager;
import com.epsilon.nano.config.settings.EpsilonSettings;
import com.epsilon.prism.collect.GrpPhi;
import com.epsilon.prism.collect.PhiCalibration;
import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;

/**
 * GRP の checkpoint の Φ（CPU）。順位分布に温度を掛け（{@link GrpPhi#calibrated} と同じ式）、期待効用にしてから較正の表を掛ける。順伝播は
 * {@link GrpSequenceForward#marginals}（行の数は GRP の maxBatch）。
 *
 * <p>行の数・温度・効用・表の式が学習の JVM の {@link GrpPhi} と同じなので、torch のスレッド数が同じなら同じ入力に最後のビットまで同じ値を返す。
 */
public final class CheckpointPhi implements HandEndPhi {
  /** 読める構造の名前の頭（{@code EpsilonGrpNetwork.architectureSummary}）。 */
  private static final String ARCHITECTURE = "grp-epsilon-v3 ";

  private final Model model;
  private final int maxBatch;
  private final double[] temperatures;
  private final PhiCalibration calibration;
  private final EpsilonUtilityProfile profile;

  private CheckpointPhi(
      Model model, int maxBatch, double[] temperatures, PhiCalibration calibration, EpsilonUtilityProfile profile) {
    this.model = model;
    this.maxBatch = maxBatch;
    this.temperatures = temperatures;
    this.calibration = calibration;
    this.profile = profile;
  }

  /**
   * 正確なディレクトリの GRP を開く。
   *
   * @param grp grp-epsilon-v3 の checkpoint のディレクトリ（一番新しい子を選ばない）
   * @param temperatures 次局の局番号ごとの温度（空なら掛けない）
   * @param calibration 較正の表（null なら掛けない）
   * @throws IOException 読めないか、構造が grp-epsilon-v3 でないとき
   */
  public static CheckpointPhi open(
      Path grp, double[] temperatures, PhiCalibration calibration, EpsilonUtilityProfile profile)
      throws IOException {
    for (double temperature : temperatures)
      if (!(temperature > 0)) throw new IllegalArgumentException("Φ temperatures must be positive");
    if (!profile.rankBased()) throw new IllegalArgumentException("Φ needs a rank utility");
    String architecture = EpsilonGrpCheckpointManager.loadManifest(grp).architecture;
    if (!architecture.startsWith(ARCHITECTURE))
      throw new IOException("GRP checkpoint " + grp + " is not grp-epsilon-v3: " + architecture);
    int maxBatch = EpsilonSettings.defaults().bind(GrpInferenceSettings.class).maxBatch();
    return new CheckpointPhi(
        EpsilonGrpCheckpointManager.load(grp, Device.cpu()), maxBatch, temperatures.clone(), calibration, profile);
  }

  /** 1 回の順伝播の行の数（学習の JVM の GRP と同じ設定の値）。 */
  public int maxBatch() {
    return maxBatch;
  }

  @Override
  public PhiValues values(List<HandEndInput> inputs) throws IOException {
    List<float[]> prefixes = new ArrayList<>(inputs.size());
    for (HandEndInput input : inputs) prefixes.add(input.prefix());
    float[][] marginals = GrpSequenceForward.marginals(model, maxBatch, prefixes);
    double[][] utilities = new double[inputs.size()][];
    double[][] ranks = new double[inputs.size()][16];
    for (int i = 0; i < inputs.size(); i++) {
      float[] prefix = prefixes.get(i);
      float[] tempered = tempered(marginals[i], prefix);
      double[] raw = new double[4];
      for (int seat = 0; seat < 4; seat++)
        raw[seat] = EpsilonUtilityTargets.expectedRankUtilityTrustedDouble(profile, tempered, seat * 4);
      utilities[i] = calibration == null ? raw : calibration.apply(raw, prefix);
      for (int k = 0; k < 16; k++) ranks[i][k] = tempered[k];
    }
    return new PhiValues(utilities, ranks);
  }

  /** {@link GrpPhi#calibrated} と同じ式（次局の局番号の温度 T で席ごとに p' ∝ p^(1/T)、T = 1 か温度が空ならそのまま）。 */
  private float[] tempered(float[] probabilities, float[] prefix) {
    if (temperatures.length == 0) return probabilities;
    int base = prefix.length - EpsilonGrpFeature.FEATURE_SIZE;
    int kyoku = Math.round(prefix[base + EpsilonGrpFeature.KYOKU_INDEX] * EpsilonGrpFeature.KYOKU_SCALE);
    double temperature = temperatures[Math.max(0, Math.min(kyoku, temperatures.length - 1))];
    if (temperature == 1) return probabilities;
    float[] result = new float[probabilities.length];
    for (int seat = 0; seat < 4; seat++) {
      double total = 0;
      double[] weights = new double[4];
      for (int rank = 0; rank < 4; rank++) {
        weights[rank] = Math.pow(Math.max(probabilities[seat * 4 + rank], 1e-9), 1 / temperature);
        total += weights[rank];
      }
      for (int rank = 0; rank < 4; rank++) result[seat * 4 + rank] = (float) (weights[rank] / total);
    }
    return result;
  }

  @Override
  public void close() {
    model.close();
  }
}
