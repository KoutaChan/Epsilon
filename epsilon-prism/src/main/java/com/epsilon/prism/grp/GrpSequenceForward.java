package com.epsilon.prism.grp;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpInference;
import com.epsilon.ai.grp.EpsilonGrpNetwork;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;

/**
 * GRP（{@link EpsilonGrpNetwork}）の CPU の順伝播を公開の API（{@link EpsilonGrpNetwork#forwardLogits}）の上に書いたもの。推論は順伝播の
 * 行の数を常に maxBatch にし（詰め物の結果は捨てる）、学習は 1 半荘の全ての局末の接頭辞を 1 回の順伝播でまとめる。
 *
 * <p>学習の順伝播は網の子（GRU・投影・ロジットの頭）を {@link EpsilonGrpNetwork#forwardLogits} と同じ順で通す。GRU は因果的なので、半荘の
 * 全系列の時刻 k + 1 の出力は、局末 k の接頭辞（長さ k + 2）の最後の出力と同じ式になる（詰め物は後ろに置くので前の時刻に効かない）。
 */
public final class GrpSequenceForward {
  /** 網の子の名前の末尾（{@link EpsilonGrpNetwork} の構築子の名前）。 */
  private static final String GRU = "grpGru", PROJECTION = "projection", LOGITS = "rankMarginalLogitsHead";

  private GrpSequenceForward() {}

  /**
   * 局末の列の順位分布（席 i・順位 r が [4i + r]）。行の数 maxBatch の固定の塊に分けて計算する。
   *
   * <p>行の数が一つに決まれば値は行の位置・隣の系列・詰め物の長さに依らない（{@code GrpPhi} の決定性の約束）ので、詰め物を減らすために系列の
   * 長さの順に並べ替えて塊を作り、結果は入力の順に戻す。足りない行は最後の系列の写しで埋める。
   *
   * @param grp GRP の model（CPU）
   * @param maxBatch 1 回の順伝播の行の数
   * @param prefixes 次局 root の公開系列
   */
  public static float[][] marginals(Model grp, int maxBatch, List<float[]> prefixes) {
    int n = prefixes.size();
    float[][] out = new float[n][];
    if (n == 0) return out;
    Integer[] order = new Integer[n];
    for (int i = 0; i < n; i++) order[i] = i;
    Arrays.sort(order, Comparator.comparingInt(i -> prefixes.get(i).length));
    List<float[]> sorted = new ArrayList<>(n + maxBatch);
    for (int i : order) sorted.add(prefixes.get(i));
    while (sorted.size() % maxBatch != 0) sorted.add(sorted.getLast());
    List<float[]> marginals = new EpsilonGrpInference(grp, maxBatch).predictMarginalProbabilities(sorted);
    for (int i = 0; i < n; i++) out[order[i]] = marginals.get(i);
    return out;
  }

  /**
   * 半荘の列の、半荘が続く全ての局末の Sinkhorn 前のロジット（[Σ 局末, 16]、半荘の順・局末の順）。学習の損失に使う。
   *
   * @param manager 結果の NDArray の持ち主
   * @param training 学習の順伝播なら true
   */
  public static NDArray gameLogits(
      EpsilonGrpNetwork network,
      ParameterStore parameters,
      NDManager manager,
      List<RecordedGame> games,
      boolean training) {
    int width = EpsilonGrpFeature.FEATURE_SIZE;
    int time = 0;
    for (RecordedGame game : games) time = Math.max(time, game.continuingHandEnds() + 1);
    float[] sequences = new float[games.size() * time * width];
    int hiddenSize = network.hiddenSize();
    boolean[] continuing = new boolean[games.size() * time * hiddenSize];
    float[] first = HandStart.INITIAL.feature();
    for (int g = 0; g < games.size(); g++) {
      RecordedGame game = games.get(g);
      int base = g * time;
      System.arraycopy(first, 0, sequences, base * width, width);
      for (int k = 0; k < game.continuingHandEnds(); k++) {
        System.arraycopy(game.starts().get(k).feature(), 0, sequences, (base + k + 1) * width, width);
        Arrays.fill(continuing, (base + k + 1) * hiddenSize, (base + k + 2) * hiddenSize, true);
      }
    }
    NDArray input = manager.create(sequences, new Shape(games.size(), time, width));
    NDArray outputs = forward(network, GRU, parameters, input, training);
    // 選びの印は出力と同じ形にする（DJL の booleanMask は印の放送をしない）。選んだ要素は行の順に並ぶ
    NDArray mask = manager.create(continuing).reshape(games.size(), time, hiddenSize);
    NDArray states = outputs.booleanMask(mask).reshape(-1, hiddenSize);
    NDArray hidden = Activation.relu(forward(network, PROJECTION, parameters, states, training));
    return forward(network, LOGITS, parameters, hidden, training).reshape(-1, EpsilonGrpRanks.MATRIX_SIZE);
  }

  /** 名前の末尾が suffix の網の子の順伝播。 */
  private static NDArray forward(
      EpsilonGrpNetwork network, String suffix, ParameterStore parameters, NDArray input, boolean training) {
    for (var entry : network.getChildren())
      if (entry.getKey().endsWith(suffix))
        return entry.getValue().forward(parameters, new NDList(input), training, new PairList<>()).singletonOrThrow();
    throw new IllegalStateException("GRP network has no child " + suffix);
  }
}
