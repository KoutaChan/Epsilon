package com.epsilon.pico.ai.grp;

import com.epsilon.ai.grp.EpsilonGrpExample;
import com.epsilon.ai.grp.EpsilonGrpFeature;
import com.epsilon.ai.grp.EpsilonGrpRanks;
import com.epsilon.pico.ai.decision.data.EpsilonDecisionSampleRecord;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Random;

/** Decision 対局生成から一意な GRP 学習例だけを抽出する。 */
public final class EpsilonGrpExamples {

  private EpsilonGrpExamples() {}

  /** 複数対局単位の学習データをまたいで使える収集処理。重複判定は (gameId, prefixSteps) 単位。 */
  public static final class Collector {
    private final Map<Key, EpsilonGrpExample> byKey = new LinkedHashMap<>();

    /** 空の GRP 学習例収集処理を作る。 */
    public Collector() {}

    /**
     * Decision サンプル群から有効な GRP の局履歴を追加する。
     *
     * @param samples 同一対局単位の学習データまたは複数対局単位の学習データの Decision サンプル
     */
    public void addAll(List<? extends EpsilonDecisionSampleRecord> samples) {
      for (EpsilonDecisionSampleRecord sample : samples) {
        add(sample);
      }
    }

    /**
     * Decision サンプルが有効な終局順位教師を持つ場合、その GRP の局履歴を追加する。
     *
     * @param sample 追加候補の Decision サンプル
     * @throws IllegalStateException 同じ対局・履歴長に異なる内容が現れた場合
     */
    public void add(EpsilonDecisionSampleRecord sample) {
      int finalRanksCode = sample.grpFinalRanksCode();
      float[] sequence = sample.grpFeatureSequence();
      int prefixSteps = EpsilonGrpFeature.steps(sequence);
      if (!EpsilonGrpRanks.isValidCode(finalRanksCode) || prefixSteps == 0) {
        return;
      }
      Key key = new Key(sample.gameId(), prefixSteps);
      EpsilonGrpExample existing = byKey.get(key);
      if (existing == null) {
        byKey.put(key, new EpsilonGrpExample(sample.gameId(), sequence, finalRanksCode));
        return;
      }
      if (existing.finalRanksCode() != finalRanksCode
          || !Arrays.equals(existing.sequence(), sequence)) {
        throw new IllegalStateException("Conflicting GRP examples for game/prefix: " + key);
      }
    }

    /**
     * 現在保持する一意な学習例数を返す。
     *
     * @return 一意化後の学習例数
     */
    public int size() {
      return byKey.size();
    }

    /**
     * 挿入順を保った変更不可スナップショットを返す。
     *
     * @return 一意化済み GRP 学習例
     */
    public List<EpsilonGrpExample> examples() {
      return List.copyOf(byKey.values());
    }

    /**
     * 一意化した後にリザーバーサンプリング（保持件数を固定する無作為抽出） するため、判断数が多い長い局を過大抽選しない。
     *
     * @param maxExamples 返す学習例数の上限。0 以下なら空
     * @param seed リザーバーサンプリング（保持件数を固定する無作為抽出） の乱数乱数シード
     * @return 上限内へ決定的に選抜した学習例
     */
    public List<EpsilonGrpExample> selectAtMost(int maxExamples, long seed) {
      return EpsilonGrpExamples.selectAtMost(examples(), maxExamples, seed);
    }
  }

  /**
   * Decision サンプル群から重複しない GRP 学習例を抽出する。
   *
   * @param samples GRP 特徴量と終局順位教師を含む Decision サンプル
   * @return 対局 ID と履歴長で一意化した学習例
   */
  public static List<EpsilonGrpExample> deduped(
      List<? extends EpsilonDecisionSampleRecord> samples) {
    Collector collector = new Collector();
    collector.addAll(samples);
    return collector.examples();
  }

  /**
   * Decision サンプル群を一意化してから指定数まで選抜する。
   *
   * @param samples GRP 特徴量と終局順位教師を含む Decision サンプル
   * @param maxExamples 返す学習例数の上限
   * @param seed リザーバーサンプリング（保持件数を固定する無作為抽出） の乱数乱数シード
   * @return 一意化・選抜済み学習例
   */
  public static List<EpsilonGrpExample> select(
      List<? extends EpsilonDecisionSampleRecord> samples, int maxExamples, long seed) {
    Collector collector = new Collector();
    collector.addAll(samples);
    return collector.selectAtMost(maxExamples, seed);
  }

  private static <T> List<T> selectAtMost(List<T> examples, int maxExamples, long seed) {
    if (maxExamples <= 0 || examples.isEmpty()) {
      return List.of();
    }
    if (examples.size() <= maxExamples) {
      return examples;
    }
    ArrayList<T> reservoir = new ArrayList<>(examples.subList(0, maxExamples));
    Random random = new Random(seed);
    for (int seen = maxExamples; seen < examples.size(); seen++) {
      int replacement = random.nextInt(seen + 1);
      if (replacement < maxExamples) {
        reservoir.set(replacement, examples.get(seen));
      }
    }
    return reservoir;
  }

  private record Key(long gameId, int prefixSteps) {}
}
