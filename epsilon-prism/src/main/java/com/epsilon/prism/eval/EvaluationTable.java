package com.epsilon.prism.eval;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.DuelSetup;
import com.epsilon.agent.policy.ModelFamily;
import com.epsilon.agent.policy.Policy;
import com.epsilon.agent.policy.PolicyOptions;
import com.epsilon.agent.policy.Series;
import com.epsilon.agent.runtime.InferenceSettings;
import com.epsilon.agent.runtime.NetworkRuntime;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.prism.core.TwinDiscard;
import com.epsilon.prism.policy.InferencePath;
import com.epsilon.prism.policy.NetworkPolicy;
import com.epsilon.prism.policy.PrismPolicy;
import com.epsilon.prism.run.Snapshot;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * 評価の卓の打ち手を、打ち手の指定から開く。evaluate と系譜の評価のジョブ（{@link EvaluationJobs}）は、打ち手の指定の読み方と、決闘の計画に書く
 * 打ち手をここで共有する。どの相手を受け付けるかは、四人打ち・三人打ちごとのモジュールが決める（{@link OpponentRules}）。
 *
 * <p>打ち手の指定（{@link Series#parse}）は次のどれか: バージョンのディレクトリか prism:DIR（PRISM の打つネットワーク。{@link Snapshot#open}）、
 * {@value #HEURISTIC}（ルールベースの打ち手）、nano:DIR・major:DIR（classic のチェックポイント。測定だけに使う）。PRISM のバージョンは、推論の
 * セクションのデバイス・バッチの上限・計算の型・推論のバックエンドで開く。classic の方策は、渡した方策の設定で開く。どちらも同じ実行のリソースに載る。
 */
public final class EvaluationTable {

  /** ルールベースの打ち手。 */
  public static final String HEURISTIC = "heuristic";

  private EvaluationTable() {}

  /**
   * 打ち手の指定から方策を開く。
   *
   * @param inference PRISM のバージョンを開く推論のセクション
   * @param twinDiscard PRISM のバージョンが同値な打牌のうち切る方
   * @param options classic の方策を開く設定
   */
  public static Policy open(
      String name,
      Variant variant,
      InferenceSettings inference,
      TwinDiscard twinDiscard,
      PolicyOptions options,
      NetworkRuntime runtime)
      throws IOException {
    Series.Spec spec = spec(name);
    if (spec.series() == Series.PRISM) {
      NetworkPolicy network =
          Snapshot.open(
              spec.checkpoint(),
              inference.networkDevices().devices(),
              runtime,
              inference.maxBatch(),
              inference.precision(),
              InferencePath.of(inference.kernels()));
      return new PrismPolicy(network, variant, Series.PRISM.label(spec.checkpoint()), null, twinDiscard);
    }
    PolicyOptions seat = spec.series().checkpointed() ? options : options.onDevice("cpu");
    return ModelFamily.of(spec.series()).open(spec.checkpoint(), variant, seat, runtime);
  }

  /**
   * 打ち手の指定を読む。接頭辞の無い指定は、バージョンのディレクトリとみなす。
   *
   * @throws IllegalArgumentException 知らない接頭辞か、パスが空のとき
   */
  public static Series.Spec spec(String name) {
    int separator = name.indexOf(':');
    if (name.equals(HEURISTIC) || (separator > 0 && knownPrefix(name.substring(0, separator)))) {
      return Series.parse(name);
    }
    return new Series.Spec(Series.PRISM, Path.of(name));
  }

  /**
   * 打ち手の指定にあるチェックポイントのパスを、絶対パスに直す。チェックポイントを開かない打ち手の指定は、そのまま返す。
   *
   * @throws IllegalArgumentException チェックポイントのディレクトリが無いとき
   */
  public static String absolute(String name) {
    Series.Spec spec = spec(name);
    if (!spec.series().checkpointed()) {
      return name;
    }
    Path checkpoint = spec.checkpoint().toAbsolutePath().normalize();
    if (!Files.isDirectory(checkpoint)) {
      throw new IllegalArgumentException("No checkpoint directory at " + checkpoint + " for " + name);
    }
    int separator = name.indexOf(':');
    boolean prefixed = separator > 0 && knownPrefix(name.substring(0, separator));
    return prefixed ? name.substring(0, separator + 1) + checkpoint : checkpoint.toString();
  }

  /** 結果に書く打ち手の名前を返す（heuristic、nano:名前、major:名前、prism:名前）。 */
  public static String label(String name) {
    Series.Spec spec = spec(name);
    return spec.series().label(spec.checkpoint());
  }

  /**
   * 決闘の計画に書く、系譜の外の打ち手を作る。チェックポイントを開く打ち手には、今のチェックポイントの SHA-256 を添える。
   *
   * @throws IOException チェックポイントが読めないとき
   */
  public static DuelSetup.Entrant entrant(String name) throws IOException {
    Series.Spec spec = spec(name);
    String sha256 = spec.series().checkpointed() ? CheckpointFiles.sha256(spec.checkpoint()) : null;
    return new DuelSetup.Entrant(name, spec.series().label(spec.checkpoint()), null, sha256);
  }

  /**
   * 計画に書いた系譜の外の打ち手のチェックポイントが、書いたときの中身のままかを確かめる。系譜のバージョンは書き換えないので確かめない。
   *
   * @throws IOException チェックポイントが読めないか、SHA-256 が計画と違うとき
   */
  public static void verify(DuelSetup.Entrant entrant) throws IOException {
    Series.Spec spec = spec(entrant.spec());
    if (entrant.version() != null || !spec.series().checkpointed()) {
      return;
    }
    String actual = CheckpointFiles.sha256(spec.checkpoint());
    if (!actual.equals(entrant.sha256())) {
      throw new IOException(
          "The checkpoint " + spec.checkpoint() + " has SHA-256 " + actual + ", but the duel was planned with "
              + entrant.sha256());
    }
  }

  private static boolean knownPrefix(String prefix) {
    for (Series series : Series.values()) {
      if (series.prefix().equals(prefix)) {
        return true;
      }
    }
    return false;
  }
}
