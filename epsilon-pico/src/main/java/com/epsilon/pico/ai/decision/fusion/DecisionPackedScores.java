package com.epsilon.pico.ai.decision.fusion;

import ai.djl.engine.fusion.FusionOutputLease;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.index.NDIndex;

/**
 * 有効行のスコアと出力バッファの利用権を保持し、非同期転送が終わるまで再利用を防ぐ。
 *
 * <p>EAGER経路のテンソルはバッチ-局所的な 管理元が所有するため、このオブジェクトは借用ビューだけを保持する。FUSION経路ではセッションの最大容量出力から作った有効行ビューと
 * {@link FusionOutputLease}を保持し、非同期D2Hが完了するまで位置の再利用を防ぐ。
 */
public final class DecisionPackedScores implements AutoCloseable {

  private final NDArray activeRows;
  private FusionOutputLease lease;
  private AutoCloseable upstream;
  private boolean activeRowsReleased;
  private boolean resourcesReleased;

  private DecisionPackedScores(
      NDArray activeRows, boolean ownsActiveRows, FusionOutputLease lease, AutoCloseable upstream) {
    this.activeRows = activeRows;
    this.lease = lease;
    this.upstream = upstream;
    activeRowsReleased = !ownsActiveRows;
  }

  /**
   * バッチ専用のリソース管理オブジェクトが所有するEAGER テンソルを借用する。
   *
   * @param activeRows 有効行だけを持つ詰めたスコアテンソル
   * @return 解放してもテンソルを解放しない借用結果
   */
  static DecisionPackedScores borrowed(NDArray activeRows) {
    return borrowed(activeRows, null);
  }

  /** EAGER 詰めたテンソルと、その計算が消費した上流利用権を束ねる。 */
  static DecisionPackedScores borrowed(NDArray activeRows, AutoCloseable upstream) {
    return new DecisionPackedScores(activeRows, false, null, upstream);
  }

  /**
   * Fusion セッションの出力バッファの枠と、必要ならそこから作った有効行ビューを所有する。
   *
   * @param activeRows 有効行だけを表すテンソル
   * @param ownsActiveRows セッションの最大容量テンソルとは別のビューを所有するなら {@code true}
   * @param lease 出力バッファの枠の利用権
   * @return 利用権付き詰めたスコア
   */
  static DecisionPackedScores leased(
      NDArray activeRows, boolean ownsActiveRows, FusionOutputLease lease) {
    return leased(activeRows, ownsActiveRows, lease, null);
  }

  /** Fusion 出力利用権と、それより上流のアフィン変換利用権を依存順に所有する。 */
  static DecisionPackedScores leased(
      NDArray activeRows, boolean ownsActiveRows, FusionOutputLease lease, AutoCloseable upstream) {
    return new DecisionPackedScores(activeRows, ownsActiveRows, lease, upstream);
  }

  /**
   * Fusion セッションの最大容量記憶領域から、この投入済み処理で有効な先頭行だけのビューを作る。
   *
   * @param storage セッションが所有する最大容量出力テンソル
   * @param activeRowCount この投入済み処理の有効行数
   * @param maximumRows 記憶領域の最大行数
   * @param lease 出力バッファの枠の利用権
   * @return パディング行を除いた利用権付き詰めたスコア
   */
  static DecisionPackedScores leasedActiveRows(
      NDArray storage, int activeRowCount, int maximumRows, FusionOutputLease lease) {
    return leasedActiveRows(storage, activeRowCount, maximumRows, lease, null);
  }

  /** 最大容量出力から有効行ビューを作り、上流アフィン変換利用権も保持する。 */
  static DecisionPackedScores leasedActiveRows(
      NDArray storage,
      int activeRowCount,
      int maximumRows,
      FusionOutputLease lease,
      AutoCloseable upstream) {
    if (activeRowCount == maximumRows) {
      return leased(storage, false, lease, upstream);
    }
    return leased(storage.get(NDIndex.sliceAxis(0, 0, activeRowCount)), true, lease, upstream);
  }

  /**
   * パディング行を含まない詰めたスコアテンソルを返す。
   *
   * @return 形状が {@code [activeRows, packedWidth]} のテンソル
   */
  public NDArray activeRows() {
    return activeRows;
  }

  /**
   * ホストへ転送する有効要素数を返す。
   *
   * @return 有効行ビューの要素数
   */
  public int elementCount() {
    return Math.toIntExact(activeRows.size());
  }

  /**
   * 詰める処理後の例外経路で、このスコアを生成するデバイス処理の完了を待つ。
   *
   * <p>FUSION経路では利用権が記憶する投入済み処理 ストリームを待つ。EAGER経路ではテンソルをホストへ読み出すことで完了を確定する。このメソッドは利用権やテンソルを解放しない。
   */
  public void synchronizeForFailure() {
    FusionOutputLease currentLease = lease;
    if (currentLease != null) {
      currentLease.synchronize();
    } else {
      activeRows.toFloatArray();
    }
  }

  /**
   * 所有する有効行ビューを閉じた後、Fusion 出力バッファの枠を再利用可能にする。
   *
   * <p>このメソッドはデバイス同期を行わない。非同期D2Hへ渡した場合は、そのイベントが完了した後に呼び出す。
   */
  @Override
  public void close() {
    if (resourcesReleased) {
      return;
    }
    if (!activeRowsReleased) {
      activeRows.close();
      activeRowsReleased = true;
    }
    if (lease != null) {
      lease.close();
      lease = null;
    }
    if (upstream != null) {
      try {
        upstream.close();
        upstream = null;
      } catch (RuntimeException | Error failure) {
        throw failure;
      } catch (Exception failure) {
        throw new IllegalStateException(failure);
      }
    }
    resourcesReleased = true;
  }
}
