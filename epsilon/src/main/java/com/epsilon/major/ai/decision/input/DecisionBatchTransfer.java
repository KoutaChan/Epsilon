package com.epsilon.major.ai.decision.input;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.pytorch.engine.PtCopyEvent;
import ai.djl.pytorch.engine.PtNDArray;
import ai.djl.pytorch.engine.PtNDManager;
import ai.djl.pytorch.engine.PtPinnedBuffer;
import com.epsilon.config.settings.DecisionTensorTransfer;
import com.epsilon.runtime.InputBatchProfile;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

/**
 * ホスト側の連続バッファをデバイスへ転送し、共通の配置定義からテンソルのビューを構築する。
 *
 * <p>入力はカテゴリ値と数値を一回ずつ転送し、{@link DecisionNetworkInputs}が同じ{@link
 * DecisionInputLayout}から論理ビューを構築する。各テンソルの要素数・オフセット・形状をこの転送層で再計算しないため、ホストとモデルの 配置契約が分岐しない。
 *
 * <p>カテゴリ値 IDは{@link DataType#INT16} 連続バッファとして転送し、論理ビューにもそのまま保持する。Embeddingや
 * 指定位置の抽出で広いインデックスが必要な箇所だけがモデル内で局所的に拡張するため、転送バイト数とデバイス側記憶領域の両方を半減できる。
 *
 * <p>{@link Workspace}を使う経路は実行枠ごとのページ固定バッファまたはダイレクトバッファを再利用し、{@link
 * DecisionHostBatch.RowSlice}を一時バッチへデータを復元せずに転送する。返却テンソルの寿命は転送先管理元に従う。
 * 推論時のプレイヤーの履歴表現、遷移、方策実行インデックスは一つのINT32本の連続バッファへ詰め、デバイス上のビューとして結び付ける。
 */
public final class DecisionBatchTransfer {

  private static final int VALUE_TARGET_STRIDE = DecisionTrainingTargets.VALUE_TARGET_SIZE + 1;

  private DecisionBatchTransfer() {}

  /**
   * 完全なホスト側バッチをダイレクトバッファ経路で転送する簡易API。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostBatch 容量まで確定済みのホスト側バッチ
   * @return 論理ビューと任意の学習教師値を所有するデバイス側バッチ
   */
  public static DecisionDeviceBatch transferTrainingToDevice(
      NDManager manager, DecisionHostBatch hostBatch) {
    return transferTrainingToDevice(
        manager,
        hostBatch.sliceRows(0, hostBatch.size()),
        null,
        DecisionTensorTransfer.DIRECT_BUFFER);
  }

  /**
   * 指定した連続行をFLOAT32でデバイスへ転送し、入力と任意の学習教師値ビューを構築する。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostRows 容量まで確定済みバッチの連続行部分領域
   * @param workspace 実行枠専用の再利用バッファ。{@code null} なら一時バッファを作る
   * @param transferMode ページ固定バッファまたはダイレクトバッファ転送方式
   * @return 論理ビューと任意の学習教師値を所有するデバイス側バッチ
   */
  public static DecisionDeviceBatch transferTrainingToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode) {
    return transferTrainingToDevice(manager, hostRows, workspace, transferMode, DataType.FLOAT32);
  }

  /**
   * 数値連続バッファを指定精度へデバイス上で一度だけ変換してから論理ビューを構築する。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostRows 容量まで確定済みバッチの連続行部分領域
   * @param workspace 実行枠専用の再利用バッファ。{@code null} なら一時バッファを作る
   * @param transferMode ページ固定バッファまたはダイレクトバッファ転送方式
   * @param inputNumericType モデルへ渡す数値入力のデバイスデータ種類
   * @return 論理ビューと任意の学習教師値を所有するデバイス側バッチ
   */
  public static DecisionDeviceBatch transferTrainingToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode,
      DataType inputNumericType) {
    requireExactlyFilled(hostRows);

    Slabs slabs =
        workspace == null
            ? Slabs.regular(manager, hostRows)
            : workspace.transferTraining(manager, hostRows, transferMode);
    return bindTrainingSlabs(
        hostRows.size(),
        hostRows.bucket(),
        slabs.inputCategories(),
        slabs.inputNumerics(),
        slabs.playerMemoryPresentIndices(),
        slabs.transitionPresentIndices(),
        slabs.targetCategories(),
        slabs.targetNumerics(),
        hostRows.hasTrainingTargets(),
        inputNumericType);
  }

  /** 完全なホスト側バッチを存在する遷移だけに絞った通常実行用の推論入力としてデバイスへ転送する。 */
  public static DecisionInferenceDeviceBatch transferInferenceToDevice(
      NDManager manager, DecisionHostBatch hostBatch) {
    return transferInferenceToDevice(
        manager,
        hostBatch.sliceRows(0, hostBatch.size()),
        null,
        DecisionTensorTransfer.DIRECT_BUFFER,
        DataType.FLOAT32);
  }

  /** 数値連続バッファを指定精度へ一度だけ変換する存在する遷移だけに絞った通常実行用の推論転送。 */
  public static DecisionInferenceDeviceBatch transferInferenceToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode,
      DataType inputNumericType) {
    requireExactlyFilled(hostRows);
    InferenceSlabs slabs =
        workspace == null
            ? InferenceSlabs.regular(manager, hostRows)
            : workspace.transferInference(manager, hostRows, transferMode);
    NDArray inputNumerics = convertInputNumerics(slabs.inputNumerics(), inputNumericType);
    DecisionInferenceInputs inputs =
        DecisionInferenceInputs.bind(slabs.layout(), slabs.inputCategories(), inputNumerics);
    return new DecisionInferenceDeviceBatch(
        slabs.inputCategories(),
        inputNumerics,
        inputs,
        slabs.playerMemoryPresentIndices(),
        slabs.transitionPresentIndices(),
        slabs.policyExecutionIndices());
  }

  /**
   * 実行枠専用ページ固定のバッファへ複数の非所有行ビューを直接抽出した状態です。
   *
   * <p>記述情報はバッファを所有しません。同じ{@link Workspace}で次の処理段階を始める前に、H2D イベントの完了を保証する必要があります。
   */
  public static final class StagedInference {
    private final Workspace workspace;
    private final DecisionHostBatch.RowBatch rows;
    private final DecisionHostBatch.InferenceIndexLayout indexLayout;
    private final DecisionInferenceInputLayout inputLayout;

    private StagedInference(
        Workspace workspace,
        DecisionHostBatch.RowBatch rows,
        DecisionHostBatch.InferenceIndexLayout indexLayout,
        DecisionInferenceInputLayout inputLayout) {
      this.workspace = workspace;
      this.rows = rows;
      this.indexLayout = indexLayout;
      this.inputLayout = inputLayout;
    }

    /** 抽出済みの合計行数を返します。 */
    public int rows() {
      return rows.size();
    }

    /** 抽出済み入力の型付き容量区分を返します。 */
    public DecisionBucket bucket() {
      return rows.bucket();
    }

    /** H2D対象の合計バイト数を返します。 */
    public long inputByteCount() {
      return inputLayout.inputByteCount() + (long) indexLayout.totalCount() * Integer.BYTES;
    }
  }

  /** 行動が一意な価値推論用に状態入力だけを抽出した状態です。 */
  public static final class StagedState {
    private final Workspace workspace;
    private final DecisionHostBatch.RowBatch rows;
    private final int playerMemoryPresentCount;

    private StagedState(
        Workspace workspace, DecisionHostBatch.RowBatch rows, int playerMemoryPresentCount) {
      this.workspace = workspace;
      this.rows = rows;
      this.playerMemoryPresentCount = playerMemoryPresentCount;
    }

    /** 抽出済みの合計行数を返します。 */
    public int rows() {
      return rows.size();
    }

    /** H2D対象の合計バイト数を返します。 */
    public long inputByteCount() {
      return rows.stateByteCount() + (long) playerMemoryPresentCount * Integer.BYTES;
    }
  }

  /**
   * 固定形状のアクセラレーターグラフが再利用する入力記憶領域を作る。
   *
   * <p>ホスト側のページ固定のバッファとデバイス側の二本の元の連続バッファはグラフの寿命中同じアドレスを保つ。各再生前に{@link
   * GraphInput#refresh(DecisionHostBatch.RowSlice)}で内容だけを更新し、数値精度変換だけを記録対象にする。
   *
   * @param manager グラフを記録するPyTorch デバイス管理元
   * @param hostRows グラフの固定形状を決める、容量まで確定済み行部分領域
   * @param workspace グラフの寿命中専有する再利用バッファ
   * @param inputNumericType 記録内でモデルへ渡す数値データ種類
   * @return 固定アドレスのグラフ入力
   */
  public static GraphInput prepareGraphInput(
      PtNDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DataType inputNumericType) {
    requireExactlyFilled(hostRows);
    return workspace.prepareGraphInput(manager, hostRows, inputNumericType);
  }

  /**
   * 推論に必要な標準形式の状態だけをデバイスへ転送する。
   *
   * <p>行動が一意に決まる局面の価値推論では行動/遷移連続バッファを転送せず、方策ネットワーク部分を型として呼び出せない {@link DecisionStateInputs}を返す。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostRows 容量まで確定済みバッチの連続行部分領域
   * @param workspace 実行枠専用の再利用バッファ。{@code null} なら一時バッファを作る
   * @param transferMode ページ固定バッファまたはダイレクトバッファ転送方式
   * @return FLOAT32の状態のみのデバイス入力
   */
  public static DecisionStateInputs transferStateToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode) {
    return transferStateToDevice(manager, hostRows, workspace, transferMode, DataType.FLOAT32);
  }

  /**
   * 状態数値連続バッファを指定精度へデバイス上で一度だけ変換する価値推論用転送。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostRows 容量まで確定済みバッチの連続行部分領域
   * @param workspace 実行枠専用の再利用バッファ。{@code null} なら一時バッファを作る
   * @param transferMode ページ固定バッファまたはダイレクトバッファ転送方式
   * @param inputNumericType モデルへ渡す数値入力のデバイスデータ種類
   * @return 状態のみのデバイス入力
   */
  public static DecisionStateInputs transferStateToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode,
      DataType inputNumericType) {
    requireExactlyFilled(hostRows);
    StateSlabs slabs =
        workspace == null
            ? StateSlabs.regular(manager, hostRows)
            : workspace.transferState(manager, hostRows, transferMode);
    return new DecisionStateInputs(
        slabs.stateCategories().reshape(hostRows.size(), DecisionInputSchema.STATE_INT_COUNT),
        convertInputNumerics(slabs.stateNumerics(), inputNumericType)
            .reshape(hostRows.size(), DecisionInputSchema.STATE_FLOAT_COUNT),
        convertInputNumerics(slabs.boundaryContext(), inputNumericType)
            .reshape(hostRows.size(), DecisionBoundaryContext.INPUT_SIZE));
  }

  private static NDArray convertInputNumerics(NDArray numerics, DataType dataType) {
    if (dataType == DataType.FLOAT32) {
      return numerics;
    }
    NDArray converted = numerics.toType(dataType, false);
    numerics.close();
    return converted;
  }

  /**
   * 既にデバイスへ転送された標準形式の学習連続バッファを論理ビューへ結び付ける。
   *
   * <p>自己対局による入力パイプラインはヒープ上の {@link DecisionHostBatch} を経由せず、この境界へ転送済み連続バッファを渡す。保存データによる
   * 事前学習の配列転送と論理的な配置を共有し、方策/価値側に別実装を作らない。
   *
   * @param rows バッチ行数
   * @param bucket 行動形状を固定する容量区分
   * @param inputCategories 標準形式の入力のカテゴリ連続バッファ
   * @param inputNumerics 標準形式の入力の数値連続バッファ。型変換時はこのメソッドが閉じる
   * @param playerMemoryPresentIndices プレイヤーの履歴表現の有効トークンインデックス
   * @param transitionPresentIndices 行動遷移の有効トークンインデックス
   * @param targetCategories 学習教師値のカテゴリ連続バッファ
   * @param targetNumerics 学習教師値の数値連続バッファ
   * @param hasTrainingTargets 教師値連続バッファを論理的な教師値へ結び付けるか
   * @param inputNumericType モデルへ渡す数値入力型
   * @return 渡されたデバイス連続バッファとその論理ビューを所有する学習バッチ
   */
  public static DecisionDeviceBatch bindTrainingSlabs(
      int rows,
      DecisionBucket bucket,
      NDArray inputCategories,
      NDArray inputNumerics,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      NDArray targetCategories,
      NDArray targetNumerics,
      boolean hasTrainingTargets,
      DataType inputNumericType) {
    if (rows <= 0) {
      throw new IllegalArgumentException("training slab rows must be positive: " + rows);
    }
    NDArray convertedInputNumerics = convertInputNumerics(inputNumerics, inputNumericType);
    DecisionInputLayout layout = new DecisionInputLayout(rows, bucket);
    DecisionNetworkInputs inputs =
        DecisionNetworkInputs.bind(layout, inputCategories, convertedInputNumerics);
    DecisionDeviceBatch.TrainingTargets trainingTargets =
        hasTrainingTargets ? bindTargets(rows, bucket, targetCategories, targetNumerics) : null;
    return new DecisionDeviceBatch(
        inputCategories,
        convertedInputNumerics,
        inputs,
        playerMemoryPresentIndices,
        transitionPresentIndices,
        trainingTargets);
  }

  /**
   * 価値のみ学習に必要な状態とスカラー効用教師値だけをデバイスへ転送する。
   *
   * <p>行動、遷移、探索適用後の方策、対局生成方策、方策教師値は転送しない。大きい合法候補容量区分でも転送量は
   * 状態スキーマと行数だけで決まり、価値ネットワーク部分の計算内容は完全な{@link DecisionDeviceBatch}を渡す場合と同一になる。
   *
   * @param manager 転送先デバイスを所有する管理元
   * @param hostRows 学習教師値を持つ、容量まで確定済み行部分領域
   * @param workspace 実行枠専用の再利用バッファ。{@code null} なら一時バッファを作る
   * @param transferMode ページ固定バッファまたはダイレクトバッファ転送方式
   * @return 状態入力・効用教師値・サンプル重みだけを持つデバイス側バッチ
   */
  public static DecisionValueDeviceBatch transferValueToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode) {
    return transferValueToDevice(manager, hostRows, workspace, transferMode, DataType.FLOAT32);
  }

  /** 状態数値入力だけを順伝播精度へ変換し、価値教師値はFLOAT32のまま保持する。 */
  public static DecisionValueDeviceBatch transferValueToDevice(
      NDManager manager,
      DecisionHostBatch.RowSlice hostRows,
      Workspace workspace,
      DecisionTensorTransfer transferMode,
      DataType inputNumericType) {
    requireExactlyFilled(hostRows);
    if (!hostRows.hasTrainingTargets()) {
      throw new IllegalArgumentException("Value transfer requires training targets");
    }
    ValueSlabs slabs =
        workspace == null
            ? ValueSlabs.regular(manager, hostRows)
            : workspace.transferValue(manager, hostRows, transferMode);
    int rows = hostRows.size();
    DecisionStateInputs inputs =
        new DecisionStateInputs(
            slabs.stateCategories().reshape(rows, DecisionInputSchema.STATE_INT_COUNT),
            convertInputNumerics(slabs.stateNumerics(), inputNumericType)
                .reshape(rows, DecisionInputSchema.STATE_FLOAT_COUNT),
            convertInputNumerics(slabs.boundaryContext(), inputNumericType)
                .reshape(rows, DecisionBoundaryContext.INPUT_SIZE));
    NDArray targets = slabs.valueTargets().reshape(rows, VALUE_TARGET_STRIDE);
    int valueEnd = DecisionTrainingTargets.VALUE_TARGET_SIZE;
    return new DecisionValueDeviceBatch(
        inputs,
        slabs.playerMemoryPresentIndices(),
        targets.get("...,0"),
        targets.get("...," + valueEnd));
  }

  private static void requireExactlyFilled(DecisionHostBatch.RowSlice hostRows) {
    DecisionHostBatch hostBatch = hostRows.source();
    if (hostBatch.size() != hostBatch.capacity()) {
      throw new IllegalStateException(
          "device batch requires an exactly filled host batch: "
              + hostBatch.size()
              + "/"
              + hostBatch.capacity());
    }
  }

  private static DecisionDeviceBatch.TrainingTargets bindTargets(
      int rows, DecisionBucket bucket, NDArray categoricalSlab, NDArray numericSlab) {
    int actions = bucket.legalActionCapacity();
    int policyElements = Math.multiplyExact(rows, actions);
    NDArray behaviorPolicy = slice(numericSlab, 0, policyElements, new Shape(rows, actions));
    NDArray rolloutPolicy =
        slice(numericSlab, policyElements, policyElements * 2, new Shape(rows, actions));
    NDArray rowTargets =
        slice(
            numericSlab,
            policyElements * 2,
            new DecisionTrainingTargetLayout(rows, bucket).numericElementCount(),
            new Shape(rows, DecisionTrainingTargets.ROW_NUMERIC_STRIDE));
    int valueEnd = DecisionTrainingTargets.VALUE_TARGET_SIZE;
    int advantageEnd = valueEnd + DecisionTrainingTargets.ADVANTAGE_SIZE;
    return new DecisionDeviceBatch.TrainingTargets(
        categoricalSlab,
        numericSlab,
        categoricalSlab.reshape(rows),
        behaviorPolicy,
        rolloutPolicy,
        rowTargets.get("...,0"),
        rowTargets.get("...," + valueEnd),
        rowTargets.get("...," + advantageEnd),
        rowTargets.get("...," + (advantageEnd + 1)));
  }

  private static NDArray slice(NDArray slab, int start, int end, Shape shape) {
    if (start == end) {
      return slab.get(new NDIndex().addSliceDim(0, 0)).reshape(shape);
    }
    return slab.get(new NDIndex().addSliceDim(start, end)).reshape(shape);
  }

  /**
   * 一つの推論/学習実行枠が専有する再利用ホストバッファ。
   *
   * <p>同時利用を同期しない。ページ固定の複製は非同期なので、次の利用前に前回の順伝播を完了して前回の管理元を閉じる必要がある。 PyTorch
   * アクセラレーター以外では通常の管理元作成経路へ戻る。
   */
  public static final class Workspace implements AutoCloseable {
    private final PtNDManager owner;
    private final PinnedSlot pinnedInputCategories;
    private final PinnedSlot pinnedInputNumerics;
    private final PinnedSlot pinnedTargetCategories;
    private final PinnedSlot pinnedTargetNumerics;
    private final PinnedSlot pinnedStateCategories;
    private final PinnedSlot pinnedStateNumerics;
    private final PinnedSlot pinnedBoundaryContext;
    private final PinnedSlot pinnedValueTargets;
    private final PinnedSlot pinnedPlayerMemoryPresentIndices;
    private final PinnedSlot pinnedTransitionPresentIndices;
    private final PinnedInferenceInputSlabs pinnedInferenceInputs;
    private final PinnedInferenceIndexSlot pinnedInferenceIndices;
    private final DirectSlot directInputCategories;
    private final DirectSlot directInputNumerics;
    private final DirectSlot directTargetCategories;
    private final DirectSlot directTargetNumerics;
    private final DirectSlot directStateCategories;
    private final DirectSlot directStateNumerics;
    private final DirectSlot directBoundaryContext;
    private final DirectSlot directValueTargets;
    private final DirectSlot directPlayerMemoryPresentIndices;
    private final DirectSlot directTransitionPresentIndices;
    private final DirectInferenceInputSlabs directInferenceInputs;
    private final DirectInferenceIndexSlot directInferenceIndices;

    /**
     * 指定実行枠が専有する転送バッファ群を作る。
     *
     * @param owner バッファの寿命と対象デバイスを所有する管理元
     */
    public Workspace(NDManager owner) {
      this.owner =
          owner instanceof PtNDManager ptOwner && owner.getDevice().isGpu() ? ptOwner : null;
      pinnedInputCategories = new PinnedSlot(this.owner, SlabKind.INPUT_CATEGORIES);
      pinnedInputNumerics = new PinnedSlot(this.owner, SlabKind.INPUT_NUMERICS);
      pinnedTargetCategories = new PinnedSlot(this.owner, SlabKind.TARGET_CATEGORIES);
      pinnedTargetNumerics = new PinnedSlot(this.owner, SlabKind.TARGET_NUMERICS);
      pinnedStateCategories = new PinnedSlot(this.owner, SlabKind.STATE_CATEGORIES);
      pinnedStateNumerics = new PinnedSlot(this.owner, SlabKind.STATE_NUMERICS);
      pinnedBoundaryContext = new PinnedSlot(this.owner, SlabKind.BOUNDARY_CONTEXT);
      pinnedValueTargets = new PinnedSlot(this.owner, SlabKind.VALUE_TARGETS);
      pinnedPlayerMemoryPresentIndices =
          new PinnedSlot(this.owner, SlabKind.PLAYER_MEMORY_PRESENT_INDICES);
      pinnedTransitionPresentIndices =
          new PinnedSlot(this.owner, SlabKind.TRANSITION_PRESENT_INDICES);
      pinnedInferenceInputs = new PinnedInferenceInputSlabs(this.owner);
      pinnedInferenceIndices = new PinnedInferenceIndexSlot(this.owner);
      directInputCategories = new DirectSlot(this.owner, SlabKind.INPUT_CATEGORIES);
      directInputNumerics = new DirectSlot(this.owner, SlabKind.INPUT_NUMERICS);
      directTargetCategories = new DirectSlot(this.owner, SlabKind.TARGET_CATEGORIES);
      directTargetNumerics = new DirectSlot(this.owner, SlabKind.TARGET_NUMERICS);
      directStateCategories = new DirectSlot(this.owner, SlabKind.STATE_CATEGORIES);
      directStateNumerics = new DirectSlot(this.owner, SlabKind.STATE_NUMERICS);
      directBoundaryContext = new DirectSlot(this.owner, SlabKind.BOUNDARY_CONTEXT);
      directValueTargets = new DirectSlot(this.owner, SlabKind.VALUE_TARGETS);
      directPlayerMemoryPresentIndices =
          new DirectSlot(this.owner, SlabKind.PLAYER_MEMORY_PRESENT_INDICES);
      directTransitionPresentIndices =
          new DirectSlot(this.owner, SlabKind.TRANSITION_PRESENT_INDICES);
      directInferenceInputs = new DirectInferenceInputSlabs(this.owner);
      directInferenceIndices = new DirectInferenceIndexSlot(this.owner);
    }

    private Slabs transferTraining(
        NDManager manager,
        DecisionHostBatch.RowSlice hostRows,
        DecisionTensorTransfer transferMode) {
      if (owner == null || !(manager instanceof PtNDManager destination)) {
        return Slabs.regular(manager, hostRows);
      }
      return switch (transferMode) {
        case PINNED_BUFFER -> transferPinned(destination, hostRows);
        case DIRECT_BUFFER -> transferDirect(destination, hostRows);
      };
    }

    private InferenceSlabs transferInference(
        NDManager manager,
        DecisionHostBatch.RowSlice hostRows,
        DecisionTensorTransfer transferMode) {
      if (owner == null || !(manager instanceof PtNDManager destination)) {
        return InferenceSlabs.regular(manager, hostRows);
      }
      DecisionHostBatch.InferenceIndexLayout indexLayout = hostRows.inferenceIndexLayout(true);
      DecisionInferenceInputLayout inputLayout = hostRows.inferenceInputLayout(indexLayout);
      return switch (transferMode) {
        case PINNED_BUFFER -> {
          InferenceInputSlabs input =
              pinnedInferenceInputs.copy(destination, hostRows, inputLayout);
          InferenceIndexSlab indices =
              pinnedInferenceIndices.copy(destination, hostRows, indexLayout);
          yield InferenceSlabs.bind(input, indices, inputLayout);
        }
        case DIRECT_BUFFER -> {
          InferenceInputSlabs input =
              directInferenceInputs.copy(destination, hostRows, inputLayout);
          InferenceIndexSlab indices =
              directInferenceIndices.copy(destination, hostRows, indexLayout);
          yield InferenceSlabs.bind(input, indices, inputLayout);
        }
      };
    }

    /**
     * 複数の行ビューを実行枠専用ページ固定の入力へ直接抽出する。
     *
     * @param rows 同じ容量区分に属する非所有行ビュー群
     * @return 後続のH2D キューへの追加に渡す軽量記述情報
     */
    public StagedInference stageInference(DecisionHostBatch.RowBatch rows) {
      if (owner == null) {
        throw new IllegalStateException("pinned inference staging requires a GPU workspace");
      }
      DecisionHostBatch.InferenceIndexLayout indexLayout;
      DecisionInferenceInputLayout inputLayout;
      try (var event =
          InputBatchProfile.beginStage(
              "major", rows.bucket(), rows.size(), "inference.layout_count")) {
        indexLayout = rows.inferenceIndexLayout(true);
        inputLayout = rows.inferenceInputLayout(indexLayout);
        if (event != null) {
          event.success = true;
        }
      }
      pinnedInferenceInputs.stage(rows, inputLayout);
      pinnedInferenceIndices.stage(rows, indexLayout);
      return new StagedInference(this, rows, indexLayout, inputLayout);
    }

    /**
     * 現在のPyTorch ストリームへ抽出済み入力のH2D転送をキューへ追加する。
     *
     * @param destination バッチ-局所的なデバイス管理元
     * @param staged この作業領域が直前に作った記述情報
     * @param inputNumericType モデルへ渡す数値型
     * @return 論理ビューを結び付ける済みのデバイス側バッチ
     */
    public DecisionInferenceDeviceBatch enqueueStagedInference(
        PtNDManager destination, StagedInference staged, DataType inputNumericType) {
      if (staged.workspace != this) {
        throw new IllegalArgumentException("staged inference belongs to another workspace");
      }
      InferenceInputSlabs input = pinnedInferenceInputs.enqueue(destination, staged.inputLayout);
      InferenceIndexSlab indices = pinnedInferenceIndices.enqueue(destination, staged.indexLayout);
      InferenceSlabs slabs = InferenceSlabs.bind(input, indices, staged.inputLayout);
      NDArray inputNumerics = convertInputNumerics(slabs.inputNumerics(), inputNumericType);
      DecisionInferenceInputs inputs =
          DecisionInferenceInputs.bind(slabs.layout(), slabs.inputCategories(), inputNumerics);
      return new DecisionInferenceDeviceBatch(
          slabs.inputCategories(),
          inputNumerics,
          inputs,
          slabs.playerMemoryPresentIndices(),
          slabs.transitionPresentIndices(),
          slabs.policyExecutionIndices());
    }

    /** 価値のみ行を実行枠専用ページ固定の状態バッファへ直接抽出する。 */
    public StagedState stageState(DecisionHostBatch.RowBatch rows) {
      if (owner == null) {
        throw new IllegalStateException("pinned state staging requires a GPU workspace");
      }
      int categoryElements;
      int numericElements;
      int boundaryElements;
      int playerMemoryPresentCount;
      try (var event =
          InputBatchProfile.beginStage("major", rows.bucket(), rows.size(), "state.layout_count")) {
        categoryElements = Math.multiplyExact(rows.size(), DecisionInputSchema.STATE_INT_COUNT);
        numericElements = Math.multiplyExact(rows.size(), DecisionInputSchema.STATE_FLOAT_COUNT);
        boundaryElements = Math.multiplyExact(rows.size(), DecisionBoundaryContext.INPUT_SIZE);
        playerMemoryPresentCount = rows.playerMemoryPresentCount();
        if (event != null) {
          event.success = true;
        }
      }
      try (var event =
          InputBatchProfile.beginStage(
              "major", rows.bucket(), rows.size(), "state.buffer_capacity")) {
        pinnedStateCategories.ensureCapacity(categoryElements);
        pinnedStateNumerics.ensureCapacity(numericElements);
        pinnedBoundaryContext.ensureCapacity(boundaryElements);
        pinnedPlayerMemoryPresentIndices.ensureCapacity(playerMemoryPresentCount);
        if (event != null) {
          event.success = true;
        }
      }
      try (var event =
          InputBatchProfile.beginStage("major", rows.bucket(), rows.size(), "state.input_copy")) {
        rows.copyStateCategoriesTo(
            pinnedStateCategories
                .buffer
                .getByteBuffer()
                .order(ByteOrder.nativeOrder())
                .asShortBuffer());
        rows.copyStateNumericsTo(
            pinnedStateNumerics
                .buffer
                .getByteBuffer()
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer());
        rows.copyBoundaryContextTo(
            pinnedBoundaryContext
                .buffer
                .getByteBuffer()
                .order(ByteOrder.nativeOrder())
                .asFloatBuffer());
        if (event != null) {
          event.categoricalBytes = (long) categoryElements * Short.BYTES;
          event.numericBytes = ((long) numericElements + boundaryElements) * Float.BYTES;
          event.success = true;
        }
      }
      try (var event =
          InputBatchProfile.beginStage("major", rows.bucket(), rows.size(), "state.index_write")) {
        rows.copyPlayerMemoryPresentIndicesTo(
            pinnedPlayerMemoryPresentIndices
                .buffer
                .getByteBuffer()
                .order(ByteOrder.nativeOrder())
                .asIntBuffer());
        if (event != null) {
          event.indexBytes = (long) playerMemoryPresentCount * Integer.BYTES;
          event.success = true;
        }
      }
      return new StagedState(this, rows, playerMemoryPresentCount);
    }

    /** 現在のPyTorch ストリームへ抽出済み状態入力のH2D転送をキューへ追加する。 */
    public DecisionStateInferenceBatch enqueueStagedState(
        PtNDManager destination, StagedState staged, DataType inputNumericType) {
      if (staged.workspace != this) {
        throw new IllegalArgumentException("staged state belongs to another workspace");
      }
      int rows = staged.rows.size();
      PtNDArray categories =
          destination.create(
              new Shape(Math.multiplyExact(rows, DecisionInputSchema.STATE_INT_COUNT)),
              DataType.INT16);
      PtNDArray numerics =
          destination.create(
              new Shape(Math.multiplyExact(rows, DecisionInputSchema.STATE_FLOAT_COUNT)),
              DataType.FLOAT32);
      PtNDArray boundary =
          destination.create(
              new Shape(Math.multiplyExact(rows, DecisionBoundaryContext.INPUT_SIZE)),
              DataType.FLOAT32);
      PtNDArray playerMemoryPresentIndices =
          destination.create(new Shape(staged.playerMemoryPresentCount), DataType.INT32);
      categories.enqueueCopyFrom(pinnedStateCategories.buffer);
      numerics.enqueueCopyFrom(pinnedStateNumerics.buffer);
      boundary.enqueueCopyFrom(pinnedBoundaryContext.buffer);
      playerMemoryPresentIndices.enqueueCopyFrom(pinnedPlayerMemoryPresentIndices.buffer);
      return new DecisionStateInferenceBatch(
          new DecisionStateInputs(
              categories.reshape(rows, DecisionInputSchema.STATE_INT_COUNT),
              convertInputNumerics(numerics, inputNumericType)
                  .reshape(rows, DecisionInputSchema.STATE_FLOAT_COUNT),
              convertInputNumerics(boundary, inputNumericType)
                  .reshape(rows, DecisionBoundaryContext.INPUT_SIZE)),
          playerMemoryPresentIndices);
    }

    private ValueSlabs transferValue(
        NDManager manager,
        DecisionHostBatch.RowSlice hostRows,
        DecisionTensorTransfer transferMode) {
      if (owner == null || !(manager instanceof PtNDManager destination)) {
        return ValueSlabs.regular(manager, hostRows);
      }
      return switch (transferMode) {
        case PINNED_BUFFER ->
            new ValueSlabs(
                pinnedStateCategories.copy(destination, hostRows),
                pinnedStateNumerics.copy(destination, hostRows),
                pinnedBoundaryContext.copy(destination, hostRows),
                pinnedPlayerMemoryPresentIndices.copy(destination, hostRows),
                pinnedValueTargets.copy(destination, hostRows));
        case DIRECT_BUFFER ->
            new ValueSlabs(
                directStateCategories.copy(destination, hostRows),
                directStateNumerics.copy(destination, hostRows),
                directBoundaryContext.copy(destination, hostRows),
                directPlayerMemoryPresentIndices.copy(destination, hostRows),
                directValueTargets.copy(destination, hostRows));
      };
    }

    private StateSlabs transferState(
        NDManager manager,
        DecisionHostBatch.RowSlice hostRows,
        DecisionTensorTransfer transferMode) {
      if (owner == null || !(manager instanceof PtNDManager destination)) {
        return StateSlabs.regular(manager, hostRows);
      }
      return switch (transferMode) {
        case PINNED_BUFFER ->
            new StateSlabs(
                pinnedStateCategories.copy(destination, hostRows),
                pinnedStateNumerics.copy(destination, hostRows),
                pinnedBoundaryContext.copy(destination, hostRows));
        case DIRECT_BUFFER ->
            new StateSlabs(
                directStateCategories.copy(destination, hostRows),
                directStateNumerics.copy(destination, hostRows),
                directBoundaryContext.copy(destination, hostRows));
      };
    }

    private Slabs transferPinned(PtNDManager destination, DecisionHostBatch.RowSlice hostRows) {
      return new Slabs(
          pinnedInputCategories.copy(destination, hostRows),
          pinnedInputNumerics.copy(destination, hostRows),
          pinnedPlayerMemoryPresentIndices.copy(destination, hostRows),
          pinnedTransitionPresentIndices.copy(destination, hostRows),
          hostRows.hasTrainingTargets() ? pinnedTargetCategories.copy(destination, hostRows) : null,
          hostRows.hasTrainingTargets() ? pinnedTargetNumerics.copy(destination, hostRows) : null);
    }

    private GraphInput prepareGraphInput(
        PtNDManager destination, DecisionHostBatch.RowSlice hostRows, DataType inputNumericType) {
      PtNDArray categories = pinnedInputCategories.allocate(destination, hostRows);
      PtNDArray numerics = pinnedInputNumerics.allocate(destination, hostRows);
      pinnedInputCategories.copyInto(categories, hostRows);
      pinnedInputNumerics.copyInto(numerics, hostRows);
      return new GraphInput(
          categories,
          numerics,
          new DecisionInputLayout(hostRows.size(), hostRows.bucket()),
          inputNumericType,
          pinnedInputCategories,
          pinnedInputNumerics);
    }

    private Slabs transferDirect(PtNDManager destination, DecisionHostBatch.RowSlice hostRows) {
      return new Slabs(
          directInputCategories.copy(destination, hostRows),
          directInputNumerics.copy(destination, hostRows),
          directPlayerMemoryPresentIndices.copy(destination, hostRows),
          directTransitionPresentIndices.copy(destination, hostRows),
          hostRows.hasTrainingTargets() ? directTargetCategories.copy(destination, hostRows) : null,
          hostRows.hasTrainingTargets() ? directTargetNumerics.copy(destination, hostRows) : null);
    }

    @Override
    public void close() {
      pinnedInputCategories.close();
      pinnedInputNumerics.close();
      pinnedTargetCategories.close();
      pinnedTargetNumerics.close();
      pinnedStateCategories.close();
      pinnedStateNumerics.close();
      pinnedBoundaryContext.close();
      pinnedValueTargets.close();
      pinnedPlayerMemoryPresentIndices.close();
      pinnedTransitionPresentIndices.close();
      pinnedInferenceInputs.close();
      pinnedInferenceIndices.close();
      directInferenceInputs.close();
      directInferenceIndices.close();
    }
  }

  /** 固定アドレスのデバイス入力と、その内容を更新する専用ページ固定のバッファ。 */
  public static final class GraphInput {
    private final PtNDArray rawCategories;
    private final PtNDArray rawNumerics;
    private final DecisionInputLayout layout;
    private final DataType inputNumericType;
    private final PinnedSlot categorySlot;
    private final PinnedSlot numericSlot;

    private GraphInput(
        PtNDArray rawCategories,
        PtNDArray rawNumerics,
        DecisionInputLayout layout,
        DataType inputNumericType,
        PinnedSlot categorySlot,
        PinnedSlot numericSlot) {
      this.rawCategories = rawCategories;
      this.rawNumerics = rawNumerics;
      this.layout = layout;
      this.inputNumericType = inputNumericType;
      this.categorySlot = categorySlot;
      this.numericSlot = numericSlot;
    }

    /**
     * 次の再生で読むホスト行を、既存デバイス側記憶領域へ非同期転送する。
     *
     * @param hostRows 記録時と同じ行数・容量区分を持つ確定済み部分領域
     */
    public void refresh(DecisionHostBatch.RowSlice hostRows) {
      if (hostRows.size() != layout.capacity() || !hostRows.bucket().equals(layout.bucket())) {
        throw new IllegalArgumentException("graph input shape changed");
      }
      categorySlot.copyInto(rawCategories, hostRows);
      numericSlot.copyInto(rawNumerics, hostRows);
    }

    /**
     * 記録中に一度だけ実行し、モデルが読む標準形式のビューを構築する。
     *
     * @return 固定記憶領域を参照する推論専用デバイス側バッチ
     */
    public DecisionDeviceBatch bind() {
      NDArray categories = rawCategories;
      NDArray numerics =
          inputNumericType == DataType.FLOAT32
              ? rawNumerics
              : rawNumerics.toType(inputNumericType, false);
      return new DecisionDeviceBatch(
          categories,
          numerics,
          DecisionNetworkInputs.bind(layout, categories, numerics),
          rawCategories.getManager().zeros(new Shape(0), DataType.INT32),
          rawCategories.getManager().zeros(new Shape(0), DataType.INT32),
          null);
    }
  }

  private record ValueSlabs(
      NDArray stateCategories,
      NDArray stateNumerics,
      NDArray boundaryContext,
      NDArray playerMemoryPresentIndices,
      NDArray valueTargets) {

    private static ValueSlabs regular(NDManager manager, DecisionHostBatch.RowSlice hostRows) {
      int rows = hostRows.size();
      short[] categories = new short[Math.multiplyExact(rows, DecisionInputSchema.STATE_INT_COUNT)];
      float[] numerics = new float[Math.multiplyExact(rows, DecisionInputSchema.STATE_FLOAT_COUNT)];
      float[] boundary = new float[Math.multiplyExact(rows, DecisionBoundaryContext.INPUT_SIZE)];
      float[] targets = new float[Math.multiplyExact(rows, VALUE_TARGET_STRIDE)];
      hostRows.copyStateCategoriesTo(ShortBuffer.wrap(categories));
      hostRows.copyStateNumericsTo(FloatBuffer.wrap(numerics));
      hostRows.copyBoundaryContextTo(FloatBuffer.wrap(boundary));
      hostRows.copyValueTargetsTo(FloatBuffer.wrap(targets));
      return new ValueSlabs(
          transferCategories(manager, categories),
          manager.create(numerics),
          manager.create(boundary),
          transferIndices(manager, hostRows, SlabKind.PLAYER_MEMORY_PRESENT_INDICES),
          manager.create(targets));
    }
  }

  private record StateSlabs(
      NDArray stateCategories, NDArray stateNumerics, NDArray boundaryContext) {

    private static StateSlabs regular(NDManager manager, DecisionHostBatch.RowSlice hostRows) {
      int rows = hostRows.size();
      short[] categories = new short[Math.multiplyExact(rows, DecisionInputSchema.STATE_INT_COUNT)];
      float[] numerics = new float[Math.multiplyExact(rows, DecisionInputSchema.STATE_FLOAT_COUNT)];
      float[] boundary = new float[Math.multiplyExact(rows, DecisionBoundaryContext.INPUT_SIZE)];
      hostRows.copyStateCategoriesTo(ShortBuffer.wrap(categories));
      hostRows.copyStateNumericsTo(FloatBuffer.wrap(numerics));
      hostRows.copyBoundaryContextTo(FloatBuffer.wrap(boundary));
      return new StateSlabs(
          transferCategories(manager, categories),
          manager.create(numerics),
          manager.create(boundary));
    }
  }

  private record Slabs(
      NDArray inputCategories,
      NDArray inputNumerics,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      NDArray targetCategories,
      NDArray targetNumerics) {

    private static Slabs regular(NDManager manager, DecisionHostBatch.RowSlice hostRows) {
      DecisionHostBatch hostBatch = hostRows.materializeDense();
      return new Slabs(
          transferCategories(manager, hostBatch.inputs().denseCategories()),
          manager.create(hostBatch.inputs().denseNumerics()),
          transferIndices(manager, hostRows, SlabKind.PLAYER_MEMORY_PRESENT_INDICES),
          transferIndices(manager, hostRows, SlabKind.TRANSITION_PRESENT_INDICES),
          hostBatch.hasTrainingTargets()
              ? manager.create(hostBatch.trainingTargets().categoricalSlab())
              : null,
          hostBatch.hasTrainingTargets()
              ? manager.create(hostBatch.trainingTargets().numericSlab())
              : null);
    }
  }

  record InferenceInputSlabs(NDArray categories, NDArray numerics) {}

  private record InferenceSlabs(
      NDArray inputCategories,
      NDArray inputNumerics,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices,
      DecisionInferenceInputLayout layout) {

    private static InferenceSlabs regular(NDManager manager, DecisionHostBatch.RowSlice hostRows) {
      DecisionHostBatch.InferenceIndexLayout indexLayout =
          hostRows.inferenceIndexLayout(manager.getDevice().isGpu());
      DecisionInferenceInputLayout inputLayout = hostRows.inferenceInputLayout(indexLayout);
      short[] categories = new short[inputLayout.categoricalElementCount()];
      float[] numerics = new float[inputLayout.numericElementCount()];
      hostRows.copyInferenceInputsTo(
          ShortBuffer.wrap(categories), FloatBuffer.wrap(numerics), inputLayout);
      InferenceInputSlabs input =
          new InferenceInputSlabs(
              transferCategories(manager, categories), manager.create(numerics));
      InferenceIndexSlab indices = InferenceIndexSlab.regular(manager, hostRows, indexLayout);
      return bind(input, indices, inputLayout);
    }

    private static InferenceSlabs bind(
        InferenceInputSlabs input,
        InferenceIndexSlab indices,
        DecisionInferenceInputLayout inputLayout) {
      return new InferenceSlabs(
          input.categories(),
          input.numerics(),
          indices.playerMemoryPresentIndices(),
          indices.transitionPresentIndices(),
          indices.policyExecutionIndices(),
          inputLayout);
    }
  }

  record InferenceIndexSlab(
      NDArray packedIndices,
      NDArray playerMemoryPresentIndices,
      NDArray transitionPresentIndices,
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices) {

    private static InferenceIndexSlab regular(
        NDManager manager,
        DecisionHostBatch.RowSlice hostRows,
        DecisionHostBatch.InferenceIndexLayout layout) {
      int[] values = new int[layout.totalCount()];
      hostRows.copyInferenceIndicesTo(IntBuffer.wrap(values), layout);
      return bind(manager.create(values), layout);
    }

    private static InferenceIndexSlab bind(
        NDArray packedIndices, DecisionHostBatch.InferenceIndexLayout layout) {
      NDArray playerMemoryPresentIndices =
          slice(
              packedIndices,
              layout.playerMemoryOffset(),
              Math.addExact(layout.playerMemoryOffset(), layout.playerMemoryCount()),
              new Shape(layout.playerMemoryCount()));
      NDArray transitionPresentIndices =
          slice(
              packedIndices,
              layout.transitionOffset(),
              Math.addExact(layout.transitionOffset(), layout.transitionCount()),
              new Shape(layout.transitionCount()));
      DecisionInferenceDeviceBatch.PolicyExecutionIndices policyExecutionIndices =
          layout.includesPolicyExecution()
              ? layout
                  .policyExecutionLayout()
                  .bind(
                      slice(
                          packedIndices,
                          layout.policyExecutionOffset(),
                          Math.addExact(
                              layout.policyExecutionOffset(), layout.policyExecutionCount()),
                          new Shape(layout.policyExecutionCount())))
              : null;
      return new InferenceIndexSlab(
          packedIndices,
          playerMemoryPresentIndices,
          transitionPresentIndices,
          policyExecutionIndices);
    }
  }

  private static final class PinnedSlot implements AutoCloseable {
    private final PtNDManager owner;
    private final SlabKind kind;
    private PtPinnedBuffer buffer;
    private PtCopyEvent graphCopyEvent;

    private PinnedSlot(PtNDManager owner, SlabKind kind) {
      this.owner = owner;
      this.kind = kind;
    }

    private NDArray copy(PtNDManager destination, DecisionHostBatch.RowSlice hostRows) {
      int elements = kind.elements(hostRows);
      ensureCapacity(elements);
      kind.copy(hostRows, buffer.getByteBuffer());
      PtNDArray array = destination.create(new Shape(elements), kind.dataType());
      array.copyFromPinnedBufferAsync(buffer);
      return array;
    }

    private PtNDArray allocate(PtNDManager destination, DecisionHostBatch.RowSlice hostRows) {
      int elements = kind.elements(hostRows);
      ensureCapacity(elements);
      return destination.create(new Shape(elements), kind.dataType());
    }

    private void copyInto(PtNDArray destination, DecisionHostBatch.RowSlice hostRows) {
      if (graphCopyEvent != null) {
        graphCopyEvent.close();
      }
      kind.copy(hostRows, buffer.getByteBuffer());
      graphCopyEvent = destination.copyFromPinnedBufferAsync(buffer);
    }

    private void ensureCapacity(int elements) {
      if (buffer != null && buffer.size() >= elements) {
        return;
      }
      close();
      buffer = owner.allocatePinned(elements, kind.dataType());
    }

    @Override
    public void close() {
      if (graphCopyEvent != null) {
        graphCopyEvent.close();
        graphCopyEvent = null;
      }
      if (buffer != null) {
        buffer.close();
        buffer = null;
      }
    }
  }

  static final class PinnedInferenceInputSlabs implements AutoCloseable {
    private final PtNDManager owner;
    private PtPinnedBuffer categoricalBuffer;
    private PtPinnedBuffer numericBuffer;

    PinnedInferenceInputSlabs(PtNDManager owner) {
      this.owner = owner;
    }

    InferenceInputSlabs copy(
        PtNDManager destination,
        DecisionHostBatch.RowSlice hostRows,
        DecisionInferenceInputLayout layout) {
      ensureCapacity(layout);
      hostRows.copyInferenceInputsTo(
          categoricalBuffer.getByteBuffer().order(ByteOrder.nativeOrder()).asShortBuffer(),
          numericBuffer.getByteBuffer().order(ByteOrder.nativeOrder()).asFloatBuffer(),
          layout);
      PtNDArray categories =
          destination.create(new Shape(layout.categoricalElementCount()), DataType.INT16);
      PtNDArray numerics =
          destination.create(new Shape(layout.numericElementCount()), DataType.FLOAT32);
      categories.copyFromPinnedBufferAsync(categoricalBuffer);
      numerics.copyFromPinnedBufferAsync(numericBuffer);
      return new InferenceInputSlabs(categories, numerics);
    }

    private void stage(DecisionHostBatch.RowBatch hostRows, DecisionInferenceInputLayout layout) {
      try (var event =
          InputBatchProfile.beginStage(
              "major", hostRows.bucket(), hostRows.size(), "inference.buffer_capacity")) {
        ensureCapacity(layout);
        if (event != null) {
          event.success = true;
        }
      }
      try (var event =
          InputBatchProfile.beginStage(
              "major", hostRows.bucket(), hostRows.size(), "inference.input_copy")) {
        hostRows.copyInferenceInputsTo(
            categoricalBuffer.getByteBuffer().order(ByteOrder.nativeOrder()).asShortBuffer(),
            numericBuffer.getByteBuffer().order(ByteOrder.nativeOrder()).asFloatBuffer(),
            layout);
        if (event != null) {
          event.categoricalBytes = (long) layout.categoricalElementCount() * Short.BYTES;
          event.numericBytes = (long) layout.numericElementCount() * Float.BYTES;
          event.success = true;
        }
      }
    }

    private InferenceInputSlabs enqueue(
        PtNDManager destination, DecisionInferenceInputLayout layout) {
      PtNDArray categories =
          destination.create(new Shape(layout.categoricalElementCount()), DataType.INT16);
      PtNDArray numerics =
          destination.create(new Shape(layout.numericElementCount()), DataType.FLOAT32);
      categories.enqueueCopyFrom(categoricalBuffer);
      numerics.enqueueCopyFrom(numericBuffer);
      return new InferenceInputSlabs(categories, numerics);
    }

    private void ensureCapacity(DecisionInferenceInputLayout layout) {
      if (categoricalBuffer == null
          || categoricalBuffer.size() < layout.categoricalElementCount()) {
        if (categoricalBuffer != null) {
          categoricalBuffer.close();
        }
        categoricalBuffer = owner.allocatePinned(layout.categoricalElementCount(), DataType.INT16);
      }
      if (numericBuffer == null || numericBuffer.size() < layout.numericElementCount()) {
        if (numericBuffer != null) {
          numericBuffer.close();
        }
        numericBuffer = owner.allocatePinned(layout.numericElementCount(), DataType.FLOAT32);
      }
    }

    @Override
    public void close() {
      if (categoricalBuffer != null) {
        categoricalBuffer.close();
        categoricalBuffer = null;
      }
      if (numericBuffer != null) {
        numericBuffer.close();
        numericBuffer = null;
      }
    }
  }

  static final class PinnedInferenceIndexSlot implements AutoCloseable {
    private final PtNDManager owner;
    private PtPinnedBuffer buffer;

    PinnedInferenceIndexSlot(PtNDManager owner) {
      this.owner = owner;
    }

    InferenceIndexSlab copy(
        PtNDManager destination,
        DecisionHostBatch.RowSlice hostRows,
        DecisionHostBatch.InferenceIndexLayout layout) {
      int elements = layout.totalCount();
      if (elements == 0) {
        return InferenceIndexSlab.bind(destination.zeros(new Shape(0), DataType.INT32), layout);
      }
      ensureCapacity(elements);
      hostRows.copyInferenceIndicesTo(
          buffer.getByteBuffer().order(ByteOrder.nativeOrder()).asIntBuffer(), layout);
      PtNDArray packedIndices = destination.create(new Shape(elements), DataType.INT32);
      packedIndices.copyFromPinnedBufferAsync(buffer);
      return InferenceIndexSlab.bind(packedIndices, layout);
    }

    private void stage(
        DecisionHostBatch.RowBatch hostRows, DecisionHostBatch.InferenceIndexLayout layout) {
      int elements = layout.totalCount();
      if (elements == 0) {
        return;
      }
      try (var event =
          InputBatchProfile.beginStage(
              "major", hostRows.bucket(), hostRows.size(), "inference.buffer_capacity")) {
        ensureCapacity(elements);
        if (event != null) {
          event.success = true;
        }
      }
      try (var event =
          InputBatchProfile.beginStage(
              "major", hostRows.bucket(), hostRows.size(), "inference.index_write")) {
        hostRows.copyInferenceIndicesTo(
            buffer.getByteBuffer().order(ByteOrder.nativeOrder()).asIntBuffer(), layout);
        if (event != null) {
          event.indexBytes = (long) elements * Integer.BYTES;
          event.success = true;
        }
      }
    }

    private InferenceIndexSlab enqueue(
        PtNDManager destination, DecisionHostBatch.InferenceIndexLayout layout) {
      int elements = layout.totalCount();
      if (elements == 0) {
        return InferenceIndexSlab.bind(destination.zeros(new Shape(0), DataType.INT32), layout);
      }
      PtNDArray packedIndices = destination.create(new Shape(elements), DataType.INT32);
      packedIndices.enqueueCopyFrom(buffer);
      return InferenceIndexSlab.bind(packedIndices, layout);
    }

    private void ensureCapacity(int elements) {
      if (buffer != null && buffer.size() >= elements) {
        return;
      }
      close();
      buffer = owner.allocatePinned(elements, DataType.INT32);
    }

    @Override
    public void close() {
      if (buffer != null) {
        buffer.close();
        buffer = null;
      }
    }
  }

  private static final class DirectSlot {
    private final PtNDManager owner;
    private final SlabKind kind;
    private ByteBuffer buffer;

    private DirectSlot(PtNDManager owner, SlabKind kind) {
      this.owner = owner;
      this.kind = kind;
    }

    private NDArray copy(PtNDManager destination, DecisionHostBatch.RowSlice hostRows) {
      int elements = kind.elements(hostRows);
      ByteBuffer source = prepare(elements);
      kind.copy(hostRows, source);
      PtNDArray array = destination.create(new Shape(elements), kind.dataType());
      array.copyFromDirectBuffer(source);
      return array;
    }

    private ByteBuffer prepare(int elements) {
      int bytes = Math.multiplyExact(elements, kind.dataType().getNumOfBytes());
      if (buffer == null || buffer.capacity() < bytes) {
        buffer = owner.allocateDirect(bytes);
      }
      ByteBuffer source = buffer.duplicate().order(ByteOrder.nativeOrder());
      source.clear();
      source.limit(bytes);
      return source;
    }
  }

  static final class DirectInferenceInputSlabs implements AutoCloseable {
    private final PtNDManager owner;
    private ByteBuffer categoricalBuffer;
    private ByteBuffer numericBuffer;

    DirectInferenceInputSlabs(PtNDManager owner) {
      this.owner = owner;
    }

    InferenceInputSlabs copy(
        PtNDManager destination,
        DecisionHostBatch.RowSlice hostRows,
        DecisionInferenceInputLayout layout) {
      ByteBuffer categorySource = prepareCategorical(layout.categoricalElementCount());
      ByteBuffer numericSource = prepareNumeric(layout.numericElementCount());
      hostRows.copyInferenceInputsTo(
          categorySource.order(ByteOrder.nativeOrder()).asShortBuffer(),
          numericSource.order(ByteOrder.nativeOrder()).asFloatBuffer(),
          layout);
      PtNDArray categories =
          destination.create(new Shape(layout.categoricalElementCount()), DataType.INT16);
      PtNDArray numerics =
          destination.create(new Shape(layout.numericElementCount()), DataType.FLOAT32);
      categories.copyFromDirectBuffer(categorySource);
      numerics.copyFromDirectBuffer(numericSource);
      return new InferenceInputSlabs(categories, numerics);
    }

    private ByteBuffer prepareCategorical(int elements) {
      int bytes = Math.multiplyExact(elements, Short.BYTES);
      if (categoricalBuffer == null || categoricalBuffer.capacity() < bytes) {
        categoricalBuffer = owner.allocateDirect(bytes);
      }
      ByteBuffer source = categoricalBuffer.duplicate().order(ByteOrder.nativeOrder());
      source.clear();
      source.limit(bytes);
      return source;
    }

    private ByteBuffer prepareNumeric(int elements) {
      int bytes = Math.multiplyExact(elements, Float.BYTES);
      if (numericBuffer == null || numericBuffer.capacity() < bytes) {
        numericBuffer = owner.allocateDirect(bytes);
      }
      ByteBuffer source = numericBuffer.duplicate().order(ByteOrder.nativeOrder());
      source.clear();
      source.limit(bytes);
      return source;
    }

    @Override
    public void close() {
      categoricalBuffer = null;
      numericBuffer = null;
    }
  }

  static final class DirectInferenceIndexSlot implements AutoCloseable {
    private final PtNDManager owner;
    private ByteBuffer buffer;

    DirectInferenceIndexSlot(PtNDManager owner) {
      this.owner = owner;
    }

    InferenceIndexSlab copy(
        PtNDManager destination,
        DecisionHostBatch.RowSlice hostRows,
        DecisionHostBatch.InferenceIndexLayout layout) {
      int elements = layout.totalCount();
      if (elements == 0) {
        return InferenceIndexSlab.bind(destination.zeros(new Shape(0), DataType.INT32), layout);
      }
      ByteBuffer source = prepare(elements);
      hostRows.copyInferenceIndicesTo(source.order(ByteOrder.nativeOrder()).asIntBuffer(), layout);
      PtNDArray packedIndices = destination.create(new Shape(elements), DataType.INT32);
      packedIndices.copyFromDirectBuffer(source);
      return InferenceIndexSlab.bind(packedIndices, layout);
    }

    private ByteBuffer prepare(int elements) {
      int bytes = Math.multiplyExact(elements, Integer.BYTES);
      if (buffer == null || buffer.capacity() < bytes) {
        buffer = owner.allocateDirect(bytes);
      }
      ByteBuffer source = buffer.duplicate().order(ByteOrder.nativeOrder());
      source.clear();
      source.limit(bytes);
      return source;
    }

    @Override
    public void close() {
      buffer = null;
    }
  }

  private enum SlabKind {
    INPUT_CATEGORIES(DataType.INT16) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.inputCategoricalElementCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyInputCategoriesTo(destination.order(ByteOrder.nativeOrder()).asShortBuffer());
      }
    },
    INPUT_NUMERICS(DataType.FLOAT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.inputNumericElementCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyInputNumericsTo(destination.order(ByteOrder.nativeOrder()).asFloatBuffer());
      }
    },
    TARGET_CATEGORIES(DataType.INT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.targetCategoricalElementCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyTargetCategoriesTo(destination.order(ByteOrder.nativeOrder()).asIntBuffer());
      }
    },
    TARGET_NUMERICS(DataType.FLOAT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.targetNumericElementCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyTargetNumericsTo(destination.order(ByteOrder.nativeOrder()).asFloatBuffer());
      }
    },
    STATE_CATEGORIES(DataType.INT16) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return Math.multiplyExact(hostRows.size(), DecisionInputSchema.STATE_INT_COUNT);
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyStateCategoriesTo(destination.order(ByteOrder.nativeOrder()).asShortBuffer());
      }
    },
    STATE_NUMERICS(DataType.FLOAT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return Math.multiplyExact(hostRows.size(), DecisionInputSchema.STATE_FLOAT_COUNT);
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyStateNumericsTo(destination.order(ByteOrder.nativeOrder()).asFloatBuffer());
      }
    },
    BOUNDARY_CONTEXT(DataType.FLOAT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return Math.multiplyExact(hostRows.size(), DecisionBoundaryContext.INPUT_SIZE);
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyBoundaryContextTo(destination.order(ByteOrder.nativeOrder()).asFloatBuffer());
      }
    },
    PLAYER_MEMORY_PRESENT_INDICES(DataType.INT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.playerMemoryPresentCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyPlayerMemoryPresentIndicesTo(
            0, destination.order(ByteOrder.nativeOrder()).asIntBuffer());
      }
    },
    TRANSITION_PRESENT_INDICES(DataType.INT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return hostRows.transitionPresentCount();
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyTransitionPresentIndicesTo(
            0, destination.order(ByteOrder.nativeOrder()).asIntBuffer());
      }
    },
    VALUE_TARGETS(DataType.FLOAT32) {
      @Override
      int elements(DecisionHostBatch.RowSlice hostRows) {
        return Math.multiplyExact(hostRows.size(), VALUE_TARGET_STRIDE);
      }

      @Override
      void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination) {
        hostRows.copyValueTargetsTo(destination.order(ByteOrder.nativeOrder()).asFloatBuffer());
      }
    };

    private final DataType dataType;

    SlabKind(DataType dataType) {
      this.dataType = dataType;
    }

    DataType dataType() {
      return dataType;
    }

    abstract int elements(DecisionHostBatch.RowSlice hostRows);

    abstract void copy(DecisionHostBatch.RowSlice hostRows, ByteBuffer destination);
  }

  private static NDArray transferCategories(NDManager manager, short[] values) {
    return manager.create(ShortBuffer.wrap(values), new Shape(values.length), DataType.INT16);
  }

  private static NDArray transferIndices(
      NDManager manager, DecisionHostBatch.RowSlice hostRows, SlabKind kind) {
    int[] values = new int[kind.elements(hostRows)];
    switch (kind) {
      case PLAYER_MEMORY_PRESENT_INDICES ->
          hostRows.copyPlayerMemoryPresentIndicesTo(0, IntBuffer.wrap(values));
      case TRANSITION_PRESENT_INDICES ->
          hostRows.copyTransitionPresentIndicesTo(0, IntBuffer.wrap(values));
      default -> throw new IllegalArgumentException("not an execution-index slab: " + kind);
    }
    return manager.create(values);
  }
}
