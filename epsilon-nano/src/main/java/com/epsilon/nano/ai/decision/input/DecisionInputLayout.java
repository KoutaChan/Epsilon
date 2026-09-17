package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.types.Shape;
import com.epsilon.core.Tile;
import java.util.List;

/**
 * Decision バッチのテンソル形状、型、順序と、連続バッファ内の配置を定義する。
 *
 * <p>状態と行動の入力、および別領域に格納する遷移の入力について、各テンソルの要素数、行の間隔、開始位置を定める。ホストでの確保、行の複製、ページ固定メモリからの転送、デバイス上のビュー生成は、この配置を共通に使用する。
 *
 * <p>入力はカテゴリ値と数値の二つの連続バッファにまとめて転送する。カテゴリ値は INT16、数値は FLOAT32 を基本とし、デバイス上でもカテゴリ値を INT16
 * のまま保持する。埋め込みなどの演算で必要な箇所だけ整数型を拡張するため、フィールドごとの小さな転送は発生しない。
 */
public final class DecisionInputLayout {

  private static final List<Tensor> TENSORS = List.of(Tensor.values());

  /** ホスト側の連続バッファのプリミティブ型種別。 */
  public enum Slab {
    /** short へ可逆圧縮して転送するカテゴリ ID 群。 */
    CATEGORICAL,
    /** float のまま転送する正規化済み数値群。 */
    NUMERIC
  }

  /** ネットワークが受け取る論理的なテンソル。宣言順はNDList契約にも使用する。 */
  public enum Tensor {
    /** 局・各プレイヤー・牌種・河・面子のカテゴリ値状態。 */
    STATE_CATEGORIES(Slab.CATEGORICAL),
    /** 最初の合法行動ごとのカテゴリ値特徴量。 */
    ACTION_CATEGORIES(Slab.CATEGORICAL),
    /** 方策グラフが条件付き候補を直接指定位置の要素を集めるための参照先の対応メタデータ。 */
    ACTION_ROUTES(Slab.CATEGORICAL),
    /** 最初の行動適用後の各遷移に属するカテゴリ値特徴量。 */
    TRANSITION_CATEGORIES(Slab.CATEGORICAL),
    /** 各遷移の34牌種別行動後の状態特徴量。 */
    TRANSITION_TILES(Slab.CATEGORICAL),
    /** 各遷移の待ち枠に対応する牌種ID。 */
    WAIT_TILE_IDS(Slab.CATEGORICAL),
    /** 各待ち牌で成立可能なRON・TSUMO別の役ビットチャンク。 */
    WAIT_YAKUS(Slab.CATEGORICAL),
    /** 局・各プレイヤー・牌種・河・面子の数値状態。 */
    STATE_NUMERICS(Slab.NUMERIC),
    /** 価値専用のGRP 4x4 周辺分布と配牌前局境界特徴量。 */
    BOUNDARY_CONTEXT(Slab.NUMERIC),
    /** 最初の合法行動ごとの数値特徴量。 */
    ACTION_NUMERICS(Slab.NUMERIC),
    /** 各遷移のシャンテン・受入れ・面子状態。 */
    TRANSITION_NUMERICS(Slab.NUMERIC),
    /** 各待ち牌のRON・TSUMO別公開得点特徴量。 */
    WAIT_SCORES(Slab.NUMERIC);

    private final Slab slab;

    Tensor(Slab slab) {
      this.slab = slab;
    }

    /**
     * この論理的なテンソルを格納するホスト側の連続バッファを返す。
     *
     * @return カテゴリ値または数値の格納先
     */
    public Slab slab() {
      return slab;
    }

    /**
     * 一行が連続バッファ上で占める要素数を返す。
     *
     * @param bucket 行動・遷移の物理容量
     * @return 一行あたりのプリミティブ型要素数
     */
    public int elementsPerRow(DecisionBucket bucket) {
      int actions = bucket.legalActionCapacity();
      if (isTransition()) {
        return actions * bucket.actionTransitionCapacity() * elementsPerTransition();
      }
      return switch (this) {
        case STATE_CATEGORIES -> DecisionInputSchema.STATE_INT_COUNT;
        case ACTION_CATEGORIES -> actions * DecisionInputSchema.ACTION_INT_STRIDE;
        case ACTION_ROUTES -> actions * DecisionInputSchema.ACTION_ROUTE_STRIDE;
        case STATE_NUMERICS -> DecisionInputSchema.STATE_FLOAT_COUNT;
        case BOUNDARY_CONTEXT -> DecisionBoundaryContext.INPUT_SIZE;
        case ACTION_NUMERICS -> actions * DecisionInputSchema.ACTION_FLOAT_STRIDE;
        default -> throw new AssertionError(this);
      };
    }

    /** 実在遷移単位で格納するテンソルかを返します。 */
    public boolean isTransition() {
      return switch (this) {
        case TRANSITION_CATEGORIES,
            TRANSITION_TILES,
            WAIT_TILE_IDS,
            WAIT_YAKUS,
            TRANSITION_NUMERICS,
            WAIT_SCORES ->
            true;
        default -> false;
      };
    }

    /** 一つの遷移が持つ要素数を返します。 */
    public int elementsPerTransition() {
      return switch (this) {
        case TRANSITION_CATEGORIES -> DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE;
        case TRANSITION_TILES -> DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT;
        case WAIT_TILE_IDS -> DecisionInputSchema.MAX_WAIT_TILE_TYPES;
        case WAIT_YAKUS ->
            DecisionInputSchema.MAX_WAIT_TILE_TYPES
                * DecisionInputSchema.ACTION_TRANSITION_WAIT_YAKU_STRIDE;
        case TRANSITION_NUMERICS -> DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE;
        case WAIT_SCORES ->
            DecisionInputSchema.MAX_WAIT_TILE_TYPES
                * DecisionInputSchema.ACTION_TRANSITION_WAIT_FLOAT_STRIDE;
        default -> throw new IllegalArgumentException("not a transition tensor: " + this);
      };
    }

    /**
     * ネットワークが受け取る論理的な形状を返す。
     *
     * @param rows バッチ行数
     * @param actions 合法手容量
     * @param transitions 行動あたりの遷移容量
     * @return このテンソルの共通形式の形状
     */
    public Shape shape(long rows, long actions, long transitions) {
      return switch (this) {
        case STATE_CATEGORIES -> new Shape(rows, DecisionInputSchema.STATE_INT_COUNT);
        case ACTION_CATEGORIES -> new Shape(rows, actions, DecisionInputSchema.ACTION_INT_STRIDE);
        case ACTION_ROUTES -> new Shape(rows, actions, DecisionInputSchema.ACTION_ROUTE_STRIDE);
        case TRANSITION_CATEGORIES ->
            new Shape(rows, actions, transitions, DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE);
        case TRANSITION_TILES ->
            new Shape(rows, actions, transitions, DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT);
        case WAIT_TILE_IDS ->
            new Shape(rows, actions, transitions, DecisionInputSchema.MAX_WAIT_TILE_TYPES);
        case WAIT_YAKUS ->
            new Shape(
                rows,
                actions,
                transitions,
                DecisionInputSchema.MAX_WAIT_TILE_TYPES,
                DecisionInputSchema.ACTION_TRANSITION_WAIT_YAKU_STRIDE);
        case STATE_NUMERICS -> new Shape(rows, DecisionInputSchema.STATE_FLOAT_COUNT);
        case BOUNDARY_CONTEXT -> new Shape(rows, DecisionBoundaryContext.INPUT_SIZE);
        case ACTION_NUMERICS -> new Shape(rows, actions, DecisionInputSchema.ACTION_FLOAT_STRIDE);
        case TRANSITION_NUMERICS ->
            new Shape(
                rows, actions, transitions, DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE);
        case WAIT_SCORES ->
            new Shape(
                rows,
                actions,
                transitions,
                DecisionInputSchema.MAX_WAIT_TILE_TYPES,
                DecisionInputSchema.ACTION_TRANSITION_WAIT_FLOAT_STRIDE);
      };
    }
  }

  /**
   * 一つの論理的なテンソルが占める連続バッファ区間。
   *
   * @param tensor 論理的なテンソルの種類
   * @param slabOffset カテゴリ値または数値連続バッファ内の先頭プリミティブ型オフセット
   * @param elementsPerRow 1行が占めるプリミティブ型数
   * @param elementCount 配置全行が占めるプリミティブ型数
   */
  public record Region(Tensor tensor, int slabOffset, int elementsPerRow, int elementCount) {
    /**
     * 指定行の先頭オフセットを返す。
     *
     * @param row 0始まりの行インデックス
     * @return 領域が属する連続バッファ内のプリミティブ型オフセット
     */
    public int rowOffset(int row) {
      return slabOffset + row * elementsPerRow;
    }

    /**
     * 指定行数と容量区分に対応する論理的なテンソル形状を返す。
     *
     * @param rows バッチ行数
     * @param bucket 行動・遷移の物理容量
     * @return この領域のネットワーク入力形状
     */
    public Shape shape(int rows, DecisionBucket bucket) {
      return tensor.shape(rows, bucket.legalActionCapacity(), bucket.actionTransitionCapacity());
    }
  }

  private final int capacity;
  private final DecisionBucket bucket;
  private final Region[] regions = new Region[TENSORS.size()];
  private final int categoricalElementCount;
  private final int numericElementCount;

  /**
   * 指定容量の二本のホスト側の連続バッファを記述する配置を作る。
   *
   * @param capacity 最大行数
   * @param bucket 行動・遷移の固定容量区分
   */
  public DecisionInputLayout(int capacity, DecisionBucket bucket) {
    this(capacity, bucket, true);
  }

  /** 行ごとの実遷移記憶領域とは別に、状態・行動だけをバッチへ確保します。 */
  static DecisionInputLayout stateAndActions(int capacity, DecisionBucket bucket) {
    return new DecisionInputLayout(capacity, bucket, false);
  }

  private DecisionInputLayout(int capacity, DecisionBucket bucket, boolean includeTransitions) {
    if (capacity < 1) {
      throw new IllegalArgumentException("capacity must be positive");
    }
    this.capacity = capacity;
    this.bucket = bucket;
    int categoricalCursor = 0;
    int numericCursor = 0;
    for (Tensor tensor : TENSORS) {
      int elementsPerRow =
          !includeTransitions && tensor.isTransition() ? 0 : tensor.elementsPerRow(bucket);
      int elementCount = Math.multiplyExact(capacity, elementsPerRow);
      int offset = tensor.slab() == Slab.CATEGORICAL ? categoricalCursor : numericCursor;
      regions[tensor.ordinal()] = new Region(tensor, offset, elementsPerRow, elementCount);
      if (tensor.slab() == Slab.CATEGORICAL) {
        categoricalCursor = Math.addExact(categoricalCursor, elementCount);
      } else {
        numericCursor = Math.addExact(numericCursor, elementCount);
      }
    }
    categoricalElementCount = categoricalCursor;
    numericElementCount = numericCursor;
  }

  /**
   * 最大行数を返す。
   *
   * @return 連続バッファへ格納できる行数
   */
  public int capacity() {
    return capacity;
  }

  /**
   * 行動・遷移容量区分を返す。
   *
   * @return この配置が固定する容量区分
   */
  public DecisionBucket bucket() {
    return bucket;
  }

  /**
   * 指定論理的なテンソルの連続バッファ区間を返す。
   *
   * @param tensor 取得する論理的なテンソル
   * @return テンソルに対応する連続領域
   */
  public Region region(Tensor tensor) {
    return regions[tensor.ordinal()];
  }

  /**
   * カテゴリ値連続バッファ全体のshort要素数を返す。
   *
   * @return カテゴリ値要素数
   */
  public int categoricalElementCount() {
    return categoricalElementCount;
  }

  /**
   * 数値連続バッファ全体のfloat要素数を返す。
   *
   * @return 数値要素数
   */
  public int numericElementCount() {
    return numericElementCount;
  }

  /**
   * 指定連続バッファのプリミティブ型要素数を返す。
   *
   * @param slab 対象連続バッファ
   * @return 配置全行分の要素数
   */
  public int elementCount(Slab slab) {
    return slab == Slab.CATEGORICAL ? categoricalElementCount : numericElementCount;
  }

  /**
   * 二本のホスト側の連続バッファが占める入力バイト数を返す。
   *
   * @return カテゴリ値と数値を合計したバイト数
   */
  public long inputByteCount() {
    return (long) Short.BYTES * categoricalElementCount + (long) Float.BYTES * numericElementCount;
  }

  /**
   * 可変のバッチ/行動/遷移軸を持つネットワーク初期化形状を共通形式の順で返す。
   *
   * @return {@link Tensor}の宣言順に並んだ入力形状
   */
  public static Shape[] networkInputShapes() {
    Shape[] shapes = new Shape[TENSORS.size()];
    for (Tensor tensor : TENSORS) {
      shapes[tensor.ordinal()] = tensor.shape(-1, -1, -1);
    }
    return shapes;
  }

  /**
   * スキーマ互換性識別子へ含める論理的なテンソル契約を返す。
   *
   * @return テンソル順序と牌種数を表す安定記述情報
   */
  public static String descriptor() {
    return TENSORS + ";tiles=" + Tile.NUM_TILE_TYPES;
  }

  static List<Tensor> tensors() {
    return TENSORS;
  }

  static int tensorCount() {
    return TENSORS.size();
  }
}
