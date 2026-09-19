package com.epsilon.major.ai.decision.input;

import com.epsilon.core.GameState;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionBuffer;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;

/**
 * Decision のカテゴリ特徴量と数値特徴量を保持し、推論・保存・学習へ渡す。
 *
 * <p>学習・復号済み入力は固定形状の二連続バッファを所有する。推論は状態・行動だけをバッチへ置き、
 * 解析済みの各行が実在する遷移分の配列を所有する。全経路のエンコーダーは同じ型付き書き込み処理を使い、 保存・学習の固定形状への展開は行複製の境界に集約する。
 *
 * <p>カテゴリ値 IDは非負のshort範囲で格納する。状態末尾の公開履歴メタデータだけは符号ビットも使用する未加工の ビット列であり、カテゴリ辞書へ渡さず専用取得メソッドで復号する。
 */
public final class DecisionHostInputs {

  // 保存・学習境界のパディングを、ヒープ/ダイレクトバッファの両方へまとめて書く読取り専用まとまり。
  private static final short[] CATEGORY_PADDING = new short[1024];
  private static final float[] NUMERIC_PADDING = new float[1024];

  private final DecisionInputLayout layout;
  private final short[] categoricalSlab;
  private final float[] numericSlab;
  private final DecisionTransitionStorage transitions;

  /**
   * 指定容量と容量区分に対応する空の標準形式の連続バッファを確保する。
   *
   * @param capacity 格納できる最大行数
   * @param bucket 行動・遷移の物理容量
   */
  public DecisionHostInputs(int capacity, DecisionBucket bucket) {
    this(new DecisionInputLayout(capacity, bucket), null);
  }

  static DecisionHostInputs inference(int capacity, DecisionBucket bucket) {
    return new DecisionHostInputs(
        DecisionInputLayout.stateAndActions(capacity, bucket),
        new DecisionTransitionStorage(capacity, bucket.legalActionCapacity()));
  }

  private DecisionHostInputs(DecisionInputLayout layout, DecisionTransitionStorage transitions) {
    this.layout = layout;
    this.transitions = transitions;
    categoricalSlab = new short[layout.categoricalElementCount()];
    numericSlab = new float[layout.numericElementCount()];
  }

  private DecisionHostInputs(
      DecisionInputLayout layout, short[] categoricalSlab, float[] numericSlab) {
    this.layout = layout;
    transitions = null;
    this.categoricalSlab = categoricalSlab;
    this.numericSlab = numericSlab;
  }

  /**
   * 永続化済み標準形式の連続バッファを独立所有記憶領域へ復元する。
   *
   * @param capacity データ本体に含まれる行数
   * @param bucket データ本体の行動・遷移容量区分
   * @param categoricalValues 標準形式の順のカテゴリ値要素
   * @param numericValues 標準形式の順の数値要素
   * @return 入力配列を複製して所有するホスト記憶領域
   */
  public static DecisionHostInputs fromEncoded(
      int capacity, DecisionBucket bucket, short[] categoricalValues, float[] numericValues) {
    DecisionHostInputs inputs = new DecisionHostInputs(capacity, bucket);
    requireLength(
        categoricalValues.length, inputs.categoricalSlab.length, "categorical input slab");
    requireLength(numericValues.length, inputs.numericSlab.length, "numeric input slab");
    System.arraycopy(categoricalValues, 0, inputs.categoricalSlab, 0, categoricalValues.length);
    System.arraycopy(numericValues, 0, inputs.numericSlab, 0, numericValues.length);
    return inputs;
  }

  /**
   * 復号済み標準形式の連続バッファの所有権を受け取り、複製せずにホスト記憶領域を作る。
   *
   * <p>呼出元は返却後に配列を読み書きしてはならない。永続化符号化・復号処理など、配列を新規確保して全要素を書き終えた境界だけで使用する。
   *
   * @param capacity データ本体に含まれる行数
   * @param bucket データ本体の行動・遷移容量区分
   * @param categoricalSlab 所有権を渡す標準形式のカテゴリ値連続バッファ
   * @param numericSlab 所有権を渡す標準形式の数値連続バッファ
   * @return 渡された配列を直接所有するホスト記憶領域
   */
  static DecisionHostInputs takeEncoded(
      int capacity, DecisionBucket bucket, short[] categoricalSlab, float[] numericSlab) {
    DecisionInputLayout layout = new DecisionInputLayout(capacity, bucket);
    requireLength(
        categoricalSlab.length, layout.categoricalElementCount(), "categorical input slab");
    requireLength(numericSlab.length, layout.numericElementCount(), "numeric input slab");
    return new DecisionHostInputs(layout, categoricalSlab, numericSlab);
  }

  /**
   * バッチ内の状態・行動と、固定長入力では遷移も含む配置を返す。
   *
   * @return この記憶領域の配置
   */
  public DecisionInputLayout layout() {
    return layout;
  }

  /**
   * 行動・遷移容量区分を返す。
   *
   * @return この記憶領域の容量区分
   */
  public DecisionBucket bucket() {
    return layout.bucket();
  }

  /**
   * 格納できる最大行数を返す。
   *
   * @return 行容量
   */
  public int capacity() {
    return layout.capacity();
  }

  /**
   * 固定長の学習・復号済み入力が所有するカテゴリ値連続バッファを返す。
   *
   * <p>返り値は内部実体となる配列である。通常の特徴量書き込みには使わず、型付き書き込み処理を使用する。
   *
   * @return 標準形式のテンソル順に連結した内部short配列
   */
  public short[] denseCategories() {
    requireDense();
    return categoricalSlab;
  }

  /** 公開履歴の指定区間の関係コードを、未加工の shortの符号に依存せず返す。 */
  public int publicHistoryRelation(int row, int query, int token, int segment) {
    int offset =
        layout.region(DecisionInputLayout.Tensor.STATE_CATEGORIES).rowOffset(row)
            + DecisionInputSchema.PUBLIC_HISTORY_RELATION_OFFSET
            + (query * DecisionPublicHistory.TOKEN_COUNT + token)
                * DecisionPublicHistory.PACKED_WORDS;
    return DecisionPublicHistory.tableIndex(
            categoricalSlab[offset], categoricalSlab[offset + 1], segment)
        - segment * DecisionPublicHistory.ENTRIES_PER_SEGMENT;
  }

  /**
   * 固定長の学習・復号済み入力が所有する数値連続バッファを返す。
   *
   * <p>返り値は内部実体となる配列である。通常の特徴量書き込みには使わず、型付き書き込み処理を使用する。
   *
   * @return 標準形式のテンソル順に連結した内部float配列
   */
  public float[] denseNumerics() {
    requireDense();
    return numericSlab;
  }

  boolean isDense() {
    return transitions == null;
  }

  private void requireDense() {
    if (transitions != null) {
      throw new IllegalStateException("compact inputs must be exported through a row slice");
    }
  }

  void prepareTransitions(int row, EngineDecisionBuffer decision) {
    if (transitions != null) {
      transitions.prepare(row, decision);
    }
  }

  short[] stateCategories() {
    return categoricalSlab;
  }

  float[] stateNumerics() {
    return numericSlab;
  }

  short[] transitionCategories(int row) {
    return transitions == null ? categoricalSlab : transitions.categories(row);
  }

  float[] transitionNumerics(int row) {
    return transitions == null ? numericSlab : transitions.numerics(row);
  }

  int[] transitionOffsets() {
    return transitions == null ? null : transitions.actionOffsets();
  }

  int transitionRowBase(int row) {
    return transitions == null ? 0 : transitions.rowBase(row);
  }

  void commitActionCount(int row, int actions) {
    if (transitions != null) {
      transitions.commitActionCount(row, actions);
    }
  }

  int compactActionCount(int row) {
    return transitions.actionCount(row);
  }

  int compactTransitionCount(int row) {
    // 状態のみの行を準備した際のダミー遷移は、推論入力には含めない。
    return transitions.actionCount(row) == 0 ? 0 : transitions.count(row);
  }

  int transitionCapacity(int row, int action) {
    return transitions == null
        ? bucket().actionTransitionCapacity()
        : transitions.count(row, action);
  }

  int transitionRegionBase(int row, DecisionInputLayout.Tensor tensor) {
    return transitions == null
        ? layout.region(tensor).rowOffset(row)
        : transitions.regionBase(row, tensor);
  }

  private int transitionIndex(int row, int action, int transition) {
    return transitions == null
        ? action * bucket().actionTransitionCapacity() + transition
        : transitions.index(row, action, transition);
  }

  /**
   * 指定行へ型付きフィールドを書き込む書き込み処理を返す。
   *
   * @param row 容量内の0始まり行インデックス
   * @return この記憶領域の指定行へ書き込む書き込み処理
   */
  public DecisionInputWriter writer(int row) {
    return new DecisionInputWriter(this, row);
  }

  /**
   * 局単位のカテゴリ値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int roundCategory(int row, DecisionInputSchema.RoundInt field) {
    return categoricalSlab[roundCategoryOffset(row, field)];
  }

  /**
   * 局単位の数値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float roundNumeric(int row, DecisionInputSchema.RoundFloat field) {
    return numericSlab[roundNumericOffset(row, field)];
  }

  /** 指定行の重みを固定したGRP局境界事前予測を独立所有配列で返す。 */
  public float[] boundaryRankPrior(int row) {
    int offset = boundaryContextOffset(row);
    return java.util.Arrays.copyOfRange(
        numericSlab, offset, offset + DecisionBoundaryContext.GRP_RANK_SIZE);
  }

  /** 指定席の4順位周辺分布だけを独立配列として返す。 */
  public float[] boundaryRankPrior(int row, int seat) {
    if (seat < 0 || seat >= GameState.NUM_PLAYERS) {
      throw new IllegalArgumentException("seat must be 0-3: " + seat);
    }
    int rankCount = DecisionBoundaryContext.GRP_RANK_SIZE / GameState.NUM_PLAYERS;
    int offset = boundaryContextOffset(row) + seat * rankCount;
    return java.util.Arrays.copyOfRange(numericSlab, offset, offset + rankCount);
  }

  /**
   * 指定相対席のプレイヤーカテゴリ値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param relativeSeat 観測者基準の相対席
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int playerCategory(int row, int relativeSeat, DecisionInputSchema.PlayerInt field) {
    return categoricalSlab[playerCategoryOffset(row, relativeSeat, field)];
  }

  /**
   * 指定相対席のプレイヤー数値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param relativeSeat 観測者基準の相対席
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float playerNumeric(int row, int relativeSeat, DecisionInputSchema.PlayerFloat field) {
    return numericSlab[playerNumericOffset(row, relativeSeat, field)];
  }

  /**
   * 指定牌種のカテゴリ値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param tileType 34牌種インデックス
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int tileCategory(int row, int tileType, DecisionInputSchema.TileInt field) {
    return categoricalSlab[tileCategoryOffset(row, tileType, field)];
  }

  /**
   * 指定牌種の数値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param tileType 34牌種インデックス
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float tileNumeric(int row, int tileType, DecisionInputSchema.TileFloat field) {
    return numericSlab[tileNumericOffset(row, tileType, field)];
  }

  /**
   * 行動候補のカテゴリ値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 容量区分内の行動候補の位置
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int actionCategory(int row, int actionSlot, DecisionInputSchema.ActionInt field) {
    return categoricalSlab[actionCategoryOffset(row, actionSlot, field)];
  }

  /** 絶対席順の現在点を100点単位で読む。 */
  public int pointLedger100(int row, int absoluteSeat) {
    return categoricalSlab[pointLedger100Offset(row, absoluteSeat)];
  }

  /** 即時和了候補のPoint Factを読む。 */
  public int actionWinFact(int row, int actionSlot, DecisionInputSchema.WinFact fact) {
    return categoricalSlab[actionWinFactOffset(row, actionSlot, fact)];
  }

  /**
   * 行動候補の方策グラフ経路 IDを読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 容量区分内の行動候補の位置
   * @param field 経路階層
   * @return 符号化済み経路 ID
   */
  public int actionRoute(int row, int actionSlot, DecisionInputSchema.ActionRoute field) {
    return categoricalSlab[actionRouteOffset(row, actionSlot, field)];
  }

  /**
   * 行動遷移のカテゴリ値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 実在する合法行動候補の位置
   * @param transitionSlot 行動内の実在遷移候補の位置
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int transitionCategory(
      int row, int actionSlot, int transitionSlot, DecisionInputSchema.ActionTransitionInt field) {
    return transitionCategories(row)[
        transitionCategoryOffset(row, actionSlot, transitionSlot, field)];
  }

  /**
   * 行動遷移の数値特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 実在する合法行動候補の位置
   * @param transitionSlot 行動内の実在遷移候補の位置
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float transitionNumeric(
      int row,
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionFloat field) {
    return transitionNumerics(row)[transitionNumericOffset(row, actionSlot, transitionSlot, field)];
  }

  /**
   * 遷移行動適用後の状態の指定牌種特徴量を読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 実在する合法行動候補の位置
   * @param transitionSlot 行動内の実在遷移候補の位置
   * @param tileType 34牌種インデックス
   * @return 符号化済み遷移-牌値
   */
  public int transitionTile(int row, int actionSlot, int transitionSlot, int tileType) {
    return transitionCategories(row)[
        transitionTileOffset(row, actionSlot, transitionSlot, tileType)];
  }

  /**
   * 遷移の待ち格納位置に対応する牌種IDを読む。
   *
   * @param row 容量内の行インデックス
   * @param actionSlot 実在する合法行動候補の位置
   * @param transitionSlot 行動内の実在遷移候補の位置
   * @param waitSlot 待ち集合内の格納位置
   * @return 符号化済み待ち牌種ID
   */
  public int waitTile(int row, int actionSlot, int transitionSlot, int waitSlot) {
    return transitionCategories(row)[waitTileOffset(row, actionSlot, transitionSlot, waitSlot)];
  }

  /** 指定待ち牌のRON・TSUMO別Point Factを読む。 */
  public int waitWinFact(
      int row,
      int actionSlot,
      int transitionSlot,
      int waitSlot,
      DecisionInputSchema.WaitWinType winType,
      DecisionInputSchema.WinFact fact) {
    return transitionCategories(row)[
        waitWinFactOffset(row, actionSlot, transitionSlot, waitSlot, winType, fact)];
  }

  int playerMemoryPresentCount(int fromInclusive, int rowCount) {
    int count = 0;
    for (int row = fromInclusive; row < fromInclusive + rowCount; row++) {
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        count +=
            1
                + playerCategory(row, player, DecisionInputSchema.PlayerInt.RIVER_COUNT)
                + playerCategory(row, player, DecisionInputSchema.PlayerInt.MELD_COUNT);
      }
    }
    return count;
  }

  void copyPlayerMemoryPresentIndicesTo(
      int fromInclusive, int rowCount, int destinationRowBase, IntBuffer destination) {
    int tokensPerPlayer =
        1
            + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            + DecisionInputSchema.MAX_MELDS_PER_PLAYER;
    for (int localRow = 0; localRow < rowCount; localRow++) {
      int sourceRow = fromInclusive + localRow;
      for (int player = 0; player < GameState.NUM_PLAYERS; player++) {
        int playerOffset =
            ((destinationRowBase + localRow) * GameState.NUM_PLAYERS + player) * tokensPerPlayer;
        destination.put(playerOffset);
        int riverCount =
            playerCategory(sourceRow, player, DecisionInputSchema.PlayerInt.RIVER_COUNT);
        for (int river = 0; river < riverCount; river++) {
          destination.put(playerOffset + 1 + river);
        }
        int meldCount = playerCategory(sourceRow, player, DecisionInputSchema.PlayerInt.MELD_COUNT);
        int meldOffset = playerOffset + 1 + DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER;
        for (int meld = 0; meld < meldCount; meld++) {
          destination.put(meldOffset + meld);
        }
      }
    }
  }

  /** 検証済み行を、学習・保存用の同一または大きい固定長容量区分へコピーする。 */
  void copyTrustedRowFrom(DecisionHostInputs source, int sourceRow, int destinationRow) {
    if (bucket().legalActionCapacity() < source.bucket().legalActionCapacity()
        || bucket().actionTransitionCapacity() < source.bucket().actionTransitionCapacity()) {
      throw new IllegalArgumentException("destination bucket is smaller than source bucket");
    }
    copyTensorRow(
        source,
        DecisionInputLayout.Tensor.STATE_CATEGORIES,
        sourceRow,
        destinationRow,
        DecisionInputSchema.STATE_INT_COUNT);
    copyTensorRow(
        source,
        DecisionInputLayout.Tensor.STATE_NUMERICS,
        sourceRow,
        destinationRow,
        DecisionInputSchema.STATE_FLOAT_COUNT);
    copyTensorRow(
        source,
        DecisionInputLayout.Tensor.BOUNDARY_CONTEXT,
        sourceRow,
        destinationRow,
        DecisionBoundaryContext.INPUT_SIZE);
    copyTensorRow(
        source,
        DecisionInputLayout.Tensor.POINT_LEDGER_100,
        sourceRow,
        destinationRow,
        GameState.NUM_PLAYERS);
    for (int action = 0; action < source.bucket().legalActionCapacity(); action++) {
      copy(
          source.categoricalSlab,
          source.actionCategoryOffset(sourceRow, action, DecisionInputSchema.ActionInt.ID),
          categoricalSlab,
          actionCategoryOffset(destinationRow, action, DecisionInputSchema.ActionInt.ID),
          DecisionInputSchema.ACTION_INT_STRIDE);
      copy(
          source.categoricalSlab,
          source.actionWinFactOffset(sourceRow, action, DecisionInputSchema.WinFact.VALID),
          categoricalSlab,
          actionWinFactOffset(destinationRow, action, DecisionInputSchema.WinFact.VALID),
          DecisionInputSchema.ACTION_WIN_FACT_STRIDE);
      copy(
          source.categoricalSlab,
          source.actionRouteOffset(
              sourceRow, action, DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE),
          categoricalSlab,
          actionRouteOffset(
              destinationRow,
              action,
              DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE),
          DecisionInputSchema.ACTION_ROUTE_STRIDE);
      for (int transition = 0;
          transition < source.transitionCapacity(sourceRow, action);
          transition++) {
        copy(
            source.transitionCategories(sourceRow),
            source.transitionCategoryOffset(
                sourceRow, action, transition, DecisionInputSchema.ActionTransitionInt.KIND),
            categoricalSlab,
            transitionCategoryOffset(
                destinationRow, action, transition, DecisionInputSchema.ActionTransitionInt.KIND),
            DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE);
        copy(
            source.transitionCategories(sourceRow),
            source.transitionTileOffset(sourceRow, action, transition, 0),
            categoricalSlab,
            transitionTileOffset(destinationRow, action, transition, 0),
            DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT);
        copy(
            source.transitionCategories(sourceRow),
            source.waitTileOffset(sourceRow, action, transition, 0),
            categoricalSlab,
            waitTileOffset(destinationRow, action, transition, 0),
            DecisionInputSchema.MAX_WAIT_TILE_TYPES);
        copy(
            source.transitionCategories(sourceRow),
            source.waitWinFactOffset(
                sourceRow,
                action,
                transition,
                0,
                DecisionInputSchema.WaitWinType.RON,
                DecisionInputSchema.WinFact.VALID),
            categoricalSlab,
            waitWinFactOffset(
                destinationRow,
                action,
                transition,
                0,
                DecisionInputSchema.WaitWinType.RON,
                DecisionInputSchema.WinFact.VALID),
            DecisionInputSchema.MAX_WAIT_TILE_TYPES
                * DecisionInputSchema.ACTION_TRANSITION_WAIT_WIN_FACT_STRIDE);
        copy(
            source.transitionNumerics(sourceRow),
            source.transitionNumericOffset(
                sourceRow,
                action,
                transition,
                DecisionInputSchema.ActionTransitionFloat.NORMALIZED_MIN_SHANTEN),
            numericSlab,
            transitionNumericOffset(
                destinationRow,
                action,
                transition,
                DecisionInputSchema.ActionTransitionFloat.NORMALIZED_MIN_SHANTEN),
            DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE);
      }
    }
  }

  void copyRowsTo(
      DecisionInputLayout.Slab slab,
      int fromInclusive,
      int rowCount,
      ShortBuffer categoricalDestination,
      FloatBuffer numericDestination) {
    for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
      if (tensor.slab() == slab) {
        copyTensorRowsTo(
            tensor, fromInclusive, rowCount, categoricalDestination, numericDestination);
      }
    }
  }

  void copyTensorRowsTo(
      DecisionInputLayout.Tensor tensor,
      int fromInclusive,
      int rowCount,
      ShortBuffer categoricalDestination,
      FloatBuffer numericDestination) {
    if (transitions == null || !tensor.isTransition()) {
      DecisionInputLayout.Region region = layout.region(tensor);
      int offset = region.rowOffset(fromInclusive);
      int length = rowCount * region.elementsPerRow();
      if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) {
        categoricalDestination.put(categoricalSlab, offset, length);
      } else {
        numericDestination.put(numericSlab, offset, length);
      }
      return;
    }
    int stride = tensor.elementsPerTransition();
    for (int row = fromInclusive; row < fromInclusive + rowCount; row++) {
      for (int action = 0; action < bucket().legalActionCapacity(); action++) {
        int count = transitions.count(row, action);
        copyTransitionRowsTo(
            tensor, row, action, count, stride, categoricalDestination, numericDestination);
        int padding = (bucket().actionTransitionCapacity() - count) * stride;
        // 保存・学習の固定形状境界では、再利用バッファの未使用領域も明示的にゼロへ戻す。
        if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) {
          while (padding > 0) {
            int countToWrite = Math.min(padding, CATEGORY_PADDING.length);
            categoricalDestination.put(CATEGORY_PADDING, 0, countToWrite);
            padding -= countToWrite;
          }
        } else {
          while (padding > 0) {
            int countToWrite = Math.min(padding, NUMERIC_PADDING.length);
            numericDestination.put(NUMERIC_PADDING, 0, countToWrite);
            padding -= countToWrite;
          }
        }
      }
    }
  }

  void copyTransitionRowsTo(
      DecisionInputLayout.Tensor tensor,
      int row,
      int action,
      int count,
      int stride,
      ShortBuffer categoricalDestination,
      FloatBuffer numericDestination) {
    int offset = transitionRegionBase(row, tensor) + transitionIndex(row, action, 0) * stride;
    int length = count * stride;
    if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) {
      categoricalDestination.put(transitionCategories(row), offset, length);
    } else {
      numericDestination.put(transitionNumerics(row), offset, length);
    }
  }

  int roundCategoryOffset(int row, DecisionInputSchema.RoundInt field) {
    return stateCategoryRowOffset(row) + field.ordinal();
  }

  int roundNumericOffset(int row, DecisionInputSchema.RoundFloat field) {
    return stateNumericRowOffset(row) + field.ordinal();
  }

  int boundaryContextOffset(int row) {
    return layout.region(DecisionInputLayout.Tensor.BOUNDARY_CONTEXT).rowOffset(row);
  }

  int playerCategoryOffset(int row, int relativeSeat, DecisionInputSchema.PlayerInt field) {
    return stateCategoryRowOffset(row)
        + DecisionInputSchema.ROUND_INT_COUNT
        + relativeSeat * DecisionInputSchema.PLAYER_INT_STRIDE
        + field.ordinal();
  }

  int playerNumericOffset(int row, int relativeSeat, DecisionInputSchema.PlayerFloat field) {
    return stateNumericRowOffset(row)
        + DecisionInputSchema.ROUND_FLOAT_COUNT
        + relativeSeat * DecisionInputSchema.PLAYER_FLOAT_STRIDE
        + field.ordinal();
  }

  int tileCategoryOffset(int row, int tileType, DecisionInputSchema.TileInt field) {
    return stateCategoryRowOffset(row)
        + DecisionInputSchema.ROUND_INT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
        + tileType * DecisionInputSchema.TILE_INT_STRIDE
        + field.ordinal();
  }

  int tileNumericOffset(int row, int tileType, DecisionInputSchema.TileFloat field) {
    return stateNumericRowOffset(row)
        + DecisionInputSchema.ROUND_FLOAT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
        + tileType * DecisionInputSchema.TILE_FLOAT_STRIDE
        + field.ordinal();
  }

  int riverCategoryOffset(
      int row, int relativeSeat, int riverIndex, DecisionInputSchema.RiverInt field) {
    return stateCategoryRowOffset(row)
        + DecisionInputSchema.ROUND_INT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
        + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE
        + (relativeSeat * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + riverIndex)
            * DecisionInputSchema.RIVER_INT_STRIDE
        + field.ordinal();
  }

  int riverNumericOffset(
      int row, int relativeSeat, int riverIndex, DecisionInputSchema.RiverFloat field) {
    return stateNumericRowOffset(row)
        + DecisionInputSchema.ROUND_FLOAT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
        + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_FLOAT_STRIDE
        + (relativeSeat * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER + riverIndex)
            * DecisionInputSchema.RIVER_FLOAT_STRIDE
        + field.ordinal();
  }

  int meldCategoryOffset(
      int row, int relativeSeat, int meldIndex, DecisionInputSchema.MeldInt field) {
    return stateCategoryRowOffset(row)
        + DecisionInputSchema.ROUND_INT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_INT_STRIDE
        + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_INT_STRIDE
        + GameState.NUM_PLAYERS
            * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            * DecisionInputSchema.RIVER_INT_STRIDE
        + (relativeSeat * DecisionInputSchema.MAX_MELDS_PER_PLAYER + meldIndex)
            * DecisionInputSchema.MELD_INT_STRIDE
        + field.ordinal();
  }

  int meldNumericOffset(
      int row, int relativeSeat, int meldIndex, DecisionInputSchema.MeldFloat field) {
    return stateNumericRowOffset(row)
        + DecisionInputSchema.ROUND_FLOAT_COUNT
        + GameState.NUM_PLAYERS * DecisionInputSchema.PLAYER_FLOAT_STRIDE
        + Tile.NUM_TILE_TYPES * DecisionInputSchema.TILE_FLOAT_STRIDE
        + GameState.NUM_PLAYERS
            * DecisionInputSchema.MAX_RIVER_EVENTS_PER_PLAYER
            * DecisionInputSchema.RIVER_FLOAT_STRIDE
        + (relativeSeat * DecisionInputSchema.MAX_MELDS_PER_PLAYER + meldIndex)
            * DecisionInputSchema.MELD_FLOAT_STRIDE
        + field.ordinal();
  }

  int actionCategoryOffset(int row, int actionSlot, DecisionInputSchema.ActionInt field) {
    return layout.region(DecisionInputLayout.Tensor.ACTION_CATEGORIES).rowOffset(row)
        + actionSlot * DecisionInputSchema.ACTION_INT_STRIDE
        + field.ordinal();
  }

  int pointLedger100Offset(int row, int absoluteSeat) {
    return layout.region(DecisionInputLayout.Tensor.POINT_LEDGER_100).rowOffset(row) + absoluteSeat;
  }

  int actionWinFactOffset(int row, int actionSlot, DecisionInputSchema.WinFact fact) {
    return layout.region(DecisionInputLayout.Tensor.ACTION_WIN_FACTS).rowOffset(row)
        + actionSlot * DecisionInputSchema.ACTION_WIN_FACT_STRIDE
        + fact.ordinal();
  }

  int actionRouteOffset(int row, int actionSlot, DecisionInputSchema.ActionRoute field) {
    return layout.region(DecisionInputLayout.Tensor.ACTION_ROUTES).rowOffset(row)
        + actionSlot * DecisionInputSchema.ACTION_ROUTE_STRIDE
        + field.ordinal();
  }

  int transitionCategoryOffset(
      int row, int actionSlot, int transitionSlot, DecisionInputSchema.ActionTransitionInt field) {
    return transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_CATEGORIES)
        + (transitionIndex(row, actionSlot, transitionSlot))
            * DecisionInputSchema.ACTION_TRANSITION_INT_STRIDE
        + field.ordinal();
  }

  int transitionNumericOffset(
      int row,
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionFloat field) {
    return transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_NUMERICS)
        + (transitionIndex(row, actionSlot, transitionSlot))
            * DecisionInputSchema.ACTION_TRANSITION_FLOAT_STRIDE
        + field.ordinal();
  }

  int transitionTileOffset(int row, int actionSlot, int transitionSlot, int tileType) {
    return transitionRegionBase(row, DecisionInputLayout.Tensor.TRANSITION_TILES)
        + (transitionIndex(row, actionSlot, transitionSlot))
            * DecisionInputSchema.ACTION_TRANSITION_TILE_COUNT
        + tileType;
  }

  int waitTileOffset(int row, int actionSlot, int transitionSlot, int waitSlot) {
    return transitionRegionBase(row, DecisionInputLayout.Tensor.WAIT_TILE_IDS)
        + (transitionIndex(row, actionSlot, transitionSlot))
            * DecisionInputSchema.MAX_WAIT_TILE_TYPES
        + waitSlot;
  }

  int waitWinFactOffset(
      int row,
      int actionSlot,
      int transitionSlot,
      int waitSlot,
      DecisionInputSchema.WaitWinType winType,
      DecisionInputSchema.WinFact fact) {
    return transitionRegionBase(row, DecisionInputLayout.Tensor.WAIT_WIN_FACTS)
        + ((transitionIndex(row, actionSlot, transitionSlot))
                    * DecisionInputSchema.MAX_WAIT_TILE_TYPES
                + waitSlot)
            * DecisionInputSchema.ACTION_TRANSITION_WAIT_WIN_FACT_STRIDE
        + winType.ordinal() * DecisionInputSchema.ACTION_WIN_FACT_STRIDE
        + fact.ordinal();
  }

  private int stateCategoryRowOffset(int row) {
    return layout.region(DecisionInputLayout.Tensor.STATE_CATEGORIES).rowOffset(row);
  }

  private int stateNumericRowOffset(int row) {
    return layout.region(DecisionInputLayout.Tensor.STATE_NUMERICS).rowOffset(row);
  }

  private void copyTensorRow(
      DecisionHostInputs source,
      DecisionInputLayout.Tensor tensor,
      int sourceRow,
      int destinationRow,
      int length) {
    DecisionInputLayout.Region sourceRegion = source.layout.region(tensor);
    DecisionInputLayout.Region destinationRegion = layout.region(tensor);
    if (tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL) {
      copy(
          source.categoricalSlab,
          sourceRegion.rowOffset(sourceRow),
          categoricalSlab,
          destinationRegion.rowOffset(destinationRow),
          length);
    } else {
      copy(
          source.numericSlab,
          sourceRegion.rowOffset(sourceRow),
          numericSlab,
          destinationRegion.rowOffset(destinationRow),
          length);
    }
  }

  private static void requireLength(int actual, int expected, String label) {
    if (actual != expected) {
      throw new IllegalArgumentException(
          label + " length mismatch: actual=" + actual + " expected=" + expected);
    }
  }

  private static void copy(
      short[] source, int sourceOffset, short[] destination, int destinationOffset, int length) {
    System.arraycopy(source, sourceOffset, destination, destinationOffset, length);
  }

  private static void copy(
      float[] source, int sourceOffset, float[] destination, int destinationOffset, int length) {
    System.arraycopy(source, sourceOffset, destination, destinationOffset, length);
  }
}
