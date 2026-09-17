package com.epsilon.nano.ai.decision.input;

import ai.djl.ndarray.NDArray;
import com.epsilon.core.Action;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.ShortBuffer;
import java.util.List;

/**
 * Decision の入力と教師値を CPU 上に保持するバッチ。
 *
 * <p>入力は {@link DecisionHostInputs}、教師値は {@link DecisionTrainingTargets}
 * が保持する。この型は構築済みの行順、有効範囲、行動と遷移の連続性を管理する。
 *
 * <p>{@link #copyRows} は独立した複製を作る。{@link #sliceRows} は元のバッチを強参照するビューを作り、非同期処理が終わるまで元データを保持する。
 */
public final class DecisionHostBatch {

  private static final int CALL_GATE = 1;
  private static final int RON_GATE = 1 << 1;
  private static final int KAN_GATE = 1 << 2;
  private static final int KYUSHU_GATE = 1 << 3;
  private static final int TSUMO_GATE = 1 << 4;
  private static final int PASS_TYPE = DecisionFeatureCodec.actionType(Action.Type.PASS);
  private static final int RON_TYPE = DecisionFeatureCodec.actionType(Action.Type.RON_AGARI);
  private static final int TSUMO_TYPE = DecisionFeatureCodec.actionType(Action.Type.TSUMO_AGARI);
  private static final int KYUSHU_TYPE = DecisionFeatureCodec.actionType(Action.Type.KYUSHU_KYUHAI);
  private static final int RIICHI_TYPE = DecisionFeatureCodec.actionType(Action.Type.RIICHI_DAHAI);
  private static final int MELD_GROUP = DecisionFeatureCodec.actionGroup(Action.Group.MELD);
  private static final int KAN_GROUP = DecisionFeatureCodec.actionGroup(Action.Group.KAN);

  private final DecisionHostInputs inputs;
  private final DecisionTrainingTargets trainingTargets;
  private int size;

  DecisionHostBatch(int capacity, DecisionBucket bucket, boolean includeTrainingTargets) {
    this(
        new DecisionHostInputs(capacity, bucket),
        includeTrainingTargets ? new DecisionTrainingTargets(capacity, bucket) : null);
  }

  static DecisionHostBatch inference(int capacity, DecisionBucket bucket) {
    return new DecisionHostBatch(DecisionHostInputs.inference(capacity, bucket), null);
  }

  private DecisionHostBatch(DecisionHostInputs inputs, DecisionTrainingTargets trainingTargets) {
    this.inputs = inputs;
    this.trainingTargets = trainingTargets;
  }

  /**
   * {@link DecisionInputSchema#fingerprint()}に一致する1-行データ本体から復元する。
   *
   * @param bucket データ本体の行動・遷移容量区分
   * @param categoricalValues 共通形式のカテゴリ値連続バッファ
   * @param numericValues 共通形式の数値連続バッファ
   * @return 入力連続バッファを独立所有する推論バッチ
   */
  public static DecisionHostBatch fromEncodedRow(
      DecisionBucket bucket, short[] categoricalValues, float[] numericValues) {
    DecisionHostBatch batch =
        new DecisionHostBatch(
            DecisionHostInputs.fromEncoded(1, bucket, categoricalValues, numericValues), null);
    batch.commitActionRow(0);
    return batch;
  }

  /**
   * 復号済み共通形式の入力連続バッファの所有権を受け取り、複製せずに推論バッチを復元する。
   *
   * <p>呼出元は返却後に配列を読み書きしてはならない。永続化のための符号化・復号処理など、配列を新規確保して全要素を書き終えた境界だけで使用する。
   *
   * @param rowCount データ本体に含まれる行数
   * @param bucket データ本体の行動・遷移容量区分
   * @param categoricalSlab 所有権を渡す共通形式のカテゴリ値連続バッファ
   * @param numericSlab 所有権を渡す共通形式の数値連続バッファ
   * @return 渡された配列を直接所有する推論バッチ
   */
  public static DecisionHostBatch takeEncodedInferenceBatch(
      int rowCount, DecisionBucket bucket, short[] categoricalSlab, float[] numericSlab) {
    DecisionHostBatch batch =
        new DecisionHostBatch(
            DecisionHostInputs.takeEncoded(rowCount, bucket, categoricalSlab, numericSlab), null);
    for (int row = 0; row < rowCount; row++) {
      batch.commitActionRow(row);
    }
    return batch;
  }

  /**
   * 復号済み共通形式の入力と教師値連続バッファの所有権を受け取り、複製せずに学習バッチを復元する。
   *
   * <p>呼出元は返却後に配列を読み書きしてはならない。符号化・復号処理やバッチ生成処理が新規確保した配列だけを渡す。
   *
   * @param rowCount データ本体に含まれる行数
   * @param bucket データ本体の行動・遷移容量区分
   * @param encodedInputCategories 共通形式の入力カテゴリ値連続バッファ
   * @param encodedInputNumerics 共通形式の入力数値連続バッファ
   * @param encodedTargetCategories 共通形式の教師値カテゴリ値連続バッファ
   * @param encodedTargetNumerics 共通形式の教師値数値連続バッファ
   * @return 渡された入力と教師値配列を直接所有する学習バッチ
   */
  public static DecisionHostBatch takeEncodedTrainingBatch(
      int rowCount,
      DecisionBucket bucket,
      short[] encodedInputCategories,
      float[] encodedInputNumerics,
      int[] encodedTargetCategories,
      float[] encodedTargetNumerics) {
    DecisionHostBatch batch =
        new DecisionHostBatch(
            DecisionHostInputs.takeEncoded(
                rowCount, bucket, encodedInputCategories, encodedInputNumerics),
            DecisionTrainingTargets.takeEncoded(
                rowCount, bucket, encodedTargetCategories, encodedTargetNumerics));
    for (int row = 0; row < rowCount; row++) {
      batch.commitActionRow(row);
    }
    return batch;
  }

  /**
   * 記憶領域へ格納できる最大行数を返す。
   *
   * @return 行容量
   */
  public int capacity() {
    return inputs.capacity();
  }

  /**
   * 確定済みの有効行数を返す。
   *
   * @return 有効な行件数
   */
  public int size() {
    return size;
  }

  /**
   * 行動・遷移容量区分を返す。
   *
   * @return このバッチの容量区分
   */
  public DecisionBucket bucket() {
    return inputs.bucket();
  }

  /**
   * このバッチが所有する入力記憶領域を返す。
   *
   * @return 変更可能な標準形式のホスト側入力
   */
  public DecisionHostInputs inputs() {
    return inputs;
  }

  /** 指定行の重みを固定したGRP局境界事前予測を返す。 */
  public float[] boundaryRankPrior(int row) {
    requireActiveRow(row);
    return inputs.boundaryRankPrior(row);
  }

  /** 指定席の4順位周辺分布だけを返す。 */
  public float[] boundaryRankPrior(int row, int seat) {
    requireActiveRow(row);
    return inputs.boundaryRankPrior(row, seat);
  }

  /**
   * 学習教師値を所有するかを返す。
   *
   * @return 学習バッチなら {@code true}
   */
  public boolean hasTrainingTargets() {
    return trainingTargets != null;
  }

  /**
   * このバッチが所有する学習教師値を返す。
   *
   * @return 変更可能な共通形式の学習教師値
   * @throws IllegalStateException 推論専用バッチである場合
   */
  public DecisionTrainingTargets trainingTargets() {
    if (trainingTargets == null) {
      throw new IllegalStateException("inference batch has no training targets");
    }
    return trainingTargets;
  }

  /**
   * 指定行のパディング前の合法行動数を返す。
   *
   * @param row 有効行インデックス
   * @return 連続して格納された合法行動数
   */
  public int legalActionCount(int row) {
    int count = 0;
    while (count < bucket().legalActionCapacity()
        && inputs.actionCategory(row, count, DecisionInputSchema.ActionInt.ID)
            != DecisionInputSchema.PAD_ID) {
      count++;
    }
    return count;
  }

  /**
   * 指定合法行動候補の位置の安定行動 IDを返す。
   *
   * @param row 有効行インデックス
   * @param slot 合法行動集合内の枠
   * @return 復号済み行動 ID
   */
  public int legalActionId(int row, int slot) {
    return DecisionFeatureCodec.decodeActionId(
        inputs.actionCategory(row, slot, DecisionInputSchema.ActionInt.ID));
  }

  /**
   * 合法行動のカテゴリ値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int actionCategory(int row, int actionSlot, DecisionInputSchema.ActionInt field) {
    return inputs.actionCategory(row, actionSlot, field);
  }

  /**
   * 合法行動の数値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float actionNumeric(int row, int actionSlot, DecisionInputSchema.ActionFloat field) {
    return inputs.actionNumeric(row, actionSlot, field);
  }

  /**
   * 合法行動に属する遷移のカテゴリ値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int transitionCategory(
      int row, int actionSlot, int transitionSlot, DecisionInputSchema.ActionTransitionInt field) {
    return inputs.transitionCategory(row, actionSlot, transitionSlot, field);
  }

  /**
   * 合法行動に属する遷移の数値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float transitionNumeric(
      int row,
      int actionSlot,
      int transitionSlot,
      DecisionInputSchema.ActionTransitionFloat field) {
    return inputs.transitionNumeric(row, actionSlot, transitionSlot, field);
  }

  /**
   * 遷移行動後の状態の指定牌種特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param tileType 34牌種インデックス
   * @return 符号化済み遷移-牌値
   */
  public int transitionTile(int row, int actionSlot, int transitionSlot, int tileType) {
    return inputs.transitionTile(row, actionSlot, transitionSlot, tileType);
  }

  /**
   * 遷移の待ち枠に対応する牌種IDを読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の枠
   * @return 符号化済み待ち牌種ID
   */
  public int waitTile(int row, int actionSlot, int transitionSlot, int waitSlot) {
    return inputs.waitTile(row, actionSlot, transitionSlot, waitSlot);
  }

  /**
   * 指定待ち牌で成立可能な役特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の枠
   * @param feature 待ち牌の役特徴量インデックス
   * @return 符号化済み役特徴量値
   */
  public int waitYaku(int row, int actionSlot, int transitionSlot, int waitSlot, int feature) {
    return inputs.waitYaku(row, actionSlot, transitionSlot, waitSlot, feature);
  }

  /**
   * 指定待ち牌の得点特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @param transitionSlot 行動内の遷移候補の位置
   * @param waitSlot 待ち集合内の枠
   * @param field 得点特徴量種別
   * @return 符号化済み得点値
   */
  public float waitScore(
      int row,
      int actionSlot,
      int transitionSlot,
      int waitSlot,
      DecisionInputSchema.ActionTransitionWaitFloat field) {
    return inputs.waitScore(row, actionSlot, transitionSlot, waitSlot, field);
  }

  /**
   * 指定した合法行動に属するパディング前の遷移数を返す。
   *
   * @param row 有効行インデックス
   * @param actionSlot 合法行動候補の位置
   * @return 行動に属する有効遷移数
   */
  public int actionTransitionCount(int row, int actionSlot) {
    return encodedActionTransitionCount(row, actionSlot);
  }

  /**
   * 観測者の絶対席を返す。
   *
   * @param row 有効行インデックス
   * @return 0始まりの席インデックス
   */
  public int playerSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.PLAYER_SEAT) - 1;
  }

  /**
   * 現在手番プレイヤーの相対席を返す。
   *
   * @param row 有効行インデックス
   * @return 観測者基準の0始まり相対席
   */
  public int currentPlayerRelativeSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.CURRENT_PLAYER_RELATIVE_SEAT) - 1;
  }

  /**
   * 応答対象イベントを発生させたプレイヤーの相対席を返す。
   *
   * @param row 有効行インデックス
   * @return 観測者基準の0始まり相対席
   */
  public int sourcePlayerRelativeSeat(int row) {
    return roundCategory(row, DecisionInputSchema.RoundInt.SOURCE_PLAYER_RELATIVE_SEAT) - 1;
  }

  /**
   * 局単位のカテゴリ値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int roundCategory(int row, DecisionInputSchema.RoundInt field) {
    return inputs.roundCategory(row, field);
  }

  /**
   * 局単位の数値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float roundNumeric(int row, DecisionInputSchema.RoundFloat field) {
    return inputs.roundNumeric(row, field);
  }

  /**
   * 指定相対席のプレイヤーカテゴリ値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param relativeSeat 観測者基準の相対席
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int playerCategory(int row, int relativeSeat, DecisionInputSchema.PlayerInt field) {
    return inputs.playerCategory(row, relativeSeat, field);
  }

  /**
   * 指定相対席のプレイヤー数値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param relativeSeat 観測者基準の相対席
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float playerNumeric(int row, int relativeSeat, DecisionInputSchema.PlayerFloat field) {
    return inputs.playerNumeric(row, relativeSeat, field);
  }

  /**
   * 指定牌種のカテゴリ値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param tileType 34牌種インデックス
   * @param field 特徴量種別
   * @return 符号化済みカテゴリ ID
   */
  public int tileCategory(int row, int tileType, DecisionInputSchema.TileInt field) {
    return inputs.tileCategory(row, tileType, field);
  }

  /**
   * 指定牌種の数値特徴量を読む。
   *
   * @param row 有効行インデックス
   * @param tileType 34牌種インデックス
   * @param field 特徴量種別
   * @return 符号化済み数値
   */
  public float tileNumeric(int row, int tileType, DecisionInputSchema.TileFloat field) {
    return inputs.tileNumeric(row, tileType, field);
  }

  /**
   * 指定範囲を独立所有バッチへ複製する。
   *
   * @param fromInclusive 複製する先頭の有効行インデックス
   * @param count 複製する行数
   * @return 入力と教師値を独立所有するバッチ
   */
  public DecisionHostBatch copyRows(int fromInclusive, int count) {
    requireRange(fromInclusive, count);
    DecisionHostBatch copy = new DecisionHostBatch(count, bucket(), hasTrainingTargets());
    for (int row = 0; row < count; row++) {
      int sourceRow = fromInclusive + row;
      copy.copyNextRowFrom(this, sourceRow, row);
      copy.commitLikeSourceRow(this, sourceRow, row);
    }
    return copy;
  }

  /**
   * 所有連続バッファを複製せずに連続行を参照する。
   *
   * <p>ビュー自身が入力元バッチを強参照するため、そのまま非同期キューへ所有権を渡せる。入力元の内容はビューの寿命中に書き換えない。
   *
   * @param fromInclusive ビューの先頭となる有効行インデックス
   * @param count ビューに含める行数
   * @return 入力元記憶領域を共有する非所有ビュー
   */
  public RowSlice sliceRows(int fromInclusive, int count) {
    return new RowSlice(this, fromInclusive, count);
  }

  /**
   * 一行を独立所有バッチへ複製する。
   *
   * @param row 複製する有効行インデックス
   * @return 容量と有効行数が1のバッチ
   */
  public DecisionHostBatch copyRow(int row) {
    return copyRows(row, 1);
  }

  /**
   * 任意順の行を独立所有バッチへ集約する。
   *
   * @param rows 入力元内の有効行インデックス配列
   * @param count 配列先頭から使用するインデックス数
   * @return {@code rows}の順序を保持したバッチ
   */
  public DecisionHostBatch selectRows(int[] rows, int count) {
    if (rows == null || count < 1 || count > rows.length) {
      throw new IllegalArgumentException("invalid selected row count: " + count);
    }
    DecisionHostBatch selected = new DecisionHostBatch(count, bucket(), hasTrainingTargets());
    for (int destination = 0; destination < count; destination++) {
      int sourceRow = rows[destination];
      requireActiveRow(sourceRow);
      selected.copyNextRowFrom(this, sourceRow, destination);
      selected.commitLikeSourceRow(this, sourceRow, destination);
    }
    return selected;
  }

  /**
   * 同一容量区分・教師値種別のバッチ群を一つの連続した記憶領域へ連結する。
   *
   * @param parts 入力順に連結する非空バッチ群
   * @return 全行を独立所有する連結バッチ
   */
  public static DecisionHostBatch concatenate(List<DecisionHostBatch> parts) {
    if (parts == null || parts.isEmpty()) {
      throw new IllegalArgumentException("parts must not be empty");
    }
    DecisionHostBatch first = parts.getFirst();
    int rows = 0;
    for (DecisionHostBatch part : parts) {
      if (!part.bucket().equals(first.bucket())
          || part.hasTrainingTargets() != first.hasTrainingTargets()) {
        throw new IllegalArgumentException("all parts must have the same bucket and target kind");
      }
      rows = Math.addExact(rows, part.size());
    }
    DecisionHostBatch merged =
        new DecisionHostBatch(rows, first.bucket(), first.hasTrainingTargets());
    int destination = 0;
    for (DecisionHostBatch part : parts) {
      for (int row = 0; row < part.size(); row++) {
        merged.copyNextRowFrom(part, row, destination);
        merged.commitLikeSourceRow(part, row, destination++);
      }
    }
    return merged;
  }

  /**
   * 同一容量区分・教師値種別の連続行ビュー群を一つの連続した記憶領域へ連結する。
   *
   * <p>各ビューの参照範囲だけを入力順に一度複製する。非同期推論ワーカーが複数の論理バッチを一つの物理バッチへまとめる場合に、元バッチ全体や中間バッチを複製せず使う。
   *
   * @param parts 入力順に連結する非空ビュー群
   * @return 全ビューの行を独立所有する連結バッチ
   */
  public static DecisionHostBatch concatenateRows(List<RowSlice> parts) {
    if (parts == null || parts.isEmpty()) {
      throw new IllegalArgumentException("parts must not be empty");
    }
    RowSlice first = parts.getFirst();
    int rows = 0;
    for (RowSlice part : parts) {
      if (!part.bucket().equals(first.bucket())
          || part.hasTrainingTargets() != first.hasTrainingTargets()) {
        throw new IllegalArgumentException("all parts must have the same bucket and target kind");
      }
      rows = Math.addExact(rows, part.size());
    }
    DecisionHostBatch merged =
        new DecisionHostBatch(rows, first.bucket(), first.hasTrainingTargets());
    int destination = 0;
    for (RowSlice part : parts) {
      for (int row = 0; row < part.size(); row++) {
        int sourceRow = part.fromInclusive() + row;
        merged.copyNextRowFrom(part.source(), sourceRow, destination);
        merged.commitLikeSourceRow(part.source(), sourceRow, destination++);
      }
    }
    return merged;
  }

  /**
   * 指定行の方策損失重みを返す。
   *
   * @param row 有効行インデックス
   * @return 方策学習の重み
   */
  public float actorWeight(int row) {
    return trainingTargets().actorWeight(row);
  }

  /**
   * 指定行の方策損失重みを設定する。
   *
   * @param row 有効行インデックス
   * @param value 新しい方策学習の重み
   */
  public void setActorWeight(int row, float value) {
    trainingTargets().setActorWeight(row, value);
  }

  /**
   * 指定行の共通サンプル重みを返す。
   *
   * @param row 有効行インデックス
   * @return サンプル重み
   */
  public float sampleWeight(int row) {
    return trainingTargets().sampleWeight(row);
  }

  /**
   * 指定行の共通サンプル重みを設定する。
   *
   * @param row 有効行インデックス
   * @param value 新しいサンプル重み
   */
  public void setSampleWeight(int row, float value) {
    trainingTargets().setSampleWeight(row, value);
  }

  /**
   * 指定行の選択行動に対するアドバンテージを返す。
   *
   * @param row 有効行インデックス
   * @return アドバンテージ値
   */
  public float advantage(int row) {
    return trainingTargets().advantage(row);
  }

  /**
   * 確定済み行だけをデバイスへ送る場合の数値要素数を返す。
   *
   * @return 有効な行に対応する数値連続バッファ要素数
   */
  public int activeInputNumericElementCount() {
    return new DecisionInputLayout(size, bucket()).numericElementCount();
  }

  void commitActionRow(int row) {
    commitRow(row, true);
  }

  void commitStateOnlyRow(int row) {
    commitRow(row, false);
  }

  void copyNextRowFrom(DecisionHostBatch source, int sourceRow, int destinationRow) {
    source.requireActiveRow(sourceRow);
    requireNextDestinationRow(destinationRow);
    if (hasTrainingTargets() != source.hasTrainingTargets()) {
      throw new IllegalArgumentException("source target kind does not match destination");
    }
    inputs.copyTrustedRowFrom(source.inputs, sourceRow, destinationRow);
    if (hasTrainingTargets()) {
      trainingTargets.copyTrustedRowFrom(source.trainingTargets, sourceRow, destinationRow);
    }
  }

  void copyNextInputRowFrom(DecisionHostBatch source, int sourceRow, int destinationRow) {
    source.requireActiveRow(sourceRow);
    requireNextDestinationRow(destinationRow);
    inputs.copyTrustedRowFrom(source.inputs, sourceRow, destinationRow);
  }

  private void commitLikeSourceRow(DecisionHostBatch source, int sourceRow, int destinationRow) {
    if (source.legalActionCount(sourceRow) == 0) {
      commitStateOnlyRow(destinationRow);
    } else {
      commitActionRow(destinationRow);
    }
  }

  private void commitRow(int row, boolean requireLegalAction) {
    requireNextDestinationRow(row);
    int legalCount = 0;
    while (legalCount < bucket().legalActionCapacity()
        && inputs.actionCategory(row, legalCount, DecisionInputSchema.ActionInt.ID)
            != DecisionInputSchema.PAD_ID) {
      legalCount++;
    }
    if (requireLegalAction && legalCount < 1) {
      throw new IllegalStateException("action row has no legal action");
    }
    for (int slot = legalCount; slot < bucket().legalActionCapacity(); slot++) {
      if (inputs.actionCategory(row, slot, DecisionInputSchema.ActionInt.ID)
          != DecisionInputSchema.PAD_ID) {
        throw new IllegalStateException(
            "legal actions must be contiguous: row=" + row + " slot=" + slot);
      }
    }
    if (legalCount > 0) {
      boolean response = isResponseAction(row, 0);
      for (int slot = 1; slot < legalCount; slot++) {
        if (isResponseAction(row, slot) != response) {
          throw new IllegalStateException(
              "turn and response actions cannot share a row: row=" + row + " slot=" + slot);
        }
      }
      for (int actionSlot = 0; actionSlot < legalCount; actionSlot++) {
        requireContiguousTransitions(row, actionSlot);
      }
    }
    inputs.commitActionCount(row, legalCount);
    size++;
  }

  private void requireContiguousTransitions(int row, int actionSlot) {
    int transitionCount = encodedActionTransitionCount(row, actionSlot);
    if (transitionCount < 1) {
      throw new IllegalStateException(
          "legal action requires a transition: row=" + row + " action=" + actionSlot);
    }
    for (int transition = transitionCount;
        transition < inputs.transitionCapacity(row, actionSlot);
        transition++) {
      if (transitionPresent(row, actionSlot, transition)) {
        throw new IllegalStateException(
            "transitions must be contiguous: row="
                + row
                + " action="
                + actionSlot
                + " transition="
                + transition);
      }
    }
  }

  private int encodedActionTransitionCount(int row, int actionSlot) {
    int count = 0;
    while (count < inputs.transitionCapacity(row, actionSlot)
        && transitionPresent(row, actionSlot, count)) {
      count++;
    }
    return count;
  }

  private boolean transitionPresent(int row, int actionSlot, int transitionSlot) {
    return inputs.transitionCategory(
            row, actionSlot, transitionSlot, DecisionInputSchema.ActionTransitionInt.PRESENT)
        != DecisionInputSchema.PAD_ID;
  }

  private boolean isResponseAction(int row, int actionSlot) {
    return DecisionFeatureCodec.isResponseActionGroup(
        inputs.actionCategory(row, actionSlot, DecisionInputSchema.ActionInt.GROUP));
  }

  private boolean isRiichiGateAction(int row, int actionSlot) {
    return inputs.actionCategory(row, actionSlot, DecisionInputSchema.ActionInt.TYPE) == RIICHI_TYPE
        && inputs.actionRoute(row, actionSlot, DecisionInputSchema.ActionRoute.DAMA_SLOT)
            != DecisionInputSchema.PAD_ID
        && inputs.actionRoute(row, actionSlot, DecisionInputSchema.ActionRoute.RIICHI_SLOT)
            != DecisionInputSchema.PAD_ID;
  }

  /** 推論準備では、確定後に行動・遷移の構造を変更しないcompact入力の検証済み行動数を使う。 */
  private int inferenceActionCount(int row) {
    return inputs.isDense() ? legalActionCount(row) : inputs.compactActionCount(row);
  }

  private int policyGateMask(int row, int actions) {
    boolean pass = false;
    boolean meld = false;
    boolean ron = false;
    boolean continuation = false;
    boolean kan = false;
    boolean kyushu = false;
    boolean tsumo = false;
    for (int action = 0; action < actions; action++) {
      int type = inputs.actionCategory(row, action, DecisionInputSchema.ActionInt.TYPE);
      int group = inputs.actionCategory(row, action, DecisionInputSchema.ActionInt.GROUP);
      pass |= type == PASS_TYPE;
      meld |= group == MELD_GROUP;
      ron |= type == RON_TYPE;
      continuation |=
          inputs.actionRoute(
                  row, action, DecisionInputSchema.ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE)
              != DecisionInputSchema.PAD_ID;
      kan |= group == KAN_GROUP;
      kyushu |= type == KYUSHU_TYPE;
      tsumo |= type == TSUMO_TYPE;
    }
    int mask = 0;
    if (pass && meld) {
      mask |= CALL_GATE;
    }
    if (ron && (pass || meld)) {
      mask |= RON_GATE;
    }
    if (continuation && kan) {
      mask |= KAN_GATE;
    }
    if (kyushu && (continuation || kan)) {
      mask |= KYUSHU_GATE;
    }
    if (tsumo && (continuation || kan || kyushu)) {
      mask |= TSUMO_GATE;
    }
    return mask;
  }

  private void requireActiveRow(int row) {
    if (row < 0 || row >= size) {
      throw new IndexOutOfBoundsException("row=" + row + " size=" + size);
    }
  }

  private void requireRange(int fromInclusive, int count) {
    if (fromInclusive < 0 || count < 1 || fromInclusive + count > size) {
      throw new IndexOutOfBoundsException(
          "row range outside batch: from=" + fromInclusive + " count=" + count + " size=" + size);
    }
  }

  private void requireNextDestinationRow(int row) {
    if (row != size || row >= capacity()) {
      throw new IllegalStateException(
          "rows must be committed once in order: row=" + row + " size=" + size);
    }
  }

  /**
   * {@link DecisionHostBatch} 内の連続行を表す非所有ビュー。
   *
   * @param source 記憶領域を所有する元バッチ
   * @param fromInclusive ビューの先頭行
   * @param size ビューに含む行数
   */
  public record RowSlice(DecisionHostBatch source, int fromInclusive, int size) {

    /** 入力元内で有効な非空範囲であることを検証する。 */
    public RowSlice {
      source.requireRange(fromInclusive, size);
    }

    /**
     * 入力元の行動・遷移容量区分を返す。
     *
     * @return 入力元と共有する容量区分
     */
    public DecisionBucket bucket() {
      return source.bucket();
    }

    /**
     * 入力元が学習教師値を所有するかを返す。
     *
     * @return 学習教師値を転送できるなら {@code true}
     */
    public boolean hasTrainingTargets() {
      return source.hasTrainingTargets();
    }

    /**
     * ビュー内の全方策入力が占めるバイト数を返す。
     *
     * @return カテゴリ値と数値入力の合計バイト数
     */
    public long inputByteCount() {
      return new DecisionInputLayout(size, bucket()).inputByteCount();
    }

    /**
     * 方策候補を伴わない価値推論で送る状態を表す二つの連続バッファのバイト数を返す。
     *
     * @return 状態カテゴリ値と状態数値の合計バイト数
     */
    public long stateByteCount() {
      return (long) size
          * (DecisionInputSchema.STATE_INT_COUNT * (long) Short.BYTES
              + (DecisionInputSchema.STATE_FLOAT_COUNT + DecisionBoundaryContext.INPUT_SIZE)
                  * (long) Float.BYTES);
    }

    /**
     * 価値学習用に転送する状態入力、教師値、サンプル重みの合計バイト数を返す。
     *
     * @return 価値学習用転送の合計バイト数
     */
    public long valueTransferByteCount() {
      return stateByteCount()
          + (long) size * (DecisionTrainingTargets.VALUE_TARGET_SIZE + 1L) * Float.BYTES;
    }

    /**
     * ビュー内のサンプル重み総和を返す。
     *
     * @return サンプル重みの総和
     */
    public double sampleWeightMass() {
      return source.trainingTargets().sampleWeightMass(fromInclusive, size);
    }

    /**
     * ビュー内のカテゴリ値入力要素数を返す。
     *
     * @return カテゴリ値 要素数
     */
    public int inputCategoricalElementCount() {
      return new DecisionInputLayout(size, bucket()).categoricalElementCount();
    }

    /**
     * ビュー内の数値入力要素数を返す。
     *
     * @return 数値 要素数
     */
    public int inputNumericElementCount() {
      return new DecisionInputLayout(size, bucket()).numericElementCount();
    }

    int playerMemoryPresentCount() {
      return source.inputs.playerMemoryPresentCount(fromInclusive, size);
    }

    void copyPlayerMemoryPresentIndicesTo(int destinationRowBase, IntBuffer destination) {
      source.inputs.copyPlayerMemoryPresentIndicesTo(
          fromInclusive, size, destinationRowBase, destination);
    }

    int transitionPresentCount() {
      int count = 0;
      for (int row = fromInclusive; row < fromInclusive + size; row++) {
        if (!source.inputs.isDense()) {
          count += source.inputs.compactTransitionCount(row);
          continue;
        }
        int actions = source.legalActionCount(row);
        for (int action = 0; action < actions; action++) {
          count += source.actionTransitionCount(row, action);
        }
      }
      return count;
    }

    PolicyExecutionLayout policyExecutionLayout() {
      int riichiActions = 0;
      int callRows = 0;
      int ronRows = 0;
      int kanRows = 0;
      int kyushuRows = 0;
      int tsumoRows = 0;
      for (int localRow = 0; localRow < size; localRow++) {
        int sourceRow = fromInclusive + localRow;
        int actions = source.inferenceActionCount(sourceRow);
        for (int action = 0; action < actions; action++) {
          if (source.isRiichiGateAction(sourceRow, action)) {
            riichiActions++;
          }
        }
        int mask = source.policyGateMask(sourceRow, actions);
        callRows += (mask & CALL_GATE) != 0 ? 1 : 0;
        ronRows += (mask & RON_GATE) != 0 ? 1 : 0;
        kanRows += (mask & KAN_GATE) != 0 ? 1 : 0;
        kyushuRows += (mask & KYUSHU_GATE) != 0 ? 1 : 0;
        tsumoRows += (mask & TSUMO_GATE) != 0 ? 1 : 0;
      }
      return new PolicyExecutionLayout(
          riichiActions, callRows, ronRows, kanRows, kyushuRows, tsumoRows);
    }

    InferenceIndexLayout inferenceIndexLayout(boolean includePolicyExecution) {
      int playerMemoryCount = playerMemoryPresentCount();
      int transitionCount = transitionPresentCount();
      PolicyExecutionLayout policyLayout =
          includePolicyExecution ? policyExecutionLayout() : PolicyExecutionLayout.empty();
      return InferenceIndexLayout.create(
          playerMemoryCount, transitionCount, policyLayout, includePolicyExecution);
    }

    DecisionInferenceInputLayout inferenceInputLayout(InferenceIndexLayout indexLayout) {
      return new DecisionInferenceInputLayout(size, bucket(), indexLayout.transitionCount());
    }

    void copyInferenceIndicesTo(IntBuffer destination, InferenceIndexLayout layout) {
      int start = destination.position();
      if (destination.remaining() < layout.totalCount()) {
        throw new IllegalArgumentException(
            "inference index destination is too small: remaining="
                + destination.remaining()
                + " required="
                + layout.totalCount());
      }
      copyPlayerMemoryPresentIndicesTo(
          0, segment(destination, layout.playerMemoryOffset(), layout.playerMemoryCount()));
      copyTransitionPresentIndicesTo(
          0, segment(destination, layout.transitionOffset(), layout.transitionCount()));
      if (layout.includesPolicyExecution()) {
        copyPolicyExecutionIndicesTo(
            segment(destination, layout.policyExecutionOffset(), layout.policyExecutionCount()),
            layout.policyExecutionLayout());
      }
      destination.position(Math.addExact(start, layout.totalCount()));
    }

    private static IntBuffer segment(IntBuffer destination, int offset, int count) {
      int start = Math.addExact(destination.position(), offset);
      IntBuffer segment = destination.duplicate();
      segment.position(start);
      segment.limit(Math.addExact(start, count));
      return segment.slice();
    }

    void copyPolicyExecutionIndicesTo(IntBuffer destination, PolicyExecutionLayout layout) {
      int riichiPosition = 0;
      int callPosition = layout.callOffset();
      int ronPosition = layout.ronOffset();
      int kanPosition = layout.kanOffset();
      int kyushuPosition = layout.kyushuOffset();
      int tsumoPosition = layout.tsumoOffset();
      int actionCapacity = bucket().legalActionCapacity();
      for (int localRow = 0; localRow < size; localRow++) {
        int sourceRow = fromInclusive + localRow;
        int actions = source.inferenceActionCount(sourceRow);
        for (int action = 0; action < actions; action++) {
          if (source.isRiichiGateAction(sourceRow, action)) {
            destination.put(riichiPosition++, localRow * actionCapacity + action);
          }
        }
        int mask = source.policyGateMask(sourceRow, actions);
        if ((mask & CALL_GATE) != 0) {
          destination.put(callPosition++, localRow);
        }
        if ((mask & RON_GATE) != 0) {
          destination.put(ronPosition++, localRow);
        }
        if ((mask & KAN_GATE) != 0) {
          destination.put(kanPosition++, localRow);
        }
        if ((mask & KYUSHU_GATE) != 0) {
          destination.put(kyushuPosition++, localRow);
        }
        if ((mask & TSUMO_GATE) != 0) {
          destination.put(tsumoPosition++, localRow);
        }
      }
      destination.position(layout.totalCount());
    }

    void copyTransitionPresentIndicesTo(int destinationRowBase, IntBuffer destination) {
      int actionCapacity = bucket().legalActionCapacity();
      int transitionCapacity = bucket().actionTransitionCapacity();
      int rowsPerBatch = Math.multiplyExact(actionCapacity, transitionCapacity);
      for (int localRow = 0; localRow < size; localRow++) {
        int sourceRow = fromInclusive + localRow;
        int actions = source.inferenceActionCount(sourceRow);
        for (int action = 0; action < actions; action++) {
          int actionOffset =
              (destinationRowBase + localRow) * rowsPerBatch + action * transitionCapacity;
          int transitions =
              source.inputs.isDense()
                  ? source.actionTransitionCount(sourceRow, action)
                  : source.inputs.transitionCapacity(sourceRow, action);
          for (int transition = 0; transition < transitions; transition++) {
            destination.put(actionOffset + transition);
          }
        }
      }
    }

    int targetCategoricalElementCount() {
      return source.trainingTargets().categoricalElementCount(size);
    }

    int targetNumericElementCount() {
      return source.trainingTargets().numericElementCount(size);
    }

    /**
     * このビューのカテゴリ値テンソルを標準形式の連続バッファ順で転送先へ連続書き込みする。
     *
     * @param destination 必要要素数以上の残量を持つ転送先バッファ
     */
    public void copyInputCategoriesTo(ShortBuffer destination) {
      source.inputs.copyRowsTo(
          DecisionInputLayout.Slab.CATEGORICAL, fromInclusive, size, destination, null);
    }

    /**
     * このビューの数値テンソルを標準形式の連続バッファ順で転送先へ連続書き込みする。
     *
     * @param destination 必要要素数以上の残量を持つ転送先バッファ
     */
    public void copyInputNumericsTo(FloatBuffer destination) {
      source.inputs.copyRowsTo(
          DecisionInputLayout.Slab.NUMERIC, fromInclusive, size, null, destination);
    }

    void copyInferenceInputsTo(
        ShortBuffer categoricalDestination,
        FloatBuffer numericDestination,
        DecisionInferenceInputLayout layout) {
      int categoricalStart = categoricalDestination.position();
      int numericStart = numericDestination.position();
      for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
        if (tensor.isTransition()) {
          continue;
        }
        DecisionInferenceInputLayout.Region region = layout.region(tensor);
        ShortBuffer categorical =
            tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                ? categoricalDestination.position(
                    Math.addExact(categoricalStart, region.slabOffset()))
                : null;
        FloatBuffer numeric =
            tensor.slab() == DecisionInputLayout.Slab.NUMERIC
                ? numericDestination.position(Math.addExact(numericStart, region.slabOffset()))
                : null;
        source.inputs.copyTensorRowsTo(tensor, fromInclusive, size, categorical, numeric);
      }

      int transitionCapacity = bucket().actionTransitionCapacity();
      boolean compact = !source.inputs.isDense();
      for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
        if (!tensor.isTransition()) {
          continue;
        }
        int stride = tensor.elementsPerTransition();
        DecisionInferenceInputLayout.Region region = layout.region(tensor);
        ShortBuffer categorical =
            tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                ? categoricalDestination.position(
                    Math.addExact(categoricalStart, region.slabOffset()))
                : null;
        FloatBuffer numeric =
            tensor.slab() == DecisionInputLayout.Slab.NUMERIC
                ? numericDestination.position(Math.addExact(numericStart, region.slabOffset()))
                : null;
        for (int localRow = 0; localRow < size; localRow++) {
          int sourceRow = fromInclusive + localRow;
          if (compact || transitionCapacity == 1) {
            int transitions =
                compact
                    ? source.inputs.compactTransitionCount(sourceRow)
                    : source.legalActionCount(sourceRow);
            source.inputs.copyTransitionRowsTo(
                tensor, sourceRow, 0, transitions, stride, categorical, numeric);
            continue;
          }
          int actions = source.legalActionCount(sourceRow);
          for (int action = 0; action < actions; action++) {
            int transitions = source.actionTransitionCount(sourceRow, action);
            source.inputs.copyTransitionRowsTo(
                tensor, sourceRow, action, transitions, stride, categorical, numeric);
          }
        }
      }
      categoricalDestination.position(
          Math.addExact(categoricalStart, layout.categoricalElementCount()));
      numericDestination.position(Math.addExact(numericStart, layout.numericElementCount()));
    }

    void copyStateCategoriesTo(ShortBuffer destination) {
      source.inputs.copyTensorRowsTo(
          DecisionInputLayout.Tensor.STATE_CATEGORIES, fromInclusive, size, destination, null);
    }

    void copyStateNumericsTo(FloatBuffer destination) {
      source.inputs.copyTensorRowsTo(
          DecisionInputLayout.Tensor.STATE_NUMERICS, fromInclusive, size, null, destination);
    }

    void copyBoundaryContextTo(FloatBuffer destination) {
      source.inputs.copyTensorRowsTo(
          DecisionInputLayout.Tensor.BOUNDARY_CONTEXT, fromInclusive, size, null, destination);
    }

    void copyTargetCategoriesTo(IntBuffer destination) {
      source.trainingTargets().copyCategoriesTo(fromInclusive, size, destination);
    }

    void copyTargetNumericsTo(FloatBuffer destination) {
      source.trainingTargets().copyNumericsTo(fromInclusive, size, destination);
    }

    void copyValueTargetsTo(FloatBuffer destination) {
      source.trainingTargets().copyValueTargetsTo(fromInclusive, size, destination);
    }

    /** 固定形状の通常転送が必要な境界でだけ実遷移記憶領域を展開します。 */
    DecisionHostBatch materializeDense() {
      return source.inputs.isDense() ? materialize() : source.copyRows(fromInclusive, size);
    }

    /**
     * ビューを通常の所有バッチとして返す。
     *
     * <p>ビューが全入力元を覆う場合だけ入力元自体を返し、それ以外は独立所有バッチへ複製する。
     *
     * @return ビューと同じ行を持つ所有バッチ
     */
    public DecisionHostBatch materialize() {
      return fromInclusive == 0 && size == source.size
          ? source
          : source.copyRows(fromInclusive, size);
    }
  }

  /**
   * 同じ容量区分に属する複数の行ビューを、連続バッファを複製せず一つの推論バッチとして扱います。
   *
   * <p>非同期推論パイプラインはこの型をCPU側の軽量記述情報として保持し、GPU 枠が空いた時点で各入力元から枠専用のページ固定の
   * バッファへ直接指定位置の要素を集めます。元バッチの所有権は移さないため、呼び出し側は推論完了まで入力元を保持します。
   */
  public static final class RowBatch {

    private final RowSlice[] slices;
    private final DecisionBucket bucket;
    private final boolean hasTrainingTargets;
    private final int size;

    private RowBatch(RowSlice[] slices) {
      if (slices.length == 0) {
        throw new IllegalArgumentException("row batch must not be empty");
      }
      bucket = slices[0].bucket();
      hasTrainingTargets = slices[0].hasTrainingTargets();
      int rows = 0;
      for (RowSlice slice : slices) {
        if (!bucket.equals(slice.bucket()) || hasTrainingTargets != slice.hasTrainingTargets()) {
          throw new IllegalArgumentException("row batch requires one bucket and one target layout");
        }
        rows = Math.addExact(rows, slice.size());
      }
      this.slices = slices;
      size = rows;
    }

    /** 一つの行ビューをコピーせず記述情報へ包みます。 */
    public static RowBatch of(RowSlice slice) {
      return new RowBatch(new RowSlice[] {slice});
    }

    /** 行ビュー参照だけを一度配列化し、同じ容量区分の記述情報を作ります。 */
    public static RowBatch of(List<RowSlice> slices) {
      return new RowBatch(slices.toArray(RowSlice[]::new));
    }

    /**
     * 呼び出し側が所有権を渡す配列を、そのまま記述情報として保持します。
     *
     * <p>返却後に配列要素を変更してはいけません。準備済みバッチ構築時の中間Listを避けるための境界です。
     */
    public static RowBatch takeOwnership(RowSlice[] slices) {
      return new RowBatch(slices);
    }

    /** 合計行数を返します。 */
    public int size() {
      return size;
    }

    /** 全ビューが共有する型付き容量区分を返します。 */
    public DecisionBucket bucket() {
      return bucket;
    }

    /** 学習教師値を含む記述情報なら{@code true}を返します。 */
    public boolean hasTrainingTargets() {
      return hasTrainingTargets;
    }

    /** 元行ビュー数を返します。 */
    public int sliceCount() {
      return slices.length;
    }

    /** 指定位置の非所有行ビューを返します。 */
    public RowSlice slice(int index) {
      return slices[index];
    }

    /** 全入力元に存在するプレイヤーごとの履歴表現トークン数を返します。 */
    int playerMemoryPresentCount() {
      int count = 0;
      for (RowSlice slice : slices) {
        count = Math.addExact(count, slice.playerMemoryPresentCount());
      }
      return count;
    }

    /** プレイヤーごとの履歴表現インデックスだけを集約バッチ上の位置へ直して書き込みます。 */
    void copyPlayerMemoryPresentIndicesTo(IntBuffer destination) {
      int globalRow = 0;
      for (RowSlice slice : slices) {
        slice.copyPlayerMemoryPresentIndicesTo(globalRow, destination);
        globalRow += slice.size;
      }
    }

    /** 推論インデックス連続バッファの集約配置を返します。 */
    InferenceIndexLayout inferenceIndexLayout(boolean includePolicyExecution) {
      int playerMemoryCount = 0;
      int transitionCount = 0;
      int riichiActions = 0;
      int callRows = 0;
      int ronRows = 0;
      int kanRows = 0;
      int kyushuRows = 0;
      int tsumoRows = 0;
      for (RowSlice slice : slices) {
        playerMemoryCount = Math.addExact(playerMemoryCount, slice.playerMemoryPresentCount());
        transitionCount = Math.addExact(transitionCount, slice.transitionPresentCount());
        if (includePolicyExecution) {
          PolicyExecutionLayout layout = slice.policyExecutionLayout();
          riichiActions = Math.addExact(riichiActions, layout.riichiActionCount());
          callRows = Math.addExact(callRows, layout.callRowCount());
          ronRows = Math.addExact(ronRows, layout.ronRowCount());
          kanRows = Math.addExact(kanRows, layout.kanRowCount());
          kyushuRows = Math.addExact(kyushuRows, layout.kyushuRowCount());
          tsumoRows = Math.addExact(tsumoRows, layout.tsumoRowCount());
        }
      }
      PolicyExecutionLayout policyLayout =
          includePolicyExecution
              ? new PolicyExecutionLayout(
                  riichiActions, callRows, ronRows, kanRows, kyushuRows, tsumoRows)
              : PolicyExecutionLayout.empty();
      return InferenceIndexLayout.create(
          playerMemoryCount, transitionCount, policyLayout, includePolicyExecution);
    }

    /** 有効要素のみの推論入力の集約配置を返します。 */
    DecisionInferenceInputLayout inferenceInputLayout(InferenceIndexLayout indexLayout) {
      return new DecisionInferenceInputLayout(size, bucket, indexLayout.transitionCount());
    }

    /** 全入力元から有効要素のみの推論入力を、集約配置上の最終位置へ直接書き込みます。 */
    void copyInferenceInputsTo(
        ShortBuffer categoricalDestination,
        FloatBuffer numericDestination,
        DecisionInferenceInputLayout layout) {
      int categoricalStart = categoricalDestination.position();
      int numericStart = numericDestination.position();
      for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
        if (tensor.isTransition()) {
          continue;
        }
        DecisionInferenceInputLayout.Region region = layout.region(tensor);
        ShortBuffer categorical =
            tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                ? categoricalDestination.position(
                    Math.addExact(categoricalStart, region.slabOffset()))
                : null;
        FloatBuffer numeric =
            tensor.slab() == DecisionInputLayout.Slab.NUMERIC
                ? numericDestination.position(Math.addExact(numericStart, region.slabOffset()))
                : null;
        for (RowSlice slice : slices) {
          slice.source.inputs.copyTensorRowsTo(
              tensor, slice.fromInclusive, slice.size, categorical, numeric);
        }
      }

      int transitionCapacity = bucket.actionTransitionCapacity();
      for (DecisionInputLayout.Tensor tensor : DecisionInputLayout.tensors()) {
        if (!tensor.isTransition()) {
          continue;
        }
        int stride = tensor.elementsPerTransition();
        DecisionInferenceInputLayout.Region region = layout.region(tensor);
        ShortBuffer categorical =
            tensor.slab() == DecisionInputLayout.Slab.CATEGORICAL
                ? categoricalDestination.position(
                    Math.addExact(categoricalStart, region.slabOffset()))
                : null;
        FloatBuffer numeric =
            tensor.slab() == DecisionInputLayout.Slab.NUMERIC
                ? numericDestination.position(Math.addExact(numericStart, region.slabOffset()))
                : null;
        for (RowSlice slice : slices) {
          boolean compact = !slice.source.inputs.isDense();
          for (int localRow = 0; localRow < slice.size; localRow++) {
            int sourceRow = slice.fromInclusive + localRow;
            if (compact || transitionCapacity == 1) {
              int transitions =
                  compact
                      ? slice.source.inputs.compactTransitionCount(sourceRow)
                      : slice.source.legalActionCount(sourceRow);
              slice.source.inputs.copyTransitionRowsTo(
                  tensor, sourceRow, 0, transitions, stride, categorical, numeric);
              continue;
            }
            int actions = slice.source.legalActionCount(sourceRow);
            for (int action = 0; action < actions; action++) {
              int transitions = slice.source.actionTransitionCount(sourceRow, action);
              slice.source.inputs.copyTransitionRowsTo(
                  tensor, sourceRow, action, transitions, stride, categorical, numeric);
            }
          }
        }
      }
      categoricalDestination.position(
          Math.addExact(categoricalStart, layout.categoricalElementCount()));
      numericDestination.position(Math.addExact(numericStart, layout.numericElementCount()));
    }

    /** 全入力元のプレイヤーごとの履歴表現、遷移、方策実行インデックスを集約順に書き込みます。 */
    void copyInferenceIndicesTo(IntBuffer destination, InferenceIndexLayout layout) {
      int start = destination.position();
      copyPlayerMemoryPresentIndicesTo(
          RowSlice.segment(destination, layout.playerMemoryOffset(), layout.playerMemoryCount()));
      IntBuffer transitions =
          RowSlice.segment(destination, layout.transitionOffset(), layout.transitionCount());
      int globalRow = 0;
      for (RowSlice slice : slices) {
        slice.copyTransitionPresentIndicesTo(globalRow, transitions);
        globalRow += slice.size;
      }
      if (layout.includesPolicyExecution()) {
        copyPolicyExecutionIndicesTo(
            RowSlice.segment(
                destination, layout.policyExecutionOffset(), layout.policyExecutionCount()),
            layout.policyExecutionLayout());
      }
      destination.position(Math.addExact(start, layout.totalCount()));
    }

    private void copyPolicyExecutionIndicesTo(IntBuffer destination, PolicyExecutionLayout layout) {
      int riichiPosition = 0;
      int callPosition = layout.callOffset();
      int ronPosition = layout.ronOffset();
      int kanPosition = layout.kanOffset();
      int kyushuPosition = layout.kyushuOffset();
      int tsumoPosition = layout.tsumoOffset();
      int globalRow = 0;
      int actionCapacity = bucket.legalActionCapacity();
      for (RowSlice slice : slices) {
        for (int localRow = 0; localRow < slice.size; localRow++, globalRow++) {
          int sourceRow = slice.fromInclusive + localRow;
          int actions = slice.source.inferenceActionCount(sourceRow);
          for (int action = 0; action < actions; action++) {
            if (slice.source.isRiichiGateAction(sourceRow, action)) {
              destination.put(riichiPosition++, globalRow * actionCapacity + action);
            }
          }
          int mask = slice.source.policyGateMask(sourceRow, actions);
          if ((mask & CALL_GATE) != 0) {
            destination.put(callPosition++, globalRow);
          }
          if ((mask & RON_GATE) != 0) {
            destination.put(ronPosition++, globalRow);
          }
          if ((mask & KAN_GATE) != 0) {
            destination.put(kanPosition++, globalRow);
          }
          if ((mask & KYUSHU_GATE) != 0) {
            destination.put(kyushuPosition++, globalRow);
          }
          if ((mask & TSUMO_GATE) != 0) {
            destination.put(tsumoPosition++, globalRow);
          }
        }
      }
      destination.position(layout.totalCount());
    }

    /** 全入力元の状態カテゴリを行順に書き込みます。 */
    void copyStateCategoriesTo(ShortBuffer destination) {
      for (RowSlice slice : slices) {
        slice.copyStateCategoriesTo(destination);
      }
    }

    /** 全入力元の状態数値を行順に書き込みます。 */
    void copyStateNumericsTo(FloatBuffer destination) {
      for (RowSlice slice : slices) {
        slice.copyStateNumericsTo(destination);
      }
    }

    /** 全入力元の境界コンテキストを行順に書き込みます。 */
    void copyBoundaryContextTo(FloatBuffer destination) {
      for (RowSlice slice : slices) {
        slice.copyBoundaryContextTo(destination);
      }
    }

    /** 状態のみの転送のバイト数を返します。 */
    public long stateByteCount() {
      return (long) size
          * (DecisionInputSchema.STATE_INT_COUNT * (long) Short.BYTES
              + (DecisionInputSchema.STATE_FLOAT_COUNT + DecisionBoundaryContext.INPUT_SIZE)
                  * (long) Float.BYTES);
    }

    /** 有効要素のみの推論入力のバイト数を返します。 */
    public long inputByteCount() {
      InferenceIndexLayout indices = inferenceIndexLayout(true);
      return inferenceInputLayout(indices).inputByteCount()
          + (long) indices.totalCount() * Integer.BYTES;
    }
  }

  record PolicyExecutionLayout(
      int riichiActionCount,
      int callRowCount,
      int ronRowCount,
      int kanRowCount,
      int kyushuRowCount,
      int tsumoRowCount) {

    static PolicyExecutionLayout empty() {
      return new PolicyExecutionLayout(0, 0, 0, 0, 0, 0);
    }

    int callOffset() {
      return riichiActionCount;
    }

    int ronOffset() {
      return callOffset() + callRowCount;
    }

    int kanOffset() {
      return ronOffset() + ronRowCount;
    }

    int kyushuOffset() {
      return kanOffset() + kanRowCount;
    }

    int tsumoOffset() {
      return kyushuOffset() + kyushuRowCount;
    }

    int totalCount() {
      return tsumoOffset() + tsumoRowCount;
    }

    DecisionInferenceDeviceBatch.PolicyExecutionIndices bind(NDArray packedIndices) {
      return new DecisionInferenceDeviceBatch.PolicyExecutionIndices(
          packedIndices,
          riichiActionCount,
          callRowCount,
          ronRowCount,
          kanRowCount,
          kyushuRowCount,
          tsumoRowCount);
    }
  }

  /** 推論用INT32 連続バッファ内の三領域を、ホスト側の要素オフセットと件数で固定する。 */
  record InferenceIndexLayout(
      int playerMemoryOffset,
      int playerMemoryCount,
      int transitionOffset,
      int transitionCount,
      int policyExecutionOffset,
      int policyExecutionCount,
      PolicyExecutionLayout policyExecutionLayout,
      boolean includesPolicyExecution) {

    InferenceIndexLayout {
      if (playerMemoryOffset != 0
          || playerMemoryCount < 0
          || transitionOffset != playerMemoryCount
          || transitionCount < 0
          || policyExecutionOffset != Math.addExact(transitionOffset, transitionCount)
          || policyExecutionCount < 0
          || policyExecutionLayout == null
          || (includesPolicyExecution
              ? policyExecutionCount != policyExecutionLayout.totalCount()
              : policyExecutionCount != 0 || policyExecutionLayout.totalCount() != 0)) {
        throw new IllegalArgumentException("invalid packed inference index layout");
      }
    }

    static InferenceIndexLayout create(
        int playerMemoryCount,
        int transitionCount,
        PolicyExecutionLayout policyExecutionLayout,
        boolean includePolicyExecution) {
      int transitionOffset = playerMemoryCount;
      int policyExecutionOffset = Math.addExact(transitionOffset, transitionCount);
      int policyExecutionCount = includePolicyExecution ? policyExecutionLayout.totalCount() : 0;
      return new InferenceIndexLayout(
          0,
          playerMemoryCount,
          transitionOffset,
          transitionCount,
          policyExecutionOffset,
          policyExecutionCount,
          policyExecutionLayout,
          includePolicyExecution);
    }

    int totalCount() {
      return Math.addExact(policyExecutionOffset, policyExecutionCount);
    }
  }
}
