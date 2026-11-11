package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.util.PairList;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.MeldFloat;
import com.epsilon.classic.input.ClassicFields.MeldInt;
import com.epsilon.classic.input.ClassicFields.PlayerFloat;
import com.epsilon.classic.input.ClassicFields.PlayerInt;
import com.epsilon.classic.input.ClassicFields.RiverFloat;
import com.epsilon.classic.input.ClassicFields.RiverInt;
import com.epsilon.classic.input.ClassicFields.RoundFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFloat;
import com.epsilon.classic.input.ClassicFields.TileInt;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicSchema.StateToken;
import com.epsilon.classic.input.Tensor;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 状態の行を、方策と価値が共通で使う表現（{@link Memory}）に変換する。
 *
 * <p>局・4 席・34 種類の牌・河・面子のフィールドを、それぞれトークンに射影する。牌・河・面子のトークンには牌の種類の埋め込みを足す。
 * 河は打った牌を、面子はいちばん小さい牌と鳴いた牌の平均を使う。空いているトークンは 0 にする。牌のトークンは、牌の関係を使う
 * Attention（{@link TileRelationEncoder}）で文脈化する。各席の要約・河・面子は、席の記憶（{@link PlayerMemoryEncoder}）で、その席の
 * トークンの中だけで文脈化する。Major はさらに公開の履歴（{@link PublicHistoryEncoder}）で要約を更新する。最後に、局・自分の手・
 * 4 席の要約を戦略の文脈（{@link StrategicContextEncoder}）でまとめて読む。
 *
 * <p>Major は席の記憶を 1 つの射影でキーとバリューにし、公開の履歴と方策ヘッドが同じキーとバリューを読む。Nano では方策ヘッドが自分で
 * 射影する。
 */
final class StateEncoder extends AbstractBlock {

  /** 局・4 席・34 種類の牌の、いつもあるトークンの数。 */
  private static final int FIXED_TOKENS = 1 + ClassicSchema.SEATS + Tiles.KINDS;

  private static final int RIVERS = ClassicSchema.SEATS * ClassicSchema.RIVER_TOKENS;
  private static final int MELDS = ClassicSchema.SEATS * ClassicSchema.MELD_TOKENS;

  /** 局・席・牌・河・面子のトークンの数（151）。 */
  static final int ENTITIES = FIXED_TOKENS + RIVERS + MELDS;

  private final ClassicSchema schema;
  private final boolean history;
  private final int hidden;
  private final int context;
  private final Section round;
  private final Section players;
  private final Section tiles;
  private final Section rivers;
  private final Section melds;
  private final int riverTile;
  private final int riverPresent;
  private final int meldBase;
  private final int meldCalled;
  private final int meldPresent;

  private final IdEmbedding stateFeatureEmbedding;
  private final IdEmbedding canonicalTileEmbedding;
  private final Linear roundFeatureProjection;
  private final Linear playerFeatureProjection;
  private final Linear tileFeatureProjection;
  private final TileRelationEncoder tileRelationEncoder;
  private final Linear riverEventProjection;
  private final Linear meldFeatureProjection;
  private final LayerNorm entityTokenLayerNorm;
  private final PlayerMemoryEncoder playerMemoryEncoder;
  private final Linear playerMemoryKeyValueProjection;
  private final PublicHistoryEncoder publicHistoryEncoder;
  private final StrategicContextEncoder strategicContextEncoder;

  /** 状態の整数のフィールドの要素ごとに、埋め込みの表でその要素の行が始まる位置 [1, 要素の数]（INT32）。 */
  private NDArray stateOffsets;

  StateEncoder(Widths widths) {
    Architecture architecture = widths.architecture();
    schema = ClassicSchema.of(architecture);
    history = architecture.publicHistory();
    hidden = widths.hidden();
    context = widths.context();
    round = new Section(schema, StateToken.ROUND, schema.layout(RoundInt.class), schema.layout(RoundFloat.class));
    players = new Section(schema, StateToken.PLAYER, schema.layout(PlayerInt.class), schema.layout(PlayerFloat.class));
    tiles = new Section(schema, StateToken.TILE, schema.layout(TileInt.class), schema.layout(TileFloat.class));
    rivers = new Section(schema, StateToken.RIVER, schema.layout(RiverInt.class), schema.layout(RiverFloat.class));
    melds = new Section(schema, StateToken.MELD, schema.layout(MeldInt.class), schema.layout(MeldFloat.class));
    riverTile = schema.layout(RiverInt.class).offset(RiverInt.TILE);
    riverPresent = schema.layout(RiverInt.class).offset(RiverInt.PRESENT);
    meldBase = schema.layout(MeldInt.class).offset(MeldInt.BASE_TILE);
    meldCalled = schema.layout(MeldInt.class).offset(MeldInt.CALLED_TILE);
    meldPresent = schema.layout(MeldInt.class).offset(MeldInt.PRESENT);

    stateFeatureEmbedding =
        addChildBlock(
            "stateFeatureEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(schema.stateDictionary())
                .setEmbeddingSize(Widths.CATEGORY)
                .build());
    canonicalTileEmbedding =
        addChildBlock(
            "canonicalTileEmbedding",
            new IdEmbedding.Builder().setDictionarySize(Tiles.KINDS + 1).setEmbeddingSize(hidden).build());
    roundFeatureProjection = addChildBlock("roundFeatureProjection", Linear.builder().setUnits(hidden).build());
    playerFeatureProjection = addChildBlock("playerFeatureProjection", Linear.builder().setUnits(hidden).build());
    tileFeatureProjection = addChildBlock("tileFeatureProjection", Linear.builder().setUnits(hidden).build());
    tileRelationEncoder = addChildBlock("tileRelationEncoder", new TileRelationEncoder(widths));
    riverEventProjection = addChildBlock("riverEventProjection", Linear.builder().setUnits(hidden).build());
    meldFeatureProjection = addChildBlock("meldFeatureProjection", Linear.builder().setUnits(hidden).build());
    entityTokenLayerNorm = addChildBlock("entityTokenLayerNorm", LayerNorm.builder().axis(2).build());
    playerMemoryEncoder = addChildBlock("playerMemoryEncoder", new PlayerMemoryEncoder(widths));
    if (history) {
      playerMemoryKeyValueProjection =
          addChildBlock(
              "playerMemoryKeyValueProjection", Linear.builder().setUnits(2L * context).optBias(false).build());
      publicHistoryEncoder = addChildBlock("publicHistoryEncoder", new PublicHistoryEncoder(widths));
    } else {
      playerMemoryKeyValueProjection = null;
      publicHistoryEncoder = null;
    }
    strategicContextEncoder = addChildBlock("strategicContextEncoder", new StrategicContextEncoder(widths));
  }

  /**
   * 状態の行を表現に変換する。
   *
   * @param batch 行のバッチ
   * @return 方策と価値が共通で使う表現
   */
  Memory encode(Pass pass, ClassicBatch batch) {
    try (ArrayScope scope = new ArrayScope()) {
      int rows = batch.rows();
      NDArray categories = batch.tensor(Tensor.STATE_CATEGORIES);
      NDArray values = history ? categories.get(":, 0:{}", schema.historyStart()) : categories;
      NDArray embedded = NDArrays.embeddingWithOffsets(values, stateOffsets, pass.table(stateFeatureEmbedding));
      NDArray numerics = pass.toCompute(batch.tensor(Tensor.STATE_NUMERICS));

      NDArray roundToken = project(pass, roundFeatureProjection, round, embedded, numerics, rows).reshape(rows, hidden);
      NDArray playerTokens = project(pass, playerFeatureProjection, players, embedded, numerics, rows);
      NDArray identities = pass.own(pass.table(canonicalTileEmbedding).get("1:"));
      NDArray tileTokens = project(pass, tileFeatureProjection, tiles, embedded, numerics, rows).add(identities);
      tileTokens = tileRelationEncoder.encode(pass, tileTokens);

      NDArray riverValues = rivers.values(values, rows);
      NDArray riverMask = riverValues.get("..., {}", riverPresent).toType(pass.compute(), false);
      NDArray riverTokens =
          ArrayScope.of(
              () ->
                  project(pass, riverEventProjection, rivers, embedded, numerics, rows)
                      .add(canonical(pass, riverValues.get("..., {}", riverTile)))
                      .mul(riverMask.expandDims(2)));

      NDArray meldValues = melds.values(values, rows);
      NDArray meldMask = meldValues.get("..., {}", meldPresent).toType(pass.compute(), false);
      NDArray base = meldValues.get("..., {}", meldBase);
      NDArray called = meldValues.get("..., {}", meldCalled);
      NDArray identityCount = present(pass, base).add(present(pass, called)).maximum(1f).expandDims(2);
      NDArray meldTokens =
          ArrayScope.of(
              () ->
                  project(pass, meldFeatureProjection, melds, embedded, numerics, rows)
                      .add(canonical(pass, base).add(canonical(pass, called)).div(identityCount))
                      .mul(meldMask.expandDims(2)));

      NDArray normalizedRound = pass.norm(entityTokenLayerNorm, roundToken);
      NDArray normalizedTiles = pass.norm(entityTokenLayerNorm, tileTokens);
      NDManager manager = batch.manager();
      NDArray playerMask =
          NDArrays.concat(
              new NDList(
                  manager.ones(new Shape(rows, ClassicSchema.SEATS, 1), pass.compute()),
                  riverMask.reshape(rows, ClassicSchema.SEATS, ClassicSchema.RIVER_TOKENS),
                  meldMask.reshape(rows, ClassicSchema.SEATS, ClassicSchema.MELD_TOKENS)),
              2);
      NDArray memoryTokens =
          NDArrays.concat(
              new NDList(
                  playerTokens.reshape(rows, ClassicSchema.SEATS, 1, hidden),
                  riverTokens.reshape(rows, ClassicSchema.SEATS, ClassicSchema.RIVER_TOKENS, hidden),
                  meldTokens.reshape(rows, ClassicSchema.SEATS, ClassicSchema.MELD_TOKENS, hidden)),
              2);
      NDArray memory = playerMemoryEncoder.encode(pass, memoryTokens, playerMask, entityTokenLayerNorm);
      NDList sections = memory.split(new long[] {1, 1 + ClassicSchema.RIVER_TOKENS}, 2);
      NDArray summaries = sections.get(0).reshape(rows, ClassicSchema.SEATS, hidden);
      NDArray publicRivers = sections.get(1).reshape(rows, RIVERS, hidden);
      NDArray publicMelds = sections.get(2).reshape(rows, MELDS, hidden);

      NDArray publicKeyValues = null;
      if (history) {
        publicKeyValues =
            pass.linear(playerMemoryKeyValueProjection, NDArrays.concat(new NDList(publicRivers, publicMelds), 1));
        NDArray publicMask = NDArrays.concat(new NDList(riverMask, meldMask), 1);
        NDArray codes =
            categories
                .get(":, {}:", schema.historyStart())
                .reshape(rows, ClassicSchema.SEATS, ClassicSchema.HISTORY_TOKENS, ClassicSchema.HISTORY_WORDS);
        summaries = publicHistoryEncoder.encode(pass, summaries, publicKeyValues, publicMask, codes);
      }
      StrategicContextEncoder.Context strategic =
          strategicContextEncoder.encode(pass, normalizedRound, normalizedTiles, summaries);

      NDArray entities =
          NDArrays.concat(
              new NDList(
                  strategic.round().reshape(rows, 1, hidden),
                  strategic.summaries(),
                  normalizedTiles,
                  publicRivers,
                  publicMelds),
              1);
      NDArray entityMask =
          NDArrays.concat(
              new NDList(manager.ones(new Shape(rows, FIXED_TOKENS), pass.compute()), riverMask, meldMask), 1);
      NDArray seatMemory;
      if (history) {
        NDList publicSections = publicKeyValues.split(new long[] {RIVERS}, 1);
        seatMemory =
            NDArrays.concat(
                new NDList(
                    pass.linear(playerMemoryKeyValueProjection, strategic.summaries())
                        .reshape(rows, ClassicSchema.SEATS, 1, 2L * context),
                    publicSections.get(0).reshape(rows, ClassicSchema.SEATS, ClassicSchema.RIVER_TOKENS, 2L * context),
                    publicSections.get(1).reshape(rows, ClassicSchema.SEATS, ClassicSchema.MELD_TOKENS, 2L * context)),
                2);
      } else {
        seatMemory =
            NDArrays.concat(
                new NDList(
                    strategic.summaries().reshape(rows, ClassicSchema.SEATS, 1, hidden),
                    sections.get(1),
                    sections.get(2)),
                2);
      }
      return new Memory(
          scope.keep(strategic.round()),
          scope.keep(entities),
          scope.keep(entityMask),
          scope.keep(normalizedTiles),
          scope.keep(seatMemory),
          scope.keep(playerMask));
    }
  }

  /** 1 種類のトークンについて、整数のフィールドの埋め込みと実数のフィールドを並べて射影する。 */
  private static NDArray project(
      Pass pass, Linear projection, Section section, NDArray embedded, NDArray numerics, int rows) {
    try (ArrayScope scope = new ArrayScope()) {
      NDArray categorical =
          embedded
              .get(":, {}:{}, :", section.categoricalStart, section.categoricalEnd())
              .reshape(rows, section.count, (long) section.categoricalStride * Widths.CATEGORY);
      NDArray numeric =
          numerics
              .get(":, {}:{}", section.numericStart, section.numericEnd())
              .reshape(rows, section.count, section.numericStride);
      return scope.keep(pass.linear(projection, categorical.concat(numeric, 2)));
    }
  }

  /** 値（牌の種類 ＋ 1）の埋め込みを返す。値が 0（パディング）の所は 0 にする。 */
  private NDArray canonical(Pass pass, NDArray storedTiles) {
    NDArray embedded = pass.embed(canonicalTileEmbedding, storedTiles.toType(DataType.INT32, false));
    return embedded.mul(present(pass, storedTiles).expandDims(storedTiles.getShape().dimension()));
  }

  /** 値が 0 でない所を 1、0 の所を 0 にする（計算の型）。 */
  private static NDArray present(Pass pass, NDArray stored) {
    return stored.neq(0).toType(pass.compute(), false);
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    int[] offsets = schema.stateOffsets();
    stateOffsets = manager.create(offsets).reshape(1, offsets.length);
    stateFeatureEmbedding.initialize(manager, dataType, new Shape(-1, offsets.length));
    canonicalTileEmbedding.initialize(manager, dataType, new Shape(-1, Tiles.KINDS));
    roundFeatureProjection.initialize(manager, dataType, round.inputShape());
    playerFeatureProjection.initialize(manager, dataType, players.inputShape());
    tileFeatureProjection.initialize(manager, dataType, tiles.inputShape());
    tileRelationEncoder.initialize(manager, dataType, new Shape(-1, Tiles.KINDS, hidden));
    riverEventProjection.initialize(manager, dataType, rivers.inputShape());
    meldFeatureProjection.initialize(manager, dataType, melds.inputShape());
    entityTokenLayerNorm.initialize(manager, dataType, new Shape(-1, ENTITIES, hidden));
    Shape memory = new Shape(-1, ClassicSchema.SEATS, PlayerMemoryEncoder.TOKENS, hidden);
    playerMemoryEncoder.initialize(
        manager, dataType, memory, new Shape(-1, ClassicSchema.SEATS, PlayerMemoryEncoder.TOKENS));
    if (history) {
      playerMemoryKeyValueProjection.initialize(manager, dataType, new Shape(-1, hidden));
      publicHistoryEncoder.initialize(manager, dataType, new Shape(-1, ClassicSchema.SEATS, hidden));
    }
    strategicContextEncoder.initialize(
        manager, dataType, new Shape(-1, hidden), new Shape(-1, Tiles.KINDS, hidden), memory);
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use encode");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, hidden), new Shape(-1, ENTITIES, hidden), new Shape(-1, ENTITIES)};
  }

  /** 状態の行のうち、1 種類のトークンが使う範囲（整数のフィールドと実数のフィールド）。 */
  private static final class Section {

    private final int count;
    private final int categoricalStart;
    private final int categoricalStride;
    private final int numericStart;
    private final int numericStride;

    Section(ClassicSchema schema, StateToken token, FieldLayout<?> categorical, FieldLayout<?> numeric) {
      count = token.count();
      categoricalStart = schema.stateStart(token, Tensor.Slab.CATEGORICAL);
      categoricalStride = categorical.stride();
      numericStart = schema.stateStart(token, Tensor.Slab.NUMERIC);
      numericStride = numeric.stride();
    }

    int categoricalEnd() {
      return categoricalStart + count * categoricalStride;
    }

    int numericEnd() {
      return numericStart + count * numericStride;
    }

    /** 整数のフィールドの値 [行, トークン, フィールド]。 */
    NDArray values(NDArray categories, int rows) {
      return categories.get(":, {}:{}", categoricalStart, categoricalEnd()).reshape(rows, count, categoricalStride);
    }

    /** 射影の入力の形。幅は、整数のフィールドの埋め込みと実数のフィールドを並べた分。 */
    Shape inputShape() {
      return new Shape(-1, (long) categoricalStride * Widths.CATEGORY + numericStride);
    }
  }
}
