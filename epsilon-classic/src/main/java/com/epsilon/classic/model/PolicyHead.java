package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.AbstractBlock;
import ai.djl.nn.Activation;
import ai.djl.nn.core.Linear;
import ai.djl.nn.norm.LayerNorm;
import ai.djl.nn.transformer.IdEmbedding;
import ai.djl.training.ParameterStore;
import ai.djl.training.initializer.ConstantInitializer;
import ai.djl.util.PairList;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicFields.ActionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicFields.ActionTransitionWaitFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFlag;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicVocabulary.ActionGroup;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.ClassicVocabulary.WaitWinType;
import com.epsilon.classic.input.Tensor;
import com.epsilon.classic.model.PointProjection.Projection;
import com.epsilon.mahjong.rules.RankUtility;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.tile.Tiles;

/**
 * 局面・合法手・合法手の後の遷移を採点して、方策のグラフ（{@link PolicyGraph}）に渡すスコアを作る。
 *
 * <p>どの合法手も 1 つ以上の遷移を持つ。チー・ポンは鳴いた後の打牌ごとに 1 つ、ほかの手は 1 つ。遷移の表現は、遷移をクエリにして
 * 34 種類の牌を読む Attention（遷移の Attention）で作る。キーとバリューは、牌の表現に 2 つの差を足したもの。1 つは、遷移の後の手牌で
 * その牌がどうなっているか（牌のコード）による差。もう 1 つは待ちの牌に足す差で、Nano は待ちのロン・ツモの役と点から、Major は
 * 点数の計算の結果から作る。Major では、牌の表現に 4 席を合わせた文脈の射影も足す。チー・ポンの遷移は、打牌の採点から作る重み
 * （勾配は止める）で 1 つにまとめる。
 *
 * <p>34 種類の牌は、牌をクエリにして各席の記憶（要約・河・面子）を読み、席ごとの文脈と 4 席を合わせた文脈を持つ。打牌の遷移は、
 * 切る牌の 4 席を合わせた文脈を読む。応答の手は、直前のイベントの牌について、その席の文脈を読む。手番の槓は、槓にする牌の 4 席を
 * 合わせた文脈を読む。
 *
 * <p>二択（ロン・鳴き・ツモ・九種九牌・槓・立直）の文脈は、グラフで下にあるノードの表現を、そのノードの条件つきの分布で重み付けて作る。
 * この重みは勾配を止めるので、上のノードの損失が下のロジットを直接動かすことはない。
 */
final class PolicyHead extends AbstractBlock {

  private static final int TRANSITION_KEY = 8;
  private static final int TRANSITION_KEYS = Widths.HEADS * TRANSITION_KEY;
  private static final int YAKU_EMBEDDING = 16;
  private static final int WIN_TYPES = WaitWinType.values().length;
  private static final int YAKU_BITS = ClassicSchema.YAKU_CHUNKS * ClassicSchema.YAKU_CHUNK_BITS;
  private static final int WAITS = ClassicSchema.MAX_WAITS;
  private static final int ALTERNATIVES = Alternative.values().length;

  /** 鳴きの種類の分岐（チー・ポン・大明槓）と、槓の種類の分岐（暗槓・加槓）の番号。 */
  private static final int[] MELD_ALTERNATIVES = ordinals(Alternative.CHI, Alternative.PON, Alternative.DAIMINKAN);

  private static final int[] KAN_ALTERNATIVES = ordinals(Alternative.ANKAN, Alternative.KAKAN);

  private final boolean major;
  private final int hidden;
  private final int context;
  private final ActionColumns columns;
  private final int transitionStride;
  private final int transitionNumerics;
  private final int transitionPresent;
  private final int discardActionId;
  private final int discardTile;
  private final int eventPlayer;
  private final int eventTile;
  private final int waitFactStride;
  private final int waitFactValid;
  private final int waitScoreStride;
  private final int[] typeFields;
  private final long[] typeSets;
  private final int[] groupFields;
  private final long[] groupSets;
  private final int[] actionOffsetValues;
  private final int[] transitionOffsetValues;

  private final IdEmbedding actionFeatureEmbedding;
  private final IdEmbedding transitionTileRelationEmbedding;
  private final Linear transitionWaitYakuProjection;
  private final PointProjection pointProjection;
  private final PointOutcomeEncoder pointOutcomeEncoder;
  private final IdEmbedding policyNodeEmbedding;
  private final IdEmbedding alternativeEmbedding;
  private final IdEmbedding alternativeOffset;
  private final Linear actionFeatureProjection;
  private final Linear transitionFeatureProjection;
  private final Linear transitionTileKeyValueProjection;
  private final Linear transitionPublicContextProjection;
  private final Linear transitionTileQueryProjection;
  private final Linear transitionWaitKeyValueProjection;
  private final Linear playerTileQueryProjection;
  private final Linear playerMemoryKeyValueProjection;
  private final PlayerTileMixer playerTileContextMixer;
  private final FeatureFusion transitionFusion;
  private final LayerNorm transitionLayerNorm;
  private final DiscardScorer discardCandidateScorer;
  private final FeatureFusion candidateFusion;
  private final LayerNorm candidateLayerNorm;
  private final Linear actionCandidateHidden;
  private final Linear actionCandidateScoreHead;
  private final Linear alternativeHidden;
  private final Linear alternativeScoreHead;
  private final BinaryGate ronGate;
  private final BinaryGate callGate;
  private final BinaryGate tsumoGate;
  private final BinaryGate kyushuGate;
  private final BinaryGate kanGate;
  private final BinaryGate riichiGate;

  /** 合法手の整数の列ごとに、埋め込みの表でその列の行が始まる位置 [1, 1, 列]（INT32）。 */
  private NDArray actionOffsets;

  /** 遷移の整数の列ごとに、埋め込みの表でその列の行が始まる位置 [1, 列]（INT32）。 */
  private NDArray transitionOffsets;

  /** 分岐ごとのノードの番号 [分岐]（INT64、Nano だけ）。 */
  private NDArray alternativeNodes;

  /** 役を 14 ビットずつ詰めた値をビットに分けるときに割る、ビットごとの 2 の冪 [14]（FP32、Nano だけ）。 */
  private NDArray yakuBitDivisors;

  PolicyHead(Widths widths, Variant variant, RankUtility utility) {
    Architecture architecture = widths.architecture();
    ClassicSchema schema = ClassicSchema.of(architecture);
    major = architecture == Architecture.MAJOR;
    hidden = widths.hidden();
    context = widths.context();
    columns = ActionColumns.of(schema);
    FieldLayout<ActionTransitionInt> transition = schema.layout(ActionTransitionInt.class);
    transitionStride = transition.stride();
    transitionNumerics = schema.layout(ActionTransitionFloat.class).stride();
    transitionPresent = transition.offset(ActionTransitionInt.PRESENT);
    discardActionId = transition.offset(ActionTransitionInt.DISCARD_ACTION_ID);
    discardTile = transition.offset(ActionTransitionInt.DISCARD_TILE);
    FieldLayout<RoundInt> round = schema.layout(RoundInt.class);
    eventPlayer = round.offset(RoundInt.EVENT_PLAYER_RELATIVE_SEAT);
    eventTile = round.offset(RoundInt.EVENT_TILE);
    FieldLayout<WinFact> facts = schema.layout(WinFact.class);
    waitFactStride = facts.stride();
    waitFactValid = facts.offset(WinFact.VALID);
    waitScoreStride = schema.layout(ActionTransitionWaitFloat.class).stride();
    ActionType[] types = ActionType.values();
    typeFields = new int[types.length];
    typeSets = new long[types.length];
    for (ActionType type : types) {
      typeFields[type.ordinal()] = columns.type();
      typeSets[type.ordinal()] = 1L << ActionColumns.stored(type);
    }
    long responses = 0;
    for (ActionGroup group : ActionGroup.values()) {
      responses |= group.response() ? 1L << (group.ordinal() + 1) : 0;
    }
    groupFields = new int[] {columns.group(), columns.group()};
    groupSets = new long[] {responses, 1L << (ActionGroup.KAN.ordinal() + 1)};
    actionOffsetValues = schema.actionOffsets();
    transitionOffsetValues = schema.transitionOffsets();

    actionFeatureEmbedding =
        addChildBlock(
            "actionFeatureEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(schema.actionDictionary())
                .setEmbeddingSize(Widths.CATEGORY)
                .build());
    transitionTileRelationEmbedding =
        addChildBlock(
            "transitionTileRelationEmbedding",
            new IdEmbedding.Builder()
                .setDictionarySize(TileFlag.CODES)
                .setEmbeddingSize(TRANSITION_KEYS + context)
                .build());
    transitionWaitYakuProjection =
        major
            ? null
            : addChildBlock(
                "transitionWaitYakuProjection",
                Linear.builder().setUnits(YAKU_EMBEDDING).optBias(false).build());
    pointProjection = major ? addChildBlock("pointProjection", new PointProjection(variant, utility)) : null;
    pointOutcomeEncoder = major ? addChildBlock("pointOutcomeEncoder", new PointOutcomeEncoder(context)) : null;
    policyNodeEmbedding =
        major ? null : addChildBlock("policyNodeEmbedding", table(PolicyNode.values().length, hidden));
    alternativeEmbedding = major ? null : addChildBlock("alternativeEmbedding", table(ALTERNATIVES, hidden));
    alternativeOffset = major ? addChildBlock("alternativeOffset", table(ALTERNATIVES, hidden)) : null;
    actionFeatureProjection = addChildBlock("actionFeatureProjection", Linear.builder().setUnits(hidden).build());
    transitionFeatureProjection =
        addChildBlock("transitionFeatureProjection", Linear.builder().setUnits(hidden).build());
    transitionTileKeyValueProjection =
        addChildBlock(
            "transitionTileKeyValueProjection", Linear.builder().setUnits(TRANSITION_KEYS + context).build());
    if (major) {
      Linear publicContext = Linear.builder().setUnits(TRANSITION_KEYS + context).build();
      publicContext.getDirectParameters().get("weight").setInitializer(new ConstantInitializer(0f));
      publicContext.getDirectParameters().get("bias").setInitializer(new ConstantInitializer(0f));
      transitionPublicContextProjection = addChildBlock("transitionPublicContextProjection", publicContext);
    } else {
      transitionPublicContextProjection = null;
    }
    transitionTileQueryProjection =
        addChildBlock("transitionTileQueryProjection", Linear.builder().setUnits(TRANSITION_KEYS).build());
    transitionWaitKeyValueProjection =
        addChildBlock(
            "transitionWaitKeyValueProjection", Linear.builder().setUnits(TRANSITION_KEYS + context).build());
    playerTileQueryProjection =
        addChildBlock("playerTileQueryProjection", Linear.builder().setUnits(context).optBias(false).build());
    playerMemoryKeyValueProjection =
        major
            ? null
            : addChildBlock(
                "playerMemoryKeyValueProjection", Linear.builder().setUnits(2L * context).optBias(false).build());
    playerTileContextMixer = addChildBlock("playerTileContextMixer", new PlayerTileMixer(context));
    transitionFusion =
        addChildBlock("transitionFusion", new FeatureFusion(hidden, hidden, context, hidden, context, hidden));
    transitionLayerNorm = addChildBlock("transitionLayerNorm", LayerNorm.builder().axis(3).build());
    discardCandidateScorer = addChildBlock("discardCandidateScorer", new DiscardScorer(hidden));
    candidateFusion =
        addChildBlock("candidateFusion", new FeatureFusion(hidden, hidden, hidden, hidden, context, hidden));
    candidateLayerNorm = addChildBlock("candidateLayerNorm", LayerNorm.builder().axis(2).build());
    actionCandidateHidden = addChildBlock("actionCandidateHidden", Linear.builder().setUnits(hidden).build());
    actionCandidateScoreHead =
        addChildBlock("actionCandidateScoreHead", Linear.builder().setUnits(1).optBias(false).build());
    alternativeHidden = addChildBlock("alternativeHidden", Linear.builder().setUnits(hidden).build());
    alternativeScoreHead = addChildBlock("alternativeScoreHead", Linear.builder().setUnits(1).build());
    int pointFeatures = major ? PointProjection.GATE_FEATURES : 0;
    ronGate = addChildBlock("ronGate", new BinaryGate(hidden, pointFeatures));
    callGate = addChildBlock("callGate", new BinaryGate(hidden, 0));
    tsumoGate = addChildBlock("tsumoGate", new BinaryGate(hidden, pointFeatures));
    kyushuGate = addChildBlock("kyushuGate", new BinaryGate(hidden, 0));
    kanGate = addChildBlock("kanGate", new BinaryGate(hidden, 0));
    riichiGate = addChildBlock("riichiGate", new BinaryGate(hidden, 0));
  }

  /**
   * 方策のスコアを作る。
   *
   * @param batch 行のバッチ
   * @param memory 状態のエンコーダーの出力（価値と共有）
   * @param state 方策に使う局面の表現 [行, 幅]
   * @return 正規化していないスコア（計算の型）
   */
  PolicyScores score(Pass pass, ClassicBatch batch, Memory memory, NDArray state) {
    try (ArrayScope scope = new ArrayScope()) {
      int rows = batch.rows();
      int actions = batch.bucket().actions();
      NDArray actionCategories = batch.tensor(Tensor.ACTION_CATEGORIES);
      NDArray legal = actionCategories.get("..., {}", columns.id()).neq(0).toType(pass.compute(), false);
      NDArray typeMasks = NDArrays.categoricalMasks(actionCategories, legal, typeFields, typeSets);
      NDArray groupMasks = NDArrays.categoricalMasks(actionCategories, legal, groupFields, groupSets);

      NDArray keyValues = major ? memory.players() : pass.linear(playerMemoryKeyValueProjection, memory.players());
      NDArray byPlayer =
          NDArrays.groupedPackedScaledDotProductAttention(
              pass.linear(playerTileQueryProjection, memory.tiles()),
              keyValues,
              memory.playerMask(),
              Widths.HEADS,
              1.0 / Math.sqrt((double) context / Widths.HEADS));
      NDArray allPlayers = playerTileContextMixer.mix(pass, byPlayer);
      Transitions transitions = transitions(pass, batch, memory, state, allPlayers);

      NDArray actionFeatures =
          NDArrays.embeddingWithOffsets(actionCategories, actionOffsets, pass.table(actionFeatureEmbedding))
              .reshape(rows, actions, (long) columns.stride() * Widths.CATEGORY);
      NDArray pointGates = null;
      NDArray extra;
      if (major) {
        Projection points = pointProjection.actions(pass, batch);
        extra = pointOutcomeEncoder.encode(pass, pass.toCompute(points.features()));
        pointGates = pass.toCompute(points.gates());
      } else {
        extra = pass.toCompute(batch.tensor(Tensor.ACTION_NUMERICS));
      }
      NDArray actionEmbeddings = Pass.silu(pass.linear(actionFeatureProjection, actionFeatures.concat(extra, 2)));
      NDArray primaryTiles = actionCategories.get("..., {}", columns.primaryTile());
      NDArray stateCategories = batch.tensor(Tensor.STATE_CATEGORIES);
      NDArray eventContext =
          NDArrays.paddedBatchGather(
              byPlayer,
              stateCategories.get(":, {}", eventPlayer).reshape(rows, 1),
              stateCategories.get(":, {}", eventTile).reshape(rows, 1));
      NDArray rootContexts =
          eventContext
              .mul(groupMasks.get("..., 0").expandDims(2))
              .add(NDArrays.paddedBatchGather(allPlayers, primaryTiles).mul(groupMasks.get("..., 1").expandDims(2)));
      NDArray stateContext = state.reshape(rows, 1, hidden);
      NDArray candidates =
          ArrayScope.of(
              () ->
                  pass.norm(
                          candidateLayerNorm,
                          Pass.silu(
                              candidateFusion.fuse(
                                  pass,
                                  actionEmbeddings,
                                  NDArrays.paddedBatchGather(memory.tiles(), primaryTiles),
                                  transitions.aggregated(),
                                  rootContexts,
                                  stateContext)))
                      .mul(legal.expandDims(2)));
      NDArray rootScores =
          pass.linear(actionCandidateScoreHead, Pass.silu(pass.linear(actionCandidateHidden, candidates)))
              .reshape(rows, actions);

      DiscardChoices choices =
          DiscardChoices.of(
              batch.tensor(Tensor.ACTION_ROUTES),
              type(typeMasks, ActionType.DAHAI),
              type(typeMasks, ActionType.RIICHI_DAHAI),
              columns);
      NDArray discard = choices.discardMask();
      NDArray shared = NDArrays.paddedBatchGather(transitions.scores().get(":, :, 0"), choices.representativeSlots());
      NDArray actionScores = rootScores.mul(discard.neg().add(1f)).add(shared.mul(discard)).mul(legal);
      NDArray damaContexts = NDArrays.paddedBatchGather(candidates, choices.damaSlots());
      NDArray riichiContexts = NDArrays.paddedBatchGather(candidates, choices.riichiSlots());
      NDArray riichiScores =
          riichiGate
              .scoreEach(pass, stateContext, damaContexts, riichiContexts)
              .reshape(rows, actions)
              .mul(choices.damaPresent())
              .mul(choices.riichiPresent())
              .mul(choices.riichiMask());

      NDArray alternatives =
          alternatives(
              pass,
              state,
              candidates,
              typeMasks,
              actionScores,
              riichiScores,
              choices,
              damaContexts,
              riichiContexts,
              pointGates);
      return new PolicyScores(scope.keep(alternatives), scope.keep(actionScores), scope.keep(riichiScores));
    }
  }

  /** 遷移の表現を作り、打牌の遷移を採点する。実在する遷移だけを詰めて計算し、結果を [行, 合法手, 遷移] の並びに戻す。 */
  private Transitions transitions(Pass pass, ClassicBatch batch, Memory memory, NDArray state, NDArray allPlayers) {
    try (ArrayScope scope = new ArrayScope()) {
      int rows = batch.rows();
      Bucket bucket = batch.bucket();
      int actions = bucket.actions();
      int per = bucket.transitions();
      long slots = (long) rows * actions * per;
      NDArray categories = batch.tensor(Tensor.TRANSITION_CATEGORIES).reshape(slots, transitionStride);
      NDArray presentValues = categories.get(":, {}", transitionPresent);
      NDArray indices = presentValues.gt(0).nonzero().reshape(-1);
      long present = indices.size();
      NDArray batchIds = indices.floorDivide((long) actions * per);
      NDArray compact = NDArrays.gatherRows(categories, indices);
      NDArray numerics =
          pass.toCompute(
              NDArrays.gatherRows(
                  batch.tensor(Tensor.TRANSITION_NUMERICS).reshape(slots, transitionNumerics), indices));
      NDArray tileCodes =
          NDArrays.gatherRows(batch.tensor(Tensor.TRANSITION_TILES).reshape(slots, Tiles.KINDS), indices);
      NDArray waitIds = NDArrays.gatherRows(batch.tensor(Tensor.WAIT_TILE_IDS).reshape(slots, WAITS), indices);
      NDArray features =
          Pass.silu(
              pass.linear(
                  transitionFeatureProjection,
                  NDArrays.embeddingWithOffsets(compact, transitionOffsets, pass.table(actionFeatureEmbedding))
                      .reshape(present, (long) transitionStride * Widths.CATEGORY)
                      .concat(numerics, 1)));

      NDArray tileKeyValues = pass.linear(transitionTileKeyValueProjection, memory.tiles());
      NDArray waitFeatures;
      if (major) {
        tileKeyValues = tileKeyValues.add(pass.linear(transitionPublicContextProjection, allPlayers));
        waitFeatures = waitOutcomes(pass, batch, indices, batchIds, per, tileCodes, waitIds);
      } else {
        waitFeatures = waitYakus(pass, batch, indices, slots);
      }
      NDArray attended;
      try (ArrayScope part = new ArrayScope()) {
        attended =
            part.keep(
                NDArrays.mappedGroupedIndexedScaledDotProductAttention(
                        pass.linear(transitionTileQueryProjection, features)
                            .reshape(present, Widths.HEADS, TRANSITION_KEY),
                        tileKeyValues,
                        batchIds.toType(DataType.INT32, false),
                        pass.ownedTable(transitionTileRelationEmbedding),
                        tileCodes.toType(DataType.INT32, false),
                        pass.linear(transitionWaitKeyValueProjection, waitFeatures),
                        waitIds.toType(DataType.INT32, false),
                        1.0 / Math.sqrt(TRANSITION_KEY))
                    .reshape(present, context));
      }
      NDArray discardTiles = compact.get(":, {}", discardTile);
      NDArray embeddings =
          Pass.silu(
              transitionFusion.fuse(
                  pass,
                  features,
                  attended,
                  NDArrays.paddedBatchGatherByBatchIndices(memory.tiles(), batchIds, discardTiles),
                  NDArrays.paddedBatchGatherByBatchIndices(allPlayers, batchIds, discardTiles),
                  NDArrays.gatherRows(state, batchIds)));
      embeddings = pass.norm(transitionLayerNorm, embeddings);
      NDArray discards = compact.get(":, {}", discardActionId).neq(0).toType(pass.compute(), false);
      NDArray scores = discardCandidateScorer.score(pass, embeddings).reshape(present).mul(discards);
      NDArray denseScores =
          NDArrays.scatterRows(scores.reshape(present, 1), indices, slots).reshape(rows, actions, per);
      NDArray denseEmbeddings = NDArrays.scatterRows(embeddings, indices, slots).reshape(rows, actions, per, hidden);
      if (per == 1) {
        return new Transitions(scope.keep(denseScores), scope.keep(denseEmbeddings.reshape(rows, actions, hidden)));
      }
      NDArray presentMask = presentValues.reshape(rows, actions, per).neq(0).toType(pass.compute(), false);
      NDArray weights = pass.toCompute(NDArrays.maskedSoftmax(denseScores, presentMask, 2).stopGradient());
      return new Transitions(
          scope.keep(denseScores), scope.keep(denseEmbeddings.mul(weights.expandDims(3)).sum(new int[] {2})));
    }
  }

  /**
   * Nano の待ちの特徴を作る。ロンとツモの役のビットを射影した埋め込みに、待ちの点の値を並べる。
   *
   * @return [遷移, 13, 2 × 16 ＋ 点の値の数]（計算の型）
   */
  private NDArray waitYakus(Pass pass, ClassicBatch batch, NDArray indices, long slots) {
    try (ArrayScope scope = new ArrayScope()) {
      long present = indices.size();
      int chunks = WAITS * WIN_TYPES * ClassicSchema.YAKU_CHUNKS;
      NDArray stored =
          NDArrays.gatherRows(batch.tensor(Tensor.WAIT_YAKUS).reshape(slots, chunks), indices)
              .toType(DataType.FLOAT32, false)
              .maximum(1f)
              .sub(1f)
              .reshape(present, WAITS, WIN_TYPES, ClassicSchema.YAKU_CHUNKS, 1);
      NDArray bits =
          stored
              .div(yakuBitDivisors)
              .floor()
              .mod(2f)
              .reshape(present, WAITS, WIN_TYPES, YAKU_BITS);
      NDArray yakus =
          pass.linear(transitionWaitYakuProjection, pass.toCompute(bits))
              .reshape(present, WAITS, (long) WIN_TYPES * YAKU_EMBEDDING);
      NDArray scores =
          pass.toCompute(
              NDArrays.gatherRows(
                      batch.tensor(Tensor.WAIT_SCORES).reshape(slots, (long) WAITS * waitScoreStride), indices)
                  .reshape(present, WAITS, waitScoreStride));
      return scope.keep(yakus.concat(scores, 2));
    }
  }

  /**
   * Major の待ちの特徴を作る。和了の仕方は待ちごとに 8 通りある（3 席からのロンとツモの 4 通りを、赤 5 で和了る場合とそうでない場合に
   * 分ける）。その結果を点数の計算で求めて射影し、役のある和了だけで平均する。計算するのは、ロンかツモに役のある待ちだけ。
   *
   * @return [遷移, 13, 文脈]（計算の型）。役のある和了が 1 つも無い待ちは 0
   */
  private NDArray waitOutcomes(
      Pass pass,
      ClassicBatch batch,
      NDArray indices,
      NDArray batchIds,
      int per,
      NDArray tileCodes,
      NDArray waitIds) {
    long present = indices.size();
    long slots = (long) batch.rows() * batch.bucket().actions() * per;
    NDArray facts =
        NDArrays.gatherRows(
                batch.tensor(Tensor.WAIT_WIN_FACTS).reshape(slots, (long) WAITS * WIN_TYPES * waitFactStride),
                indices)
            .toType(DataType.INT32, false)
            .reshape(present * WAITS, WIN_TYPES, waitFactStride);
    NDArray waits = facts.get(":, :, {}", waitFactValid).sum(new int[] {1}).gt(0).nonzero().reshape(-1);
    long count = waits.size();
    if (count == 0) {
      return pass.working().zeros(new Shape(present, WAITS, context), pass.compute());
    }
    NDArray owners = waits.floorDivide(WAITS);
    NDArray kinds = NDArrays.gatherRows(waitIds.reshape(present * WAITS), waits).toType(DataType.INT64, false);
    NDArray aka =
        NDArrays.gatherRows(tileCodes, owners)
            .gather(kinds.maximum(1).sub(1).reshape(count, 1), 1)
            .reshape(count)
            .toType(DataType.INT32, false)
            .maximum(1)
            .sub(1)
            .mod(2)
            .mul(kinds.neq(0).toType(DataType.INT32, false));
    NDArray actionTypes =
        batch
            .tensor(Tensor.ACTION_CATEGORIES)
            .get("..., {}", columns.type())
            .reshape((long) batch.rows() * batch.bucket().actions());
    NDArray riichi =
        NDArrays.gatherRows(actionTypes, NDArrays.gatherRows(indices, owners).floorDivide(per))
            .eq(ActionColumns.stored(ActionType.RIICHI_DAHAI))
            .toType(DataType.INT32, false);
    NDArray rowsOfWaits = NDArrays.gatherRows(batchIds, owners);
    Projection projection =
        pointProjection.waits(
            pass,
            NDArrays.gatherRows(batch.tensor(Tensor.POINT_LEDGER_100), rowsOfWaits).toType(DataType.INT32, false),
            NDArrays.gatherRows(batch.tensor(Tensor.STATE_CATEGORIES), rowsOfWaits),
            NDArrays.gatherRows(facts, waits),
            aka,
            riichi);
    NDArray outcomes = pointOutcomeEncoder.encode(pass, pass.toCompute(projection.features()));
    NDArray cases = pass.toCompute(projection.valid().sum(new int[] {1}).maximum(1f).reshape(count, 1));
    NDArray pooled = outcomes.sum(new int[] {1}).div(cases);
    return NDArrays.scatterRows(pooled, waits, present * WAITS).reshape(present, WAITS, context);
  }

  /** 分岐のスコアを求める。種類の分岐は種類の中の候補の分布で、二択は下のノードの分布で文脈を集め、その文脈から採点する。 */
  private NDArray alternatives(
      Pass pass,
      NDArray state,
      NDArray candidates,
      NDArray typeMasks,
      NDArray actionScores,
      NDArray riichiScores,
      DiscardChoices choices,
      NDArray damaContexts,
      NDArray riichiContexts,
      NDArray pointGates) {
    try (ArrayScope scope = new ArrayScope()) {
      long rows = state.getShape().get(0);
      NDManager working = pass.working();
      NDArray pooled =
          pass.toCompute(NDArrays.groupedMaskedSoftmaxPool(actionScores.stopGradient(), typeMasks, candidates));
      NDArray typePresence = typeMasks.transpose(2, 0, 1).sum(new int[] {2}).minimum(1f);
      NDArray zeroContext = working.zeros(new Shape(rows, hidden), pass.compute());
      NDArray zeroPresence = working.zeros(new Shape(rows), pass.compute());
      NDList contexts = new NDList(ALTERNATIVES);
      NDList presence = new NDList(ALTERNATIVES);
      for (Alternative alternative : Alternative.values()) {
        ActionType type = alternative.type();
        contexts.add(type == null ? zeroContext : pooled.get("{}, :, :", type.ordinal()));
        presence.add(type == null ? zeroPresence : typePresence.get("{}, :", type.ordinal()));
      }
      NDArray alternativeContexts = NDArrays.stack(contexts, 1);
      NDArray alternativePresence = NDArrays.stack(presence, 1);
      NDArray states = state.reshape(rows, 1, hidden).broadcast(rows, ALTERNATIVES, hidden);
      NDArray preactivation;
      if (major) {
        preactivation =
            pass.linear(alternativeHidden, states.concat(alternativeContexts, 2))
                .add(pass.own(pass.table(alternativeOffset).reshape(1, ALTERNATIVES, hidden)));
      } else {
        NDArray fixed =
            pass.embed(policyNodeEmbedding, alternativeNodes)
                .concat(pass.table(alternativeEmbedding), 1)
                .reshape(1, ALTERNATIVES, 2L * hidden)
                .broadcast(rows, ALTERNATIVES, 2L * hidden);
        preactivation =
            pass.linear(alternativeHidden, NDArrays.concat(new NDList(states, alternativeContexts, fixed), 2));
      }
      NDArray scored =
          pass.linear(alternativeScoreHead, Pass.silu(preactivation))
              .reshape(rows, ALTERNATIVES)
              .mul(alternativePresence);

      NDArray passContext = pooled.get("{}, :, :", ActionType.PASS.ordinal());
      NDArray passPresence = typePresence.get("{}, :", ActionType.PASS.ordinal()).reshape(rows, 1);
      NDArray meldContext = pool(pass, scored, alternativePresence, alternativeContexts, MELD_ALTERNATIVES);
      NDArray meldPresence = any(alternativePresence, MELD_ALTERNATIVES);
      NDArray callScore =
          callGate.score(pass, state, passContext, meldContext, null).mul(passPresence.mul(meldPresence));
      NDArray declinedRon = mix(passContext, meldContext, callScore, passPresence, meldPresence);
      NDArray declinedRonPresence = either(passPresence, meldPresence);
      NDArray ronScore =
          ronGate
              .score(
                  pass,
                  state,
                  declinedRon,
                  context(alternativeContexts, Alternative.RON),
                  pointGates == null ? null : pointGates(pointGates, typeMasks, ActionType.RON_AGARI))
              .mul(declinedRonPresence.mul(has(alternativePresence, Alternative.RON)));

      NDArray continuation = continuation(pass, choices, damaContexts, riichiContexts, actionScores, riichiScores);
      NDArray kanContext = pool(pass, scored, alternativePresence, alternativeContexts, KAN_ALTERNATIVES);
      NDArray continuationPresence = choices.representativeMask().sum(new int[] {1}).reshape(rows, 1).minimum(1f);
      NDArray kanPresence = any(alternativePresence, KAN_ALTERNATIVES);
      NDArray kanScore =
          kanGate.score(pass, state, continuation, kanContext, null).mul(continuationPresence.mul(kanPresence));
      NDArray continued = mix(continuation, kanContext, kanScore, continuationPresence, kanPresence);
      NDArray continuedPresence = either(continuationPresence, kanPresence);
      NDArray kyushuContext = context(alternativeContexts, Alternative.KYUSHU);
      NDArray kyushuPresence = has(alternativePresence, Alternative.KYUSHU);
      NDArray kyushuScore =
          kyushuGate.score(pass, state, continued, kyushuContext, null).mul(continuedPresence.mul(kyushuPresence));
      NDArray declinedTsumo = mix(continued, kyushuContext, kyushuScore, continuedPresence, kyushuPresence);
      NDArray declinedTsumoPresence = either(continuedPresence, kyushuPresence);
      NDArray tsumoScore =
          tsumoGate
              .score(
                  pass,
                  state,
                  declinedTsumo,
                  context(alternativeContexts, Alternative.TSUMO),
                  pointGates == null ? null : pointGates(pointGates, typeMasks, ActionType.TSUMO_AGARI))
              .mul(declinedTsumoPresence.mul(has(alternativePresence, Alternative.TSUMO)));

      NDList columns = new NDList(ALTERNATIVES);
      for (Alternative alternative : Alternative.values()) {
        columns.add(
            switch (alternative) {
              case RON -> ronScore;
              case CALL -> callScore;
              case TSUMO -> tsumoScore;
              case KYUSHU -> kyushuScore;
              case KAN -> kanScore;
              default -> scored.get(":, {}", alternative.ordinal()).reshape(rows, 1);
            });
      }
      return scope.keep(NDArrays.concat(columns, 1));
    }
  }

  /**
   * 打牌へ進む側の文脈を作る。物理の打牌ごとに、ダマと立直の 2 つの表現を立直の確率で合わせる。それを物理の打牌の分布でさらに
   * 合わせる。重みの勾配は止める。
   *
   * @return [行, 幅]
   */
  private static NDArray continuation(
      Pass pass,
      DiscardChoices choices,
      NDArray damaContexts,
      NDArray riichiContexts,
      NDArray actionScores,
      NDArray riichiScores) {
    NDArray riichiProbability =
        Activation.sigmoid(NDArrays.paddedBatchGather(riichiScores, choices.riichiSlots())).stopGradient();
    NDArray joint = choices.damaPresent().mul(choices.riichiPresent());
    NDArray riichiMass = joint.mul(riichiProbability);
    NDArray riichiWeight = choices.riichiPresent().sub(joint).add(riichiMass).stopGradient();
    NDArray damaWeight = choices.damaPresent().sub(riichiMass).stopGradient();
    NDArray identities =
        damaContexts.mul(damaWeight.expandDims(2)).add(riichiContexts.mul(riichiWeight.expandDims(2)));
    NDArray identityWeights =
        pass.toCompute(NDArrays.maskedSoftmax(actionScores, choices.representativeMask(), 1).stopGradient());
    return identities.mul(identityWeights.expandDims(2)).sum(new int[] {1});
  }

  /**
   * 二択の 2 つの側の文脈を、二択の分布で合わせる（重みの勾配は止める）。片側しか無ければ、その側の文脈を返す。
   *
   * @param baseline 基準の側 [行, 幅]
   * @param selected 選ぶ側 [行, 幅]
   * @param logit 選ぶ側のロジット [行, 1]
   * @param baselinePresence 基準の側があれば 1 [行, 1]
   * @param selectedPresence 選ぶ側があれば 1 [行, 1]
   * @return 合わせた文脈 [行, 幅]
   */
  private static NDArray mix(
      NDArray baseline, NDArray selected, NDArray logit, NDArray baselinePresence, NDArray selectedPresence) {
    NDArray joint = baselinePresence.mul(selectedPresence);
    NDArray selectedMass = joint.mul(Activation.sigmoid(logit).stopGradient());
    NDArray selectedWeight = selectedPresence.sub(joint).add(selectedMass).stopGradient();
    NDArray baselineWeight = baselinePresence.sub(selectedMass).stopGradient();
    return baseline.mul(baselineWeight).add(selected.mul(selectedWeight));
  }

  /**
   * 種類の分岐の文脈を、分岐のスコアの分布で合わせて [行, 幅] で返す。重みの勾配は止める。
   *
   * @param alternatives 合わせる分岐の番号
   */
  private static NDArray pool(Pass pass, NDArray scored, NDArray presence, NDArray contexts, int[] alternatives) {
    return pass.toCompute(NDArrays.indexedMaskedSoftmaxPool(scored.stopGradient(), presence, contexts, alternatives));
  }

  /**
   * どれかの分岐があれば 1 の配列 [行, 1] を返す。
   *
   * @param alternatives 分岐の番号
   */
  private static NDArray any(NDArray presence, int[] alternatives) {
    NDArray sum = presence.get(":, {}", alternatives[0]);
    for (int index = 1; index < alternatives.length; index++) {
      sum = sum.add(presence.get(":, {}", alternatives[index]));
    }
    return sum.minimum(1f).reshape(-1, 1);
  }

  /** 分岐があれば 1 の配列 [行, 1] を返す。 */
  private static NDArray has(NDArray presence, Alternative alternative) {
    return presence.get(":, {}", alternative.ordinal()).minimum(1f).reshape(-1, 1);
  }

  private static int[] ordinals(Alternative... alternatives) {
    int[] ordinals = new int[alternatives.length];
    for (int index = 0; index < alternatives.length; index++) {
      ordinals[index] = alternatives[index].ordinal();
    }
    return ordinals;
  }

  private static NDArray either(NDArray first, NDArray second) {
    return first.add(second).minimum(1f);
  }

  private static NDArray context(NDArray contexts, Alternative alternative) {
    return contexts.get(":, {}, :", alternative.ordinal());
  }

  /** 指定した種別の手について、二択に直接足す和了の結果 [行, 6] を返す。その種別の手は 1 行に 1 つまでなので、合法手の軸で足して取り出す。 */
  private static NDArray pointGates(NDArray gates, NDArray typeMasks, ActionType type) {
    return gates.mul(type(typeMasks, type).expandDims(2)).sum(new int[] {1});
  }

  private static NDArray type(NDArray typeMasks, ActionType type) {
    return typeMasks.get("..., {}", type.ordinal());
  }

  private static IdEmbedding table(int rows, int width) {
    return new IdEmbedding.Builder().setDictionarySize(rows).setEmbeddingSize(width).build();
  }

  @Override
  protected void initializeChildBlocks(NDManager manager, DataType dataType, Shape... inputShapes) {
    actionOffsets = manager.create(actionOffsetValues).reshape(1, 1, actionOffsetValues.length);
    transitionOffsets = manager.create(transitionOffsetValues).reshape(1, transitionOffsetValues.length);
    actionFeatureEmbedding.initialize(manager, dataType, new Shape(-1, -1, columns.stride()));
    transitionTileRelationEmbedding.initialize(manager, dataType, new Shape(-1, -1, -1, Tiles.KINDS));
    Shape state = new Shape(-1, hidden);
    if (major) {
      pointProjection.initialize(manager, dataType, new Shape(-1, PointProjection.FEATURES));
      pointOutcomeEncoder.initialize(manager, dataType, new Shape(-1, PointProjection.FEATURES));
      alternativeOffset.initialize(manager, dataType, new Shape(ALTERNATIVES));
    } else {
      long[] nodes = new long[ALTERNATIVES];
      for (Alternative alternative : Alternative.values()) {
        nodes[alternative.ordinal()] = alternative.node().ordinal();
      }
      alternativeNodes = manager.create(nodes);
      float[] divisors = new float[ClassicSchema.YAKU_CHUNK_BITS];
      for (int bit = 0; bit < divisors.length; bit++) {
        divisors[bit] = 1 << bit;
      }
      yakuBitDivisors = manager.create(divisors);
      transitionWaitYakuProjection.initialize(manager, dataType, new Shape(-1, YAKU_BITS));
      policyNodeEmbedding.initialize(manager, dataType, new Shape(ALTERNATIVES));
      alternativeEmbedding.initialize(manager, dataType, new Shape(ALTERNATIVES));
    }
    int actionExtra = major ? context : ClassicSchema.of(Architecture.NANO).layout(ActionFloat.class).stride();
    actionFeatureProjection.initialize(
        manager, dataType, new Shape(-1, (long) columns.stride() * Widths.CATEGORY + actionExtra));
    transitionFeatureProjection.initialize(
        manager, dataType, new Shape(-1, (long) transitionStride * Widths.CATEGORY + transitionNumerics));
    transitionTileKeyValueProjection.initialize(manager, dataType, state);
    if (major) {
      transitionPublicContextProjection.initialize(manager, dataType, new Shape(-1, context));
    }
    transitionTileQueryProjection.initialize(manager, dataType, state);
    int waitWidth = major ? context : WIN_TYPES * YAKU_EMBEDDING + waitScoreStride;
    transitionWaitKeyValueProjection.initialize(manager, dataType, new Shape(-1, waitWidth));
    playerTileQueryProjection.initialize(manager, dataType, state);
    if (!major) {
      playerMemoryKeyValueProjection.initialize(manager, dataType, state);
    }
    playerTileContextMixer.initialize(
        manager, dataType, new Shape(-1, ClassicSchema.SEATS, Tiles.KINDS, context));
    transitionFusion.initialize(manager, dataType, state);
    transitionLayerNorm.initialize(manager, dataType, new Shape(-1, -1, -1, hidden));
    discardCandidateScorer.initialize(manager, dataType, state);
    candidateFusion.initialize(manager, dataType, state);
    candidateLayerNorm.initialize(manager, dataType, new Shape(-1, -1, hidden));
    actionCandidateHidden.initialize(manager, dataType, state);
    actionCandidateScoreHead.initialize(manager, dataType, state);
    alternativeHidden.initialize(manager, dataType, new Shape(-1, (major ? 2L : 4L) * hidden));
    alternativeScoreHead.initialize(manager, dataType, state);
    for (BinaryGate gate : new BinaryGate[] {ronGate, callGate, tsumoGate, kyushuGate, kanGate, riichiGate}) {
      gate.initialize(manager, dataType, state);
    }
  }

  @Override
  protected NDList forwardInternal(
      ParameterStore parameterStore, NDList inputs, boolean training, PairList<String, Object> params) {
    throw new UnsupportedOperationException("Use score");
  }

  @Override
  public Shape[] getOutputShapes(Shape[] inputShapes) {
    return new Shape[] {new Shape(-1, ALTERNATIVES), new Shape(-1, -1), new Shape(-1, -1)};
  }

  /**
   * 遷移の計算の結果。
   *
   * @param scores 打牌の遷移のロジット [行, 合法手, 遷移]（打牌でない遷移とパディングは 0）
   * @param aggregated 合法手ごとに 1 つにまとめた遷移の表現 [行, 合法手, 幅]
   */
  private record Transitions(NDArray scores, NDArray aggregated) {}
}
