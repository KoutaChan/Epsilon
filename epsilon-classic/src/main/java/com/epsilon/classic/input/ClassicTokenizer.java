package com.epsilon.classic.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationSchema.LastEvent;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.input.ClassicFields.ActionFloat;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.ActionRoute;
import com.epsilon.classic.input.ClassicFields.ActionTransitionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicFields.ActionTransitionWaitFloat;
import com.epsilon.classic.input.ClassicFields.BoundaryFloat;
import com.epsilon.classic.input.ClassicFields.MeldFloat;
import com.epsilon.classic.input.ClassicFields.MeldInt;
import com.epsilon.classic.input.ClassicFields.PlayerFloat;
import com.epsilon.classic.input.ClassicFields.PlayerInt;
import com.epsilon.classic.input.ClassicFields.RiverFloat;
import com.epsilon.classic.input.ClassicFields.RiverInt;
import com.epsilon.classic.input.ClassicFields.RoundFloat;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFlag;
import com.epsilon.classic.input.ClassicFields.TileFloat;
import com.epsilon.classic.input.ClassicFields.TileInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicSchema.StateToken;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.ClassicVocabulary.DiscardContext;
import com.epsilon.classic.input.ClassicVocabulary.DrawSource;
import com.epsilon.classic.input.ClassicVocabulary.EventType;
import com.epsilon.classic.input.ClassicVocabulary.NextStep;
import com.epsilon.classic.input.ClassicVocabulary.RonFuritenKind;
import com.epsilon.classic.input.ClassicVocabulary.TransitionKind;
import com.epsilon.classic.input.ClassicVocabulary.WaitWinType;
import com.epsilon.classic.input.ClassicVocabulary.WinConditions;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.TurnEvent;
import com.epsilon.mahjong.engine.WinSettlementProjection.SettlementAssumption;
import com.epsilon.mahjong.rules.ScoreMath;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.score.WinScore;
import com.epsilon.mahjong.tile.Tiles;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の記録と追加の入力（{@link ClassicExtras}）を、テンソルの 1 行に書く。テンソルはアーキテクチャごとに 12 個ある。どのフィールドも、
 * 判断の記録とルールの関数（手の形の解析と採点）だけから求める。書く前に行を 0（パディング）で埋める。
 *
 * <p>ルールで決まる事実は、判断の記録から作り直す。牌の枚数は {@code TileCounts}、順位と現物は {@code SeatFacts}、河と面子は
 * {@link EventScan}、合法手の後の手は {@link OwnHand}、手の形と待ちは {@link HandAnalysis} で求める。合法手の後の遷移は、チー・ポンなら
 * 判断の記録にある鳴いた後の打牌の数だけあり、ほかの手では 1 つ。
 *
 * <p>宣言牌への応答の判断では、宣言した席の立直はまだ成立していない（供託は応答が出そろってから払う）。それでも席のフィールドには、
 * 宣言牌で決まる立直と一発の印を書き、現物も立直した席と同じに書く。
 *
 * <p>ウォームアップの後はメモリを確保しない。作業領域を持つのでスレッドセーフではなく、スレッドごとに 1 つ作る。値がフィールドの幅を
 * 外れると、書くときに IllegalStateException を投げる。
 */
public final class ClassicTokenizer {

  private static final int SEATS = ClassicSchema.SEATS;
  private static final int RIVER = ClassicSchema.RIVER_TOKENS;
  private static final int MELDS = ClassicSchema.MELD_TOKENS;
  private static final int WAITS = ClassicSchema.MAX_WAITS;
  private static final int WIN_TYPES = WaitWinType.values().length;
  private static final int NO_KEY = -1;

  /** 符のコードで表せる符の上限。これより大きい符は上限として書く。 */
  private static final int MAX_FU = 110;

  /** 30 符以上の符をコードにするときの刻み。 */
  private static final int FU_UNIT = 10;

  /** GRP の事前分布が無いときに書く、絶対席と順位の組ごとの周辺確率。 */
  private static final float UNIFORM_RANK = 1f / SEATS;

  private final Architecture architecture;
  private final ClassicSchema schema;
  private final Variant variant;
  private final TileCounts tiles;
  private final SeatFacts seats;
  private final EventScan scan = new EventScan();

  /** 判断の時点の自分の手。 */
  private final HandAnalysis current;

  /** 合法手の後の手。 */
  private final HandAnalysis after;

  private final int[] roundStartScores = new int[SEATS];
  private final int[] afterWin = new int[SEATS];

  /** 物理の打牌の番号ごとに、立直でない打牌が合法手の何番目にあるか（無ければ −1）。下の配列は、同じ並びで立直の打牌の位置を持つ。 */
  private final int[] damaSlots = new int[ClassicVocabulary.DISCARD_IDENTITIES];

  private final int[] riichiSlots = new int[ClassicVocabulary.DISCARD_IDENTITIES];

  private final FieldLayout<RoundInt> roundInt;
  private final FieldLayout<RoundFloat> roundFloat;
  private final FieldLayout<PlayerInt> playerInt;
  private final FieldLayout<PlayerFloat> playerFloat;
  private final FieldLayout<TileInt> tileInt;
  private final FieldLayout<TileFloat> tileFloat;
  private final FieldLayout<RiverInt> riverInt;
  private final FieldLayout<RiverFloat> riverFloat;
  private final FieldLayout<MeldInt> meldInt;
  private final FieldLayout<MeldFloat> meldFloat;
  private final FieldLayout<ActionInt> actionInt;
  private final FieldLayout<ActionFloat> actionFloat;
  private final FieldLayout<ActionRoute> actionRoute;
  private final FieldLayout<WinFact> winFact;
  private final FieldLayout<ActionTransitionInt> transitionInt;
  private final FieldLayout<ActionTransitionFloat> transitionFloat;
  private final FieldLayout<ActionTransitionWaitFloat> waitFloat;
  private final FieldLayout<BoundaryFloat> boundary;

  /** 書き込み先の整数と実数の配列。この後に続くフィールドには、書いている行の始まる位置をテンソルごとに持つ。 */
  private short[] categorical;

  private float[] numeric;
  private int transitionsPerAction;
  private int stateCategories;
  private int stateNumerics;
  private int actionCategories;
  private int actionNumerics;
  private int actionRoutes;
  private int actionWinFacts;
  private int pointLedger;
  private int transitionCategories;
  private int transitionTiles;
  private int waitTileIds;
  private int waitYakus;
  private int waitWinFacts;
  private int transitionNumerics;
  private int waitScores;
  private int boundaryContext;

  /**
   * 1 つのアーキテクチャの入力を書く作業領域を作る。
   *
   * @param architecture 入力を書くアーキテクチャ
   * @param variant ルール（四人打ちだけ）
   * @throws IllegalArgumentException 四人打ちでないルールを渡したとき
   */
  public ClassicTokenizer(Architecture architecture, Variant variant) {
    if (variant.seats() != SEATS) {
      throw new IllegalArgumentException("Classic inputs need " + SEATS + " seats, not " + variant);
    }
    this.architecture = Objects.requireNonNull(architecture, "architecture");
    this.variant = variant;
    schema = ClassicSchema.of(architecture);
    tiles = new TileCounts(variant);
    seats = new SeatFacts(variant);
    current = new HandAnalysis(variant);
    after = new HandAnalysis(variant);
    roundInt = schema.layout(RoundInt.class);
    roundFloat = schema.layout(RoundFloat.class);
    playerInt = schema.layout(PlayerInt.class);
    playerFloat = schema.layout(PlayerFloat.class);
    tileInt = schema.layout(TileInt.class);
    tileFloat = schema.layout(TileFloat.class);
    riverInt = schema.layout(RiverInt.class);
    riverFloat = schema.layout(RiverFloat.class);
    meldInt = schema.layout(MeldInt.class);
    meldFloat = schema.layout(MeldFloat.class);
    actionInt = schema.layout(ActionInt.class);
    actionFloat = schema.layout(ActionFloat.class);
    actionRoute = schema.layout(ActionRoute.class);
    winFact = schema.layout(WinFact.class);
    transitionInt = schema.layout(ActionTransitionInt.class);
    transitionFloat = schema.layout(ActionTransitionFloat.class);
    waitFloat = schema.layout(ActionTransitionWaitFloat.class);
    boundary = schema.layout(BoundaryFloat.class);
  }

  public Architecture architecture() {
    return architecture;
  }

  /**
   * 判断を 1 行に書く。方策だけを使うときの書き方で、価値の入力には一様な GRP の事前分布を書く。局の始まりの値は 0 のままにする。
   *
   * @param o 判断の記録
   * @param extras 判断と同じ時点の追加の入力
   * @param into 書き込み先
   * @param row 書く行
   * @throws IllegalArgumentException ルールかアーキテクチャが合わないとき、または判断が書き込み先の区分に収まらないとき
   */
  public void write(Observation o, ClassicExtras extras, ClassicInputs into, int row) {
    encode(o, extras, null, into, row);
  }

  /**
   * 判断を 1 行に書き、価値の入力に GRP の事前分布と局の始まりの値を書く。
   *
   * @param o 判断の記録
   * @param extras 判断と同じ時点の追加の入力
   * @param grpPrior 局の始まりの GRP の周辺確率。絶対席の順に、席ごとの順位の確率を 4 つずつ並べる。どれも 0 以上の有限の値
   * @param into 書き込み先
   * @param row 書く行
   * @throws IllegalArgumentException ルールかアーキテクチャが合わないとき、判断が書き込み先の区分に収まらないとき、または事前分布の
   *     要素の数か値が正しくないとき
   */
  public void write(Observation o, ClassicExtras extras, float[] grpPrior, ClassicInputs into, int row) {
    Objects.requireNonNull(grpPrior, "grpPrior");
    if (grpPrior.length != BoundaryFloat.GRP_RANK.count()) {
      throw new IllegalArgumentException("A GRP prior holds " + BoundaryFloat.GRP_RANK.count() + " values");
    }
    for (float probability : grpPrior) {
      if (!(probability >= 0) || !Float.isFinite(probability)) {
        throw new IllegalArgumentException("GRP prior values must be finite and non-negative: " + probability);
      }
    }
    encode(o, extras, grpPrior, into, row);
  }

  private void encode(Observation o, ClassicExtras extras, float[] grpPrior, ClassicInputs into, int row) {
    Objects.requireNonNull(extras, "extras");
    if (o.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    if (into.schema() != schema) {
      throw new IllegalArgumentException(
          "Inputs of " + into.schema().architecture() + " cannot hold " + architecture + " rows");
    }
    int legal = o.legalActions();
    int maxTransitions = Buckets.maxTransitions(o);
    if (!into.bucket().fits(legal, maxTransitions)) {
      throw new IllegalArgumentException(
          legal + " actions with up to " + maxTransitions + " transitions do not fit " + into.bucket());
    }
    bind(into, row);
    tiles.load(o);
    seats.load(o);
    scan.load(o);
    current.load(o, tiles, seats);
    current.analyzeShape();
    writeRound(o);
    writePlayers(o);
    writeTiles(o);
    writeRivers();
    writeMelds();
    if (architecture.publicHistory()) {
      writeHistory(o);
    }
    if (has(Tensor.POINT_LEDGER_100)) {
      for (int seat = 0; seat < SEATS; seat++) {
        categorical[pointLedger + o.absoluteSeat(seat)] = (short) (o.score(seat) / 100);
      }
    }
    writeBoundary(o, grpPrior);
    writeRoutes(o);
    writeActions(o);
  }

  /** 行を 0 で埋め、その行がテンソルごとにどこから始まるかを求める。 */
  private void bind(ClassicInputs into, int row) {
    into.clear(row);
    categorical = into.categorical();
    numeric = into.numeric();
    transitionsPerAction = into.bucket().transitions();
    stateCategories = into.start(Tensor.STATE_CATEGORIES, row);
    stateNumerics = into.start(Tensor.STATE_NUMERICS, row);
    actionCategories = into.start(Tensor.ACTION_CATEGORIES, row);
    actionNumerics = into.start(Tensor.ACTION_NUMERICS, row);
    actionRoutes = into.start(Tensor.ACTION_ROUTES, row);
    actionWinFacts = into.start(Tensor.ACTION_WIN_FACTS, row);
    pointLedger = into.start(Tensor.POINT_LEDGER_100, row);
    transitionCategories = into.start(Tensor.TRANSITION_CATEGORIES, row);
    transitionTiles = into.start(Tensor.TRANSITION_TILES, row);
    waitTileIds = into.start(Tensor.WAIT_TILE_IDS, row);
    waitYakus = into.start(Tensor.WAIT_YAKUS, row);
    waitWinFacts = into.start(Tensor.WAIT_WIN_FACTS, row);
    transitionNumerics = into.start(Tensor.TRANSITION_NUMERICS, row);
    waitScores = into.start(Tensor.WAIT_SCORES, row);
    boundaryContext = into.start(Tensor.BOUNDARY_CONTEXT, row);
  }

  private void writeRound(Observation o) {
    int base = stateCategories + schema.stateStart(StateToken.ROUND, Tensor.Slab.CATEGORICAL);
    LastEvent last = o.lastEvent();
    int lastSeat = last == null ? 0 : o.lastEventSeat();
    EngineDecisionKind kind = o.decisionKind();
    boolean ownTurn = kind == EngineDecisionKind.TURN || kind == EngineDecisionKind.POST_CALL_DAHAI;
    boolean beforeFirst = o.beforeFirstDiscard(0);
    category(base, roundInt, RoundInt.PLAYER_SEAT, o.observer() + 1);
    category(base, roundInt, RoundInt.CURRENT_PLAYER_RELATIVE_SEAT, (ownTurn ? 0 : lastSeat) + 1);
    category(base, roundInt, RoundInt.SOURCE_PLAYER_RELATIVE_SEAT, lastSeat + 1);
    category(base, roundInt, RoundInt.KYOKU_INDEX, o.kyoku() + 1);
    category(base, roundInt, RoundInt.DEALER_RELATIVE_SEAT, o.dealer() + 1);
    category(base, roundInt, RoundInt.BAKAZE, o.roundWindTile() + 1);
    category(base, roundInt, RoundInt.JIKAZE, variant.seatWind(o.kyoku(), o.observer()) + 1);
    category(base, roundInt, RoundInt.HONBA, o.honba());
    category(base, roundInt, RoundInt.KYOTAKU, o.kyotaku());
    category(base, roundInt, RoundInt.WALL_REMAINING, o.remainingWall());
    category(base, roundInt, RoundInt.TURN_NUMBER, scan.turn());
    category(base, roundInt, RoundInt.TOTAL_KAN, scan.kans());
    category(base, roundInt, RoundInt.DORA_INDICATOR_COUNT, o.indicators());
    flag(base, roundInt, RoundInt.ALL_LAST, o.kyoku() >= variant.lastRound());
    flag(base, roundInt, RoundInt.INITIAL_DISCARD_CYCLE, o.firstCycle());
    flag(base, roundInt, RoundInt.SELF_BEFORE_FIRST_DISCARD, beforeFirst);
    flag(base, roundInt, RoundInt.UNINTERRUPTED_FIRST_DRAW, beforeFirst && !o.callMade());
    flag(base, roundInt, RoundInt.FIRST_TURN_CALL_OCCURRED, o.callMade());
    flag(base, roundInt, RoundInt.LAST_LIVE_TILE, o.remainingWall() == 0);
    category(base, roundInt, RoundInt.EVENT_TYPE, eventType(last).ordinal() + 1);
    TurnEvent.KanKind kan = o.lastKanKind();
    category(base, roundInt, RoundInt.EVENT_KAN_KIND, kan == null ? 0 : VocabularyCodec.kanKind(kan).ordinal() + 1);
    category(base, roundInt, RoundInt.EVENT_PLAYER_RELATIVE_SEAT, last == null ? 0 : lastSeat + 1);
    category(base, roundInt, RoundInt.EVENT_TILE, o.lastEventTile() + 1);
    flag(base, roundInt, RoundInt.EVENT_IS_AKA, o.lastEventAka());
    DrawSource source = o.lastEventRinshan() ? DrawSource.RINSHAN : DrawSource.WALL;
    category(base, roundInt, RoundInt.EVENT_DRAW_SOURCE, last == LastEvent.DRAW ? source.ordinal() + 1 : 0);
    flag(base, roundInt, RoundInt.EVENT_RINSHAN, o.lastEventRinshan());
    flag(base, roundInt, RoundInt.EVENT_DEFERRED_DORA, o.deferredDora());

    int numbers = stateNumerics + schema.stateStart(StateToken.ROUND, Tensor.Slab.NUMERIC);
    int self = o.score(0);
    int highest = Integer.MIN_VALUE;
    int lowest = Integer.MAX_VALUE;
    int highestOther = Integer.MIN_VALUE;
    for (int seat = 0; seat < SEATS; seat++) {
      highest = Math.max(highest, o.score(seat));
      lowest = Math.min(lowest, o.score(seat));
      if (seat != 0) {
        highestOther = Math.max(highestOther, o.score(seat));
      }
    }
    number(numbers, roundFloat, RoundFloat.HONBA, o.honba());
    number(numbers, roundFloat, RoundFloat.KYOTAKU, o.kyotaku());
    number(numbers, roundFloat, RoundFloat.WALL_REMAINING, o.remainingWall());
    number(numbers, roundFloat, RoundFloat.TURN, scan.turn());
    number(numbers, roundFloat, RoundFloat.SELF_SCORE, self);
    number(numbers, roundFloat, RoundFloat.SCORE_LEAD, self - highestOther);
    number(numbers, roundFloat, RoundFloat.SCORE_TO_FIRST, self - highest);
    number(numbers, roundFloat, RoundFloat.SCORE_TO_FOURTH, self - lowest);
  }

  private void writePlayers(Observation o) {
    int categories = stateCategories + schema.stateStart(StateToken.PLAYER, Tensor.Slab.CATEGORICAL);
    int numbers = stateNumerics + schema.stateStart(StateToken.PLAYER, Tensor.Slab.NUMERIC);
    long ownRiver = seats.riverKinds(0);
    for (int seat = 0; seat < SEATS; seat++) {
      int base = categories + seat * playerInt.stride();
      int absolute = o.absoluteSeat(seat);
      int declaration = scan.riichiIndex(seat);
      boolean self = seat == 0;
      category(base, playerInt, PlayerInt.RELATIVE_SEAT, seat + 1);
      category(base, playerInt, PlayerInt.ABSOLUTE_SEAT, absolute + 1);
      category(base, playerInt, PlayerInt.JIKAZE, variant.seatWind(o.kyoku(), absolute) + 1);
      category(base, playerInt, PlayerInt.RIICHI_STATUS, VocabularyCodec.riichiStatus(riichi(o, seat)).ordinal() + 1);
      flag(base, playerInt, PlayerInt.IPPATSU, o.ippatsu(seat) || pending(o, seat));
      flag(base, playerInt, PlayerInt.BEFORE_FIRST_DISCARD, o.beforeFirstDiscard(seat));
      flag(base, playerInt, PlayerInt.SELF_TEMPORARY_FURITEN, self && o.furitenTemporary());
      flag(base, playerInt, PlayerInt.SELF_PERMANENT_FURITEN, self && (current.waits() & ownRiver) != 0);
      flag(base, playerInt, PlayerInt.MENZEN, scan.openMelds(seat) == 0);
      category(base, playerInt, PlayerInt.RANK, seats.rank(seat) + 1);
      category(base, playerInt, PlayerInt.MELD_COUNT, scan.meldCount(seat));
      category(base, playerInt, PlayerInt.OPEN_MELD_COUNT, scan.openMelds(seat));
      category(base, playerInt, PlayerInt.KAN_COUNT, scan.kans(seat));
      category(base, playerInt, PlayerInt.RIVER_COUNT, Math.min(scan.riverSize(seat), RIVER));
      category(base, playerInt, PlayerInt.RIICHI_DECLARATION_INDEX, riverPlace(declaration));
      category(base, playerInt, PlayerInt.DISCARDS_AFTER_RIICHI, Math.min(scan.afterRiichi(seat), RIVER));
      int tsumogiri = Math.min(scan.tsumogiriAfterRiichi(seat), RIVER);
      category(base, playerInt, PlayerInt.POST_RIICHI_TSUMOGIRI_COUNT, tsumogiri);

      int values = numbers + seat * playerFloat.stride();
      number(values, playerFloat, PlayerFloat.SCORE, o.score(seat));
      number(values, playerFloat, PlayerFloat.SCORE_FROM_SELF, o.score(seat) - o.score(0));
      number(values, playerFloat, PlayerFloat.RIVER_PROGRESS, scan.riverSize(seat));
      number(values, playerFloat, PlayerFloat.OPEN_MELD_FRACTION, scan.openMelds(seat));
      number(values, playerFloat, PlayerFloat.RIICHI_DECLARATION_PROGRESS, declaration < 0 ? 0 : declaration + 1);
      number(values, playerFloat, PlayerFloat.DEALER, seat == o.dealer() ? 1 : 0);
    }
  }

  private void writeTiles(Observation o) {
    int categories = stateCategories + schema.stateStart(StateToken.TILE, Tensor.Slab.CATEGORICAL);
    int numbers = stateNumerics + schema.stateStart(StateToken.TILE, Tensor.Slab.NUMERIC);
    long improving = current.improving();
    long waits = current.waits();
    long shimocha = genbutsu(o, 1);
    long toimen = genbutsu(o, 2);
    long kamicha = genbutsu(o, 3);
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int base = categories + kind * tileInt.stride();
      boolean wait = Tiles.contains(waits, kind);
      category(base, tileInt, TileInt.TILE_TYPE, kind + 1);
      category(base, tileInt, TileInt.SELF_HAND_COUNT, tiles.hand(kind));
      flag(base, tileInt, TileInt.SELF_HAS_AKA, Tiles.contains(tiles.handRed(), kind));
      category(base, tileInt, TileInt.VISIBLE_COUNT, tiles.visible(kind));
      category(base, tileInt, TileInt.DORA_INDICATOR_MULTIPLICITY, tiles.indicators(kind));
      category(base, tileInt, TileInt.DORA_MULTIPLICITY, tiles.dora(kind));
      category(base, tileInt, TileInt.SELF_RIVER_COUNT, scan.riverCount(0, kind));
      category(base, tileInt, TileInt.SHIMOCHA_RIVER_COUNT, scan.riverCount(1, kind));
      category(base, tileInt, TileInt.TOIMEN_RIVER_COUNT, scan.riverCount(2, kind));
      category(base, tileInt, TileInt.KAMICHA_RIVER_COUNT, scan.riverCount(3, kind));
      flag(base, tileInt, TileInt.SELF_UKEIRE, Tiles.contains(improving, kind));
      flag(base, tileInt, TileInt.SELF_WAIT, wait);
      flag(base, tileInt, TileInt.SELF_DISCARDED_WAIT, wait && scan.riverCount(0, kind) > 0);
      flag(base, tileInt, TileInt.SHIMOCHA_RIICHI_GENBUTSU, Tiles.contains(shimocha, kind));
      flag(base, tileInt, TileInt.TOIMEN_RIICHI_GENBUTSU, Tiles.contains(toimen, kind));
      flag(base, tileInt, TileInt.KAMICHA_RIICHI_GENBUTSU, Tiles.contains(kamicha, kind));

      int values = numbers + kind * tileFloat.stride();
      number(values, tileFloat, TileFloat.SELF_HAND_FRACTION, tiles.hand(kind));
      number(values, tileFloat, TileFloat.VISIBLE_FRACTION, tiles.visible(kind));
      number(values, tileFloat, TileFloat.UNSEEN_FRACTION, tiles.unseen(kind));
      number(values, tileFloat, TileFloat.DORA_MULTIPLICITY, tiles.dora(kind));
    }
  }

  private void writeRivers() {
    int categories = stateCategories + schema.stateStart(StateToken.RIVER, Tensor.Slab.CATEGORICAL);
    int numbers = stateNumerics + schema.stateStart(StateToken.RIVER, Tensor.Slab.NUMERIC);
    for (int seat = 0; seat < SEATS; seat++) {
      for (int index = 0; index < scan.riverTokens(seat); index++) {
        int token = seat * RIVER + index;
        int base = categories + token * riverInt.stride();
        int sequence = scan.riverSequence(seat, index);
        category(base, riverInt, RiverInt.RELATIVE_PLAYER, seat + 1);
        category(base, riverInt, RiverInt.INDEX, index + 1);
        category(base, riverInt, RiverInt.TILE, scan.riverTile(seat, index) + 1);
        flag(base, riverInt, RiverInt.IS_AKA, scan.riverAka(seat, index));
        flag(base, riverInt, RiverInt.TSUMOGIRI, scan.riverTsumogiri(seat, index));
        flag(base, riverInt, RiverInt.RIICHI, scan.riverRiichi(seat, index));
        flag(base, riverInt, RiverInt.CALLED, scan.riverCalled(seat, index));
        category(base, riverInt, RiverInt.TURN, scan.riverTurn(seat, index) + 1);
        category(base, riverInt, RiverInt.GLOBAL_SEQUENCE, sequence + 1);
        flag(base, riverInt, RiverInt.PRESENT, true);
        int values = numbers + token * riverFloat.stride();
        number(values, riverFloat, RiverFloat.PLAYER_PROGRESS, index);
        number(values, riverFloat, RiverFloat.GLOBAL_PROGRESS, sequence);
      }
    }
  }

  private void writeMelds() {
    int categories = stateCategories + schema.stateStart(StateToken.MELD, Tensor.Slab.CATEGORICAL);
    int numbers = stateNumerics + schema.stateStart(StateToken.MELD, Tensor.Slab.NUMERIC);
    for (int seat = 0; seat < SEATS; seat++) {
      for (int index = 0; index < scan.meldCount(seat); index++) {
        int token = seat * MELDS + index;
        int base = categories + token * meldInt.stride();
        Meld meld = scan.meld(seat, index);
        int call = scan.callRiver(seat, index);
        int kan = scan.kanRiver(seat, index);
        category(base, meldInt, MeldInt.RELATIVE_PLAYER, seat + 1);
        category(base, meldInt, MeldInt.INDEX, index + 1);
        category(base, meldInt, MeldInt.TYPE, VocabularyCodec.meldType(meld.type()).ordinal() + 1);
        category(base, meldInt, MeldInt.BASE_TILE, meld.base() + 1);
        category(base, meldInt, MeldInt.CALLED_TILE, meld.called() + 1);
        category(base, meldInt, MeldInt.SOURCE, VocabularyCodec.meldSource(meld.source()).ordinal() + 1);
        category(base, meldInt, MeldInt.AKA_SOURCE, VocabularyCodec.meldAkaSource(meld.aka()).ordinal() + 1);
        category(base, meldInt, MeldInt.SIZE, meld.size());
        category(base, meldInt, MeldInt.CALL_AFTER_RIVER, riverPlace(call));
        category(base, meldInt, MeldInt.KAN_AFTER_RIVER, riverPlace(kan));
        flag(base, meldInt, MeldInt.PRESENT, true);
        int values = numbers + token * meldFloat.stride();
        number(values, meldFloat, MeldFloat.PLAYER_PROGRESS, Math.max(0, call));
        number(values, meldFloat, MeldFloat.KAN_PROGRESS, Math.max(0, kan));
      }
    }
  }

  /** クエリの席ごとに、河と面子の各トークンの時点が、その席の立直・副露・槓の時点とどう前後するかを書く。 */
  private void writeHistory(Observation o) {
    int base = stateCategories + schema.historyStart();
    for (int query = 0; query < SEATS; query++) {
      int riichi = scan.riichiEvent(query);
      if (riichi == EventScan.NONE) {
        riichi = o.riichi(query) != RiichiState.NONE ? PublicHistory.UNKNOWN : PublicHistory.ABSENT;
      }
      int open = PublicHistory.ABSENT;
      int kan = PublicHistory.ABSENT;
      for (int index = 0; index < scan.meldCount(query); index++) {
        Meld.Type type = scan.meld(query, index).type();
        if (!type.concealed()) {
          open = Math.max(open, scan.meldEvent(query, index));
        }
        if (type.kan()) {
          int kanEvent = type == Meld.Type.KAKAN ? scan.kakanEvent(query, index) : scan.meldEvent(query, index);
          kan = Math.max(kan, kanEvent);
        }
      }
      int tokens = base + query * ClassicSchema.HISTORY_TOKENS * ClassicSchema.HISTORY_WORDS;
      for (int author = 0; author < SEATS; author++) {
        int relative = Math.floorMod(author - query, SEATS);
        for (int index = 0; index < scan.riverTokens(author); index++) {
          int packed =
              PublicHistory.pack(scan.riverEvent(author, index), PublicHistory.ABSENT, riichi, open, kan, relative);
          history(tokens, author * RIVER + index, packed);
        }
        for (int index = 0; index < scan.meldCount(author); index++) {
          int kakan = scan.kakanEvent(author, index);
          int update = kakan == EventScan.NONE ? PublicHistory.ABSENT : kakan;
          int packed = PublicHistory.pack(scan.meldEvent(author, index), update, riichi, open, kan, relative);
          history(tokens, SEATS * RIVER + author * MELDS + index, packed);
        }
      }
    }
  }

  private void history(int tokens, int token, int packed) {
    int at = tokens + token * ClassicSchema.HISTORY_WORDS;
    categorical[at] = (short) packed;
    categorical[at + 1] = (short) (packed >>> Short.SIZE);
  }

  /** GRP の事前分布と局の始まりの値を書く。局の始まりの点と供託は、この局で立直が成立した席が払った供託を戻して求める。 */
  private void writeBoundary(Observation o, float[] grpPrior) {
    int base = boundaryContext;
    int prior = base + boundary.offset(BoundaryFloat.GRP_RANK);
    for (int index = 0; index < BoundaryFloat.GRP_RANK.count(); index++) {
      numeric[prior + index] = grpPrior == null ? UNIFORM_RANK : grpPrior[index] / BoundaryFloat.GRP_RANK.scale();
    }
    if (grpPrior == null) {
      return;
    }
    int kyotaku = o.kyotaku();
    for (int seat = 0; seat < SEATS; seat++) {
      boolean deposited = o.riichi(seat) != RiichiState.NONE;
      roundStartScores[o.absoluteSeat(seat)] = o.score(seat) + (deposited ? Variant.DEPOSIT : 0);
      kyotaku -= deposited ? 1 : 0;
    }
    number(base, boundary, BoundaryFloat.KYOKU_INDEX, o.kyoku());
    number(base, boundary, BoundaryFloat.HONBA, o.honba());
    number(base, boundary, BoundaryFloat.KYOTAKU, kyotaku);
    int scores = base + boundary.offset(BoundaryFloat.SCORE);
    for (int seat = 0; seat < SEATS; seat++) {
      numeric[scores + seat] = roundStartScores[seat] / BoundaryFloat.SCORE.scale();
    }
    number(base, boundary, BoundaryFloat.PRESENT, 1);
  }

  /** 同じ牌を打つ合法手どうしで、立直でない打牌と立直の打牌の位置を対応づけて書く。代表は立直でない打牌で、無ければ立直の打牌。 */
  private void writeRoutes(Observation o) {
    Arrays.fill(damaSlots, -1);
    Arrays.fill(riichiSlots, -1);
    for (int action = 0; action < o.legalActions(); action++) {
      Action legal = o.action(action);
      int identity = VocabularyCodec.discardIdentity(legal);
      if (identity >= 0) {
        (legal.type() == Action.Type.DAHAI ? damaSlots : riichiSlots)[identity] = action;
      }
    }
    for (int action = 0; action < o.legalActions(); action++) {
      int identity = VocabularyCodec.discardIdentity(o.action(action));
      if (identity < 0) {
        continue;
      }
      int dama = damaSlots[identity];
      int riichi = riichiSlots[identity];
      int representative = dama >= 0 ? dama : riichi;
      int base = actionRoutes + action * actionRoute.stride();
      flag(base, actionRoute, ActionRoute.DISCARD_IDENTITY_REPRESENTATIVE, action == representative);
      category(base, actionRoute, ActionRoute.DISCARD_REPRESENTATIVE_SLOT, representative + 1);
      category(base, actionRoute, ActionRoute.DAMA_SLOT, dama + 1);
      category(base, actionRoute, ActionRoute.RIICHI_SLOT, riichi + 1);
    }
  }

  private void writeActions(Observation o) {
    boolean ronLegal = false;
    for (int action = 0; action < o.legalActions(); action++) {
      ronLegal |= o.action(action).type() == Action.Type.RON_AGARI;
    }
    long ownRiver = seats.riverKinds(0);
    int callDiscard = 0;
    int previousKey = NO_KEY;
    for (int action = 0; action < o.legalActions(); action++) {
      Action legal = o.action(action);
      ActionType type = VocabularyCodec.actionType(legal.type());
      RiichiState riichi = resultingRiichi(o, type);
      Meld meld = current.hand().meldOf(o, legal);
      writeAction(o, action, legal, type, riichi, meld);
      RonFuritenKind ronFuriten = ronFuriten(o, type, ronLegal, meld);
      long called = calledIntoMeld(o, type, meld);
      int transitions = Buckets.transitions(o, action);
      for (int transition = 0; transition < transitions; transition++) {
        Action discard =
            switch (type) {
              case DAHAI, RIICHI_DAHAI -> legal;
              case CHI, PON -> o.callDiscard(callDiscard + transition);
              default -> null;
            };
        long river = discard == null ? ownRiver : ownRiver | Tiles.bit(discard.tile());
        int key = discardKey(type, legal);
        if (key == NO_KEY || key != previousKey) {
          after.play(current, legal, meld, discard, riichi);
          after.analyze(o, tiles, river);
        }
        previousKey = key;
        int index = action * transitionsPerAction + transition;
        writeTransition(o, index, type, discard, river, ronFuriten, called);
      }
      if (type == ActionType.CHI || type == ActionType.PON) {
        callDiscard += transitions;
      }
    }
  }

  private void writeAction(Observation o, int action, Action legal, ActionType type, RiichiState riichi, Meld meld) {
    int base = actionCategories + action * actionInt.stride();
    category(base, actionInt, ActionInt.ID, VocabularyCodec.actionIndex(legal) + 1);
    category(base, actionInt, ActionInt.TYPE, type.ordinal() + 1);
    category(base, actionInt, ActionInt.GROUP, type.group().ordinal() + 1);
    category(base, actionInt, ActionInt.PRIMARY_TILE, legal.tile() + 1);
    category(base, actionInt, ActionInt.TILE_SELECTION, VocabularyCodec.tileSelection(legal.selection()).ordinal() + 1);
    if (type == ActionType.CHI) {
      category(base, actionInt, ActionInt.CHI_BASE, legal.base() + 1);
      category(base, actionInt, ActionInt.CHI_CALLED_POSITION, legal.tile() - legal.base() + 1);
    }
    category(base, actionInt, ActionInt.DISCARD_IDENTITY, VocabularyCodec.discardIdentity(legal) + 1);
    category(
        base,
        actionInt,
        ActionInt.RESULTING_MELD_AKA_SOURCE,
        meld == null ? 0 : VocabularyCodec.meldAkaSource(meld.aka()).ordinal() + 1);
    category(base, actionInt, ActionInt.RESULTING_RIICHI_STATUS, VocabularyCodec.riichiStatus(riichi).ordinal() + 1);
    category(base, actionInt, ActionInt.WIN_CONTEXT, winConditions(o, type).ordinal() + 1);
    int win = o.winOf(action);
    if (win >= 0) {
      flag(base, actionInt, ActionInt.URA_ELIGIBLE, o.winUraEligible(win));
      category(
          base,
          actionInt,
          ActionInt.SETTLEMENT_ASSUMPTION,
          VocabularyCodec.settlement(o.winSettlement(win)).ordinal() + 1);
      category(base, actionInt, ActionInt.SOLE_WIN_PROJECTED_RANK, o.winRank(win, 0) + 1);
    }
    flag(base, actionInt, ActionInt.STARTS_IPPATSU, type == ActionType.RIICHI_DAHAI);
    flag(base, actionInt, ActionInt.BREAKS_IPPATSU, type.createsMeld());
    category(base, actionInt, ActionInt.FOLLOW_UP_KIND, type.next().ordinal() + 1);
    flag(base, actionInt, ActionInt.TERMINAL, type.next() == NextStep.ROUND_COMPLETE);
    if (has(Tensor.ACTION_NUMERICS)) {
      writeActionNumbers(o, action, type, win);
    }
    if (has(Tensor.ACTION_WIN_FACTS) && win >= 0) {
      int facts = actionWinFacts + action * winFact.stride();
      flag(facts, winFact, WinFact.VALID, true);
      category(facts, winFact, WinFact.HAN_WITHOUT_URA, o.winHan(win));
      category(facts, winFact, WinFact.FU_CODE, o.winFuCode(win));
      category(facts, winFact, WinFact.YAKUMAN_MULTIPLIER, o.winYakuman(win));
      flag(facts, winFact, WinFact.REQUIRES_PAO_CORRECTION, o.winPao(win));
    }
  }

  /** 立直の宣言で払う供託と、その場で和了したときの点・点の移動・和了の後の点差を書く。 */
  private void writeActionNumbers(Observation o, int action, ActionType type, int win) {
    int base = actionNumerics + action * actionFloat.stride();
    int cost = type == ActionType.RIICHI_DAHAI ? Variant.DEPOSIT : 0;
    int self = o.score(0) - cost;
    int highest = self;
    int lowest = self;
    for (int seat = 1; seat < SEATS; seat++) {
      highest = Math.max(highest, o.score(seat));
      lowest = Math.min(lowest, o.score(seat));
    }
    number(base, actionFloat, ActionFloat.RIICHI_DECLARATION_COST, cost);
    number(base, actionFloat, ActionFloat.SELF_SCORE_AFTER_DECLARATION, self);
    number(base, actionFloat, ActionFloat.SCORE_TO_FIRST_AFTER_DECLARATION, self - highest);
    number(base, actionFloat, ActionFloat.SCORE_TO_FOURTH_AFTER_DECLARATION, self - lowest);
    if (win < 0) {
      return;
    }
    int han = o.winHan(win);
    int fu = decodeFu(o.winFuCode(win));
    number(base, actionFloat, ActionFloat.NORMALIZED_VISIBLE_HAN_WITHOUT_URA, han);
    number(base, actionFloat, ActionFloat.NORMALIZED_VISIBLE_FU, fu);
    number(base, actionFloat, ActionFloat.NORMALIZED_VISIBLE_BASE_POINTS, ScoreMath.base(han, fu, o.winYakuman(win)));
    number(base, actionFloat, ActionFloat.NORMALIZED_URA_INDICATOR_COUNT, o.winUraIndicators(win));
    for (int seat = 0; seat < SEATS; seat++) {
      afterWin[seat] = o.score(seat) + o.winPayment(win, seat);
      number(base, actionFloat, paymentField(seat), o.winPayment(win, seat));
    }
    if (o.winSettlement(win) == SettlementAssumption.SOLE_RON) {
      int bonus = variant.ronHonba() * o.honba();
      int payer = o.lastEventSeat();
      for (int seat = 1; seat < SEATS; seat++) {
        if (seat != payer && o.winPayment(win, seat) < 0) {
          payer = seat;
        }
      }
      afterWin[0] += bonus;
      afterWin[payer] -= bonus;
    }
    afterWin[0] += Variant.DEPOSIT * o.kyotaku();
    for (int seat = 0; seat < SEATS; seat++) {
      number(base, actionFloat, gapField(seat), afterWin[0] - afterWin[seat]);
    }
  }

  private void writeTransition(
      Observation o, int index, ActionType type, Action discard, long river, RonFuritenKind ronFuriten, long called) {
    int base = transitionCategories + index * transitionInt.stride();
    category(base, transitionInt, ActionTransitionInt.KIND, transitionKind(type, discard).ordinal() + 1);
    category(base, transitionInt, ActionTransitionInt.DISCARD_CONTEXT, discardContext(o, type, discard).ordinal() + 1);
    if (discard != null) {
      int dama = VocabularyCodec.actionIndex(dama(discard));
      category(base, transitionInt, ActionTransitionInt.DISCARD_ACTION_ID, dama + 1);
      category(base, transitionInt, ActionTransitionInt.DISCARD_TILE, discard.tile() + 1);
      category(
          base,
          transitionInt,
          ActionTransitionInt.TILE_SELECTION,
          VocabularyCodec.tileSelection(discard.selection()).ordinal() + 1);
    }
    flag(base, transitionInt, ActionTransitionInt.RESULTING_DISCARD_FURITEN, (after.waits() & river) != 0);
    category(base, transitionInt, ActionTransitionInt.RESULTING_RON_FURITEN_KIND, ronFuriten.ordinal() + 1);
    flag(base, transitionInt, ActionTransitionInt.SPECIAL_HANDS_AVAILABLE, after.special());
    flag(base, transitionInt, ActionTransitionInt.PRESENT, true);
    writeTransitionNumbers(index, discard);
    writeTransitionTiles(index, discard, called);
    writeWaits(index);
  }

  private void writeTransitionNumbers(int index, Action discard) {
    int base = transitionNumerics + index * transitionFloat.stride();
    long improving = after.improving();
    long waits = after.waits();
    int acceptance = tiles.live(improving);
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_MIN_SHANTEN, after.shanten() + 1);
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_NORMAL_SHANTEN, after.standardShanten() + 1);
    if (after.special()) {
      number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_CHIITOI_SHANTEN, after.chiitoitsuShanten() + 1);
      number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_KOKUSHI_SHANTEN, after.kokushiShanten() + 1);
    }
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_UKEIRE_COUNT, acceptance);
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_UKEIRE_KINDS, OwnHand.liveKinds(tiles, improving));
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_SHAPE_WAIT_KINDS, Long.bitCount(waits));
    int unseen = tiles.totalUnseen();
    float once = hit(acceptance, unseen, 1);
    float twice = hit(acceptance, unseen, 2);
    float thrice = hit(acceptance, unseen, 3);
    number(base, transitionFloat, ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW, once);
    number(base, transitionFloat, ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS, twice);
    number(base, transitionFloat, ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS, thrice);
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_LIVE_SHAPE_WAIT_COPIES, tiles.live(waits));
    kindsAndCopies(
        base,
        after.intrinsicRon(),
        ActionTransitionFloat.NORMALIZED_INTRINSIC_RON_YAKU_WAIT_KINDS,
        ActionTransitionFloat.NORMALIZED_INTRINSIC_RON_YAKU_WAIT_COPIES);
    kindsAndCopies(
        base,
        after.intrinsicTsumo(),
        ActionTransitionFloat.NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_KINDS,
        ActionTransitionFloat.NORMALIZED_INTRINSIC_TSUMO_YAKU_WAIT_COPIES);
    kindsAndCopies(
        base,
        after.freeRon(),
        ActionTransitionFloat.NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS,
        ActionTransitionFloat.NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_COPIES);
    kindsAndCopies(
        base,
        after.furitenRon(),
        ActionTransitionFloat.NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_KINDS,
        ActionTransitionFloat.NORMALIZED_NEXT_FURITEN_ONLY_RON_YAKU_TENPAI_UKEIRE_COPIES);
    kindsAndCopies(
        base,
        after.tsumoReach(),
        ActionTransitionFloat.NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_KINDS,
        ActionTransitionFloat.NORMALIZED_NEXT_TSUMO_YAKU_TENPAI_UKEIRE_COPIES);
    OwnHand hand = after.hand();
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_DORA_COUNT, hand.dora(tiles));
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_AKA_TILE_COUNT, hand.aka());
    number(
        base,
        transitionFloat,
        ActionTransitionFloat.NORMALIZED_PRIMARY_TILE_UNSEEN_COPIES,
        discard == null ? 0 : tiles.unseen(discard.tile()));
    number(base, transitionFloat, ActionTransitionFloat.MENZEN, hand.menzen() ? 1 : 0);
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_MELD_COUNT, hand.meldCount());
    number(base, transitionFloat, ActionTransitionFloat.NORMALIZED_CONCEALED_TILE_COUNT, hand.concealed());
  }

  /** 遷移の後の手について、牌の種類ごとのコードを書く。 */
  private void writeTransitionTiles(int index, Action discard, long called) {
    int base = transitionTiles + index * Tiles.KINDS;
    long improving = after.improving();
    long waits = after.waits();
    long ronWaits = after.ronWaits();
    long tsumoWaits = after.tsumoWaits();
    long redWins = waits & tiles.unseenRed();
    int discarded = discard == null ? TurnEvent.NONE : discard.tile();
    for (int kind = 0; kind < Tiles.KINDS; kind++) {
      int flags = 0;
      flags |= Tiles.contains(waits, kind) ? TileFlag.SHAPE_WAIT.bit() : 0;
      flags |= Tiles.contains(ronWaits, kind) ? TileFlag.RON_YAKU_WAIT.bit() : 0;
      flags |= Tiles.contains(tsumoWaits, kind) ? TileFlag.TSUMO_YAKU_WAIT.bit() : 0;
      flags |= kind == discarded ? TileFlag.DISCARDED.bit() : 0;
      flags |= Tiles.contains(called, kind) ? TileFlag.CALLED_INTO_MELD.bit() : 0;
      flags |= Tiles.contains(redWins, kind) ? TileFlag.AKA_WINNING_TILE.bit() : 0;
      int ukeire = Tiles.contains(improving, kind) ? tiles.unseen(kind) : 0;
      categorical[base + kind] = (short) TileFlag.code(after.hand().count(kind), ukeire, flags);
    }
  }

  /**
   * 遷移の後の手の待ちを、種類の昇順に 13 種類まで書く。待ちごとに、その種類と、ロン・ツモそれぞれの役と点（Nano）か点の事実（Major）を書く。
   */
  private void writeWaits(int index) {
    for (int wait = 0; wait < after.waitCount(); wait++) {
      int slot = index * WAITS + wait;
      categorical[waitTileIds + slot] = (short) (after.waitKind(wait) + 1);
      writeWait(slot, WaitWinType.RON, after.ron(wait));
      writeWait(slot, WaitWinType.TSUMO, after.tsumo(wait));
    }
  }

  /**
   * 1 つの待ちについて、ロンかツモの一方の役・点・点の事実を書く。
   *
   * @param slot 遷移と待ちの通し番号（遷移の番号 × 13 ＋ 待ちの番号）
   * @param winType ロンかツモか
   * @param score その和了の点（役が無ければ基本点 0）
   */
  private void writeWait(int slot, WaitWinType winType, WinScore score) {
    int pair = slot * WIN_TYPES + winType.ordinal();
    if (has(Tensor.WAIT_YAKUS)) {
      long yaku = VocabularyCodec.yaku(score.yaku());
      for (int chunk = 0; chunk < ClassicSchema.YAKU_CHUNKS; chunk++) {
        categorical[waitYakus + pair * ClassicSchema.YAKU_CHUNKS + chunk] = (short) yakuChunk(yaku, chunk);
      }
    }
    if (has(Tensor.WAIT_SCORES)) {
      int values = waitScores + slot * waitFloat.stride();
      if (winType == WaitWinType.RON) {
        number(values, waitFloat, ActionTransitionWaitFloat.RON_NORMALIZED_VISIBLE_HAN, score.han());
        number(values, waitFloat, ActionTransitionWaitFloat.RON_NORMALIZED_FU, score.fu());
        number(values, waitFloat, ActionTransitionWaitFloat.RON_NORMALIZED_BASE_POINTS, score.base());
      } else {
        number(values, waitFloat, ActionTransitionWaitFloat.TSUMO_NORMALIZED_VISIBLE_HAN, score.han());
        number(values, waitFloat, ActionTransitionWaitFloat.TSUMO_NORMALIZED_FU, score.fu());
        number(values, waitFloat, ActionTransitionWaitFloat.TSUMO_NORMALIZED_BASE_POINTS, score.base());
      }
    }
    if (has(Tensor.WAIT_WIN_FACTS) && score.base() > 0) {
      int facts = waitWinFacts + pair * winFact.stride();
      flag(facts, winFact, WinFact.VALID, true);
      category(facts, winFact, WinFact.HAN_WITHOUT_URA, score.han());
      category(facts, winFact, WinFact.FU_CODE, fuCode(score.fu()));
      category(facts, winFact, WinFact.YAKUMAN_MULTIPLIER, score.yakuman());
      flag(facts, winFact, WinFact.REQUIRES_PAO_CORRECTION, after.pao(score));
    }
  }

  /** 合法手で面子に入る牌の種類。チー・ポンの後の打牌では、鳴いた面子の種類を返す。 */
  private long calledIntoMeld(Observation o, ActionType type, Meld meld) {
    if (meld != null) {
      return meld.kinds();
    }
    boolean afterCall = type == ActionType.DAHAI && o.decisionKind() == EngineDecisionKind.POST_CALL_DAHAI;
    return afterCall ? current.hand().lastMeld().kinds() : 0;
  }

  /** 行動の後の自分の立直の状態。立直の打牌で、自分がまだ打牌しておらず局に副露も無ければダブル立直になる。 */
  private static RiichiState resultingRiichi(Observation o, ActionType type) {
    if (type == ActionType.RIICHI_DAHAI) {
      return RoundRules.declaration(o.beforeFirstDiscard(0), o.callMade());
    }
    return o.riichi(0);
  }

  /** 席の立直の状態。宣言牌への応答の判断では、宣言した席は宣言牌で決まる立直をしているものとして扱う。 */
  private RiichiState riichi(Observation o, int seat) {
    RiichiState established = o.riichi(seat);
    return established != RiichiState.NONE ? established : scan.declaration(seat);
  }

  /** 席が立直を宣言して、宣言牌がまだ通っていないか。宣言牌への応答の判断のときだけ真になる。 */
  private boolean pending(Observation o, int seat) {
    return o.riichi(seat) == RiichiState.NONE && scan.riichiIndex(seat) != EventScan.NONE;
  }

  /** 和了の状況。ツモは天和・地和、嶺上、海底、ふつうのツモの順に、ロンは槍槓、河底、ふつうのロンの順に調べる。 */
  private static WinConditions winConditions(Observation o, ActionType type) {
    if (type == ActionType.TSUMO_AGARI) {
      if (o.beforeFirstDiscard(0) && !o.callMade()) {
        return o.dealer() == 0 ? WinConditions.TENHOU : WinConditions.CHIIHOU;
      }
      if (o.lastEventRinshan()) {
        return WinConditions.RINSHAN;
      }
      return o.remainingWall() == 0 ? WinConditions.HAITEI : WinConditions.TSUMO;
    }
    if (type == ActionType.RON_AGARI) {
      if (o.decisionKind() == EngineDecisionKind.CHANKAN) {
        return WinConditions.CHANKAN;
      }
      return o.remainingWall() == 0 ? WinConditions.HOUTEI : WinConditions.RON;
    }
    return WinConditions.NONE;
  }

  /**
   * この手を選んだ後に残る見逃しの振聴。ロンを選べば振聴は残らない。ロンできる局面か、すでに見逃しの振聴なら、立直中は局の終わりまで
   * 残る。立直していなければ、打牌の無い手（見送り・槓）では同巡の間残り、自分が打牌する手では消える。ただし、下家から鳴いた後の
   * 最初の打牌では消えずに残る。
   *
   * @param meld この手が作る面子（作らない手は null）
   */
  private static RonFuritenKind ronFuriten(Observation o, ActionType type, boolean ronLegal, Meld meld) {
    if (type == ActionType.RON_AGARI || !(ronLegal || o.furitenTemporary())) {
      return RonFuritenKind.NONE;
    }
    if (o.riichi(0) != RiichiState.NONE) {
      return RonFuritenKind.RIICHI_PERSISTENT;
    }
    boolean cleared =
        switch (type) {
          case DAHAI, RIICHI_DAHAI -> !RoundRules.afterCallFromShimocha(o);
          case CHI, PON -> !RoundRules.calledFromShimocha(meld);
          default -> false;
        };
    return cleared ? RonFuritenKind.NONE : RonFuritenKind.TEMPORARY;
  }

  private static TransitionKind transitionKind(ActionType type, Action discard) {
    if (discard != null) {
      return TransitionKind.DISCARD;
    }
    return switch (type.next()) {
      case DECISION_COMPLETE -> TransitionKind.IDENTITY;
      case RINSHAN_DRAW -> TransitionKind.RINSHAN_PENDING;
      case ROUND_COMPLETE -> TransitionKind.TERMINAL;
      case IMMEDIATE_DISCARD -> throw new IllegalStateException(type + " has no discard");
    };
  }

  private DiscardContext discardContext(Observation o, ActionType type, Action discard) {
    if (discard == null) {
      return DiscardContext.NONE;
    }
    return switch (type) {
      case CHI -> DiscardContext.AFTER_CHI;
      case PON -> DiscardContext.AFTER_PON;
      default -> {
        if (o.decisionKind() != EngineDecisionKind.POST_CALL_DAHAI) {
          yield DiscardContext.TURN;
        }
        yield current.hand().lastMeld().type() == Meld.Type.CHI ? DiscardContext.AFTER_CHI : DiscardContext.AFTER_PON;
      }
    };
  }

  /** 打牌と立直の打牌について、遷移の後の手を決めるキー（打つ種類、赤か、立直か）。キーが同じなら後の手も同じになる。ほかの手は −1。 */
  private static int discardKey(ActionType type, Action legal) {
    if (type != ActionType.DAHAI && type != ActionType.RIICHI_DAHAI) {
      return NO_KEY;
    }
    int key = legal.tile() << 2;
    key |= legal.selection().red() ? 2 : 0;
    return key | (type == ActionType.RIICHI_DAHAI ? 1 : 0);
  }

  /** 立直の打牌と同じ牌の、立直でない打牌。 */
  private static Action dama(Action discard) {
    if (discard.type() != Action.Type.RIICHI_DAHAI) {
      return discard;
    }
    return Action.of(Action.code(Action.Type.DAHAI, discard.tile(), discard.selection()));
  }

  private static EventType eventType(LastEvent last) {
    if (last == null) {
      return EventType.NONE;
    }
    return switch (last) {
      case DRAW -> EventType.DRAW;
      case DISCARD -> EventType.DISCARD;
      case KAN_ATTEMPT -> EventType.KAN_ATTEMPT;
      case NUKI -> throw new IllegalStateException("The classic input has no nuki");
    };
  }

  /** 河の位置か河の打牌の数に 1 を足し、24 を上限にした値。無い（−1）ときは 0。 */
  private static int riverPlace(int place) {
    return place < 0 ? 0 : Math.min(place + 1, RIVER);
  }

  /** 立直している席の現物。宣言牌がまだ通っていない席も立直として扱う。立直していなければ 0。 */
  private long genbutsu(Observation o, int seat) {
    return riichi(o, seat) != RiichiState.NONE ? seats.safe(seat) : 0;
  }

  private void kindsAndCopies(int base, long kinds, ActionTransitionFloat count, ActionTransitionFloat copies) {
    number(base, transitionFloat, count, Long.bitCount(kinds));
    number(base, transitionFloat, copies, tiles.live(kinds));
  }

  /**
   * 見えない牌から戻さずに何回か引いて、受け入れを 1 枚以上引く確率。受け入れが K 枚、見えない牌が N 枚、ツモが d 回なら
   * 1 − Π_{i<d} (N − K − i) / (N − i)。
   *
   * @param k 受け入れの枚数 K
   * @param n 見えない牌の枚数 N
   * @param draws ツモの回数 d
   * @return 受け入れを 1 枚以上引く確率。K か N が 0 なら 0
   */
  static float hit(int k, int n, int draws) {
    if (k == 0 || n == 0) {
      return 0f;
    }
    double miss = 1;
    for (int draw = 0; draw < Math.min(draws, n); draw++) {
      int left = n - k - draw;
      if (left <= 0) {
        return 1f;
      }
      miss *= left / (double) (n - draw);
    }
    return (float) (1 - miss);
  }

  /** 符のコード（{@link WinFact#FU_CODE}）。0 符は 0、20 符は 1、25 符は 2、30 符以上は 10 で割った値（110 符が上限）。 */
  static int fuCode(int fu) {
    return switch (fu) {
      case 0 -> 0;
      case 20 -> 1;
      case 25 -> 2;
      default -> Math.min(fu, MAX_FU) / FU_UNIT;
    };
  }

  /** 符のコードを符に戻す。 */
  private static int decodeFu(int code) {
    return switch (code) {
      case 0 -> 0;
      case 1 -> 20;
      case 2 -> 25;
      default -> code * FU_UNIT;
    };
  }

  /**
   * 役のビットを下から 14 ビットずつに分け、そのうちの 1 つに 1 を足した値。0 はパディングに使う。
   *
   * @param yaku classic の役の集合（ビットの並び）
   * @param chunk 下から何番目の 14 ビットか
   * @return 1 以上 2^14 以下
   */
  static int yakuChunk(long yaku, int chunk) {
    long mask = (1L << ClassicSchema.YAKU_CHUNK_BITS) - 1;
    return (int) (yaku >>> chunk * ClassicSchema.YAKU_CHUNK_BITS & mask) + 1;
  }

  private static ActionFloat paymentField(int seat) {
    return switch (seat) {
      case 0 -> ActionFloat.NORMALIZED_PAYMENT_FLOOR_SELF;
      case 1 -> ActionFloat.NORMALIZED_PAYMENT_FLOOR_SHIMOCHA;
      case 2 -> ActionFloat.NORMALIZED_PAYMENT_FLOOR_TOIMEN;
      default -> ActionFloat.NORMALIZED_PAYMENT_FLOOR_KAMICHA;
    };
  }

  private static ActionFloat gapField(int seat) {
    return switch (seat) {
      case 0 -> ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_SELF;
      case 1 -> ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_SHIMOCHA;
      case 2 -> ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_TOIMEN;
      default -> ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_KAMICHA;
    };
  }

  private boolean has(Tensor tensor) {
    return tensor.in(architecture);
  }

  /** 整数のフィールドに書く。アーキテクチャに無いフィールドは書かない。値が幅を外れると IllegalStateException を投げる。 */
  private <E extends Enum<E> & CategoricalField> void category(int base, FieldLayout<E> layout, E field, int value) {
    int offset = layout.offset(field);
    if (offset < 0) {
      return;
    }
    if (value < 0 || value >= field.width()) {
      throw new IllegalStateException(field.getDeclaringClass().getSimpleName() + "." + field + " is " + value
          + ", outside [0, " + field.width() + ")");
    }
    categorical[base + offset] = (short) value;
  }

  private <E extends Enum<E> & CategoricalField> void flag(int base, FieldLayout<E> layout, E field, boolean value) {
    category(base, layout, field, value ? 1 : 0);
  }

  /** 実数のフィールドに、量を尺度で割った値を書く。アーキテクチャに無いフィールドは書かない。 */
  private <E extends Enum<E> & NumericField> void number(int base, FieldLayout<E> layout, E field, float value) {
    int offset = layout.offset(field);
    if (offset >= 0) {
      numeric[base + offset] = value / field.scale();
    }
  }
}
