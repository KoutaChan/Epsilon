package com.epsilon.prism.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.RuleFlag;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.input.InputSchema.OpenSuit;

/** 各列が共有する、1 回の判断の公開情報と手牌の解析結果。 */
final class RowFacts {
  private static final int NONE = -1;
  Observation observation;
  final Variant variant;
  final int seatCount;
  final int firstSeat;
  final int firstTile;
  final int firstEvent;
  final int[] prefix;
  final int[] declaration;
  final int[] lastDiscard;
  final int tileCount;
  final TileTokens tokens;
  final boolean nuki;
  final boolean chi;
  final OpenSuit[] openSuits;
  final EventTable events;
  final TileCounts tiles;
  final SeatFacts seatFacts;
  final SuitCanonicalizer suits;
  final PublicSeatFacts seats;
  final TileSafety safety;
  final ScoreLadder ladder;
  final TransitionHands hands = new TransitionHands();
  final ShapeAnalysis shapes;
  final WaitScoring waits;
  private final WaitProjection projection = new WaitProjection(InputSchema.UTILITY);
  final WinFacts wins = new WinFacts(InputSchema.UTILITY);
  private int[] tokenOf;
  int[] physicalOf;
  boolean thirteen;
  private boolean projectionReady;

  /**
   * 前の判断の行を写した記録・公開の位置・副露のイベントの数（全てを書くときは 0）。公開の時点で確定する列は、これより後の位置だけを
   * 書く（前の位置は写した行にある）。
   */
  int fromRecord;

  int fromPosition;
  int fromMeld;

  /** この判断で確定した記録の数（最後の記録より前）と、その公開の位置の数。 */
  int settledRecords;

  int settledPositions;

  long currentWaits;
  long currentUkeire;
  RiichiState selfRiichi;
  boolean furitenTemporary;

  RowFacts(InputFormat schema) {
    variant = schema.variant();
    seatCount = variant.seats();
    firstSeat = schema.block(InputSchema.KeyBlock.SEAT).firstKey();
    firstTile = schema.block(InputSchema.KeyBlock.TILE).firstKey();
    firstEvent = schema.fixedTokens();
    prefix = new int[seatCount];
    declaration = new int[seatCount];
    lastDiscard = new int[seatCount];
    tokens = schema.tileTokens();
    tileCount = tokens.count();
    nuki = variant.has(RuleFlag.NUKI);
    chi = variant.has(RuleFlag.CHI);
    openSuits =
        tokens.movable(Tiles.Group.MAN)
            ? new OpenSuit[] {OpenSuit.FIRST, OpenSuit.SECOND, OpenSuit.THIRD}
            : new OpenSuit[] {OpenSuit.MANZU, OpenSuit.FIRST, OpenSuit.SECOND};
    events = new EventTable(variant);
    tiles = new TileCounts(variant);
    seatFacts = new SeatFacts(variant);
    suits = new SuitCanonicalizer(variant);
    seats = new PublicSeatFacts(seatCount);
    safety = new TileSafety(seatCount);
    ladder = new ScoreLadder(variant);
    shapes = new ShapeAnalysis(variant);
    waits = new WaitScoring(variant);
    tokenOf = suits.tileOrder().tokenOf;
    physicalOf = suits.tileOrder().physicalOf;
  }

  void load(Observation o) {
    load(o, 0);
  }

  /**
   * 判断の事実を作る。
   *
   * @param fromRecord 前の判断の行を写す記録の数（{@link EventHistory#match}。全てを書くなら 0）
   */
  void load(Observation o, int fromRecord) {
    observation = o;
    if (o.variant() != variant) {
      throw new IllegalArgumentException(
          "Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    this.fromRecord = fromRecord;
    events.load(o, tiles, seatFacts, fromRecord);
    fromPosition =
        fromRecord == 0
            ? 0
            : fromRecord == events.events() ? events.publicEvents() : events.position(fromRecord);
    fromMeld = 0;
    while (fromMeld < events.meldEntries() && events.meldEntry(fromMeld) < fromRecord) fromMeld++;
    settledRecords = Math.max(fromRecord, events.events() - 1);
    settledPositions =
        settledRecords == events.events() ? events.publicEvents() : events.position(settledRecords);
    for (int seat = 0; seat < seatCount; seat++) {
      declaration[seat] = events.position(events.riichiDeclaration(seat));
      lastDiscard[seat] = events.position(events.lastDiscard(seat));
    }
    suits.load(o, events, tiles);
    tokenOf = suits.tileOrder().tokenOf;
    physicalOf = suits.tileOrder().physicalOf;
    seats.load(o, events, seatFacts, tiles);
    safety.load(events, seatFacts);
    ladder.load(o);
    projectionReady = false;
    hands.load(o, tiles, seatFacts);
    shapes.reset();
    selfRiichi = o.riichi(0);
    furitenTemporary = o.furitenTemporary();
    AfterHand current = hands.current();
    thirteen = current.effectiveTileCount() == Variant.HAND;
    currentWaits = 0L;
    currentUkeire = 0L;
    if (thirteen) {
      shapes.analyze(current, tiles, ShapeAnalysis.CURRENT);
      currentWaits = shapes.waits();
      currentUkeire = shapes.ukeire();
    }
  }

  WaitProjection projection() {
    if (!projectionReady) {
      projection.load(observation);
      projectionReady = true;
    }
    return projection;
  }

  /** 物理の牌の種類を、正準化した牌のトークンに変える。負の値（牌なし）には「なし」を返す。 */
  int token(int physicalTile) {
    if (physicalTile < 0) {
      return NONE;
    }
    int token = tokenOf[physicalTile];
    if (token < 0) {
      throw new IllegalStateException(variant + " has no tile kind " + physicalTile);
    }
    return token;
  }

  /**
   * この先に鳴きと北抜きがないときの、自分の残りツモの回数。次に山から引く席から数えて、自分の番が何回来るかを数える。手番と鳴いた後の打牌
   * では下家が次に引き、他家の応答では打牌した席の下家が次に引く。槍槓と、抜いた北への応答では、宣言した席が嶺上牌を 1 枚引いて山が 1 枚
   * 減った後に、その席の下家が次に引く。
   */
  int drawsLeft(Observation o) {
    EngineDecisionKind kind = o.decisionKind();
    int ahead =
        switch (kind) {
          case TURN, POST_CALL_DAHAI -> seatCount;
          case RESPONSE, CHANKAN, NUKI_RESPONSE -> Math.floorMod(-o.lastEventSeat(), seatCount);
        };
    int wall = o.remainingWall();
    if (kind == EngineDecisionKind.CHANKAN || kind == EngineDecisionKind.NUKI_RESPONSE) {
      wall--;
    }
    return wall < ahead ? 0 : (wall - ahead) / seatCount + 1;
  }

  /** 南場の最後の局から、西場の最後の局の 1 つ前までで、トップが返し点に届いていない（このままなら半荘が続く）。 */
  boolean westPossible(Observation o) {
    int kyoku = o.kyoku();
    if (kyoku < variant.lastRound() || kyoku >= variant.finalRound()) {
      return false;
    }
    int top = Integer.MIN_VALUE;
    for (int seat = 0; seat < seatCount; seat++) {
      top = Math.max(top, o.score(seat));
    }
    return top < variant.returnPoints();
  }

  OpenSuit openSuit(int physicalSuit) {
    if (physicalSuit == PublicSeatFacts.MIXED) {
      return null;
    }
    return physicalSuit == PublicSeatFacts.HONORS
        ? OpenSuit.HONORS
        : openSuits[suits.suit(physicalSuit)];
  }

  long drawnMask() {
    int drawn = observation.drawnTile();
    return drawn < 0 ? 0L : Tiles.bit(drawn);
  }

  void yakuhai(InputColumn column) {
    int roundWind = observation.roundWindTile();
    for (int seat = 0; seat < seatCount; seat++) {
      int seatWind = PublicSeatFacts.seatWindTile(observation, seat);
      for (int honor = Tiles.TON; honor < Tiles.KINDS; honor++) {
        int han = PublicSeatFacts.yakuhai(honor, seatWind, roundWind);
        if (han != 0) column.count(token(honor), seat, han);
      }
    }
  }

  private static final TileSafety.Safety[] SAFETIES = TileSafety.Safety.values();

  void safety(InputColumn column, int first) {
    for (int seat = 1; seat < seatCount; seat++) {
      for (var grade : SAFETIES) {
        column.values(this, first, seat - 1, grade, safety.mask(seat, grade));
      }
    }
  }

  int eventTile(int position) {
    int e = events.entity(position);
    return token(e < 0 ? observation.indicatorTile(-e) : events.tile(e));
  }

  void ownerEvents(InputColumn column, int[] positions) {
    for (int seat = 0; seat < seatCount; seat++) column.id(firstSeat + seat, positions[seat]);
    for (int e = 0; e < events.events(); e++)
      column.id(firstEvent + events.position(e), positions[events.author(e)]);
  }

  /** 公開の位置 {@link #fromPosition} から後のイベントの物理牌を書く。 */
  void eventTiles(InputColumn column, int first) {
    for (int pos = fromPosition; pos < events.publicEvents(); pos++) {
      int e = events.entity(pos);
      int kind = e < 0 ? observation.indicatorTile(-e) : events.tile(e);
      column.id(first + pos, tokens.token(kind));
    }
  }

  /** 席のトークンに今の位置を、記録 {@link #fromRecord} から後のイベントに、そのイベントまでの位置を書く。 */
  void prefixEvents(InputColumn column, boolean riichi) {
    int[] finalPositions = riichi ? declaration : lastDiscard;
    for (int seat = 0; seat < seatCount; seat++) {
      column.id(firstSeat + seat, finalPositions[seat]);
      prefix[seat] =
          events.position(riichi ? events.riichiBefore(seat) : events.lastDiscardBefore(seat));
    }
    for (int e = fromRecord; e < events.events(); e++) {
      int seat = events.author(e);
      int position = events.position(e);
      if (riichi ? events.riichi(e) : events.kind(e) == EventTable.PublicEvent.DISCARD)
        prefix[seat] = position;
      column.id(firstEvent + position, prefix[seat]);
    }
  }
}
