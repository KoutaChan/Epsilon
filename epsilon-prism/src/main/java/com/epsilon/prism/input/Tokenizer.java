package com.epsilon.prism.input;

import com.epsilon.calculate.scoring.RiichiState;
import com.epsilon.calculate.scoring.WinMethod;
import com.epsilon.core.Action;
import com.epsilon.core.GameState;
import com.epsilon.core.Meld;
import com.epsilon.core.Tile;
import com.epsilon.engine.EngineDecisionKind;
import com.epsilon.engine.HanchanProgression;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.InputSchema.ActionRole;
import com.epsilon.prism.input.InputSchema.Axis;
import com.epsilon.prism.input.InputSchema.DiscardExact;
import com.epsilon.prism.input.InputSchema.EventField;
import com.epsilon.prism.input.InputSchema.EventScalar;
import com.epsilon.prism.input.InputSchema.Furiten;
import com.epsilon.prism.input.InputSchema.KeyBlock;
import com.epsilon.prism.input.InputSchema.OpenSuit;
import com.epsilon.prism.input.InputSchema.PublicEvent;
import com.epsilon.prism.input.InputSchema.RelationAttribute;
import com.epsilon.prism.input.InputSchema.RiichiExact;
import com.epsilon.prism.input.InputSchema.RoundField;
import com.epsilon.prism.input.InputSchema.RoundScalar;
import com.epsilon.prism.input.InputSchema.Safety;
import com.epsilon.prism.input.InputSchema.SeatField;
import com.epsilon.prism.input.InputSchema.SeatScalar;
import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.input.InputSchema.TransitionField;
import com.epsilon.prism.input.InputSchema.TransitionScalar;
import com.epsilon.prism.input.InputSchema.WaitClass;
import com.epsilon.prism.input.InputSchema.WinExact;
import com.epsilon.prism.input.InputSchema.Wind;
import com.epsilon.prism.input.InputWriter.TokenWriter;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WaitScoring.WinningTile;
import com.epsilon.prism.observe.Observation;
import java.util.Arrays;
import java.util.Objects;

/**
 * 記録 O から 1 行の網の入力（{@link InputSchema} の全ての欄・数値・厳密な欄・索引・関係の属性・鍵の符号）と、規則事前 p0 の特徴を書く。O の
 * 純関数で、推論・学習・Reviewer で同じ値になる。符号化のスレッドに 1 つ持ち、派生の作業域はこのクラスの中にだけ持つ。書くときに割り当てを
 * しない。
 *
 * <p>トークンは局 1・席 4（観測席から見た相対席の順）・牌 34（正準の色の順）・出来事（公開の順。槓ドラの表示は {@link Observation#kanDoraEvent}
 * の位置）の順に書く。遷移は合法手の順に、代表の手だけ {@link Observation#transitionsOf} の数を書く（チー・ポンは鳴いた後の打牌の順、代表でない
 * 双子は代表の遷移を指す）。色の正準の写しは判断の初めに 1 回決め、牌の軸の全てにかける。
 *
 * <p>後の手は {@link TransitionHands} が作り、同じ後の手は 1 判断で 1 回だけ解析する（{@link ShapeAnalysis}）。待ちの採点と射影は聴牌の遷移
 * だけ行う。
 */
public final class Tokenizer {

  private static final int SEATS = GameState.NUM_PLAYERS;
  private static final int TILES = Tile.NUM_TILE_TYPES;
  private static final int THIRTEEN = 13;
  private static final int NONE = -1;
  private static final Wind[] WINDS = Wind.values();
  private static final Safety[] SAFETIES = Safety.values();
  private static final OpenSuit[] OPEN_SUITS = OpenSuit.values();
  private static final ScoreLadder.Pass[] PASSES = ScoreLadder.Pass.values();
  private static final DiscardExact[] DISCARD_EXACTS = DiscardExact.values();
  private static final WinExact[] WIN_EXACTS = WinExact.values();
  private static final RiichiExact[] RIICHI_EXACTS = RiichiExact.values();
  private static final byte[] ROUND_CODES = {(byte) KeyBlock.roundCode()};
  private static final byte[][] SEAT_CODES = seatCodes();
  private static final byte EMPTY_TILE_CODE = (byte) KeyBlock.tileCode(0, false, null, false);

  private final EventTable events = new EventTable();
  private final TileCounts tiles = new TileCounts();
  private final SuitCanonicalizer suits = new SuitCanonicalizer();
  private final SeatFacts seats = new SeatFacts();
  private final TileSafety safety = new TileSafety();
  private final ScoreLadder ladder = new ScoreLadder();
  private final TransitionHands hands = new TransitionHands();
  private final ShapeAnalysis shapes = new ShapeAnalysis();
  private final WaitScoring waits = new WaitScoring();
  private final WaitProjection projection = new WaitProjection();
  private final WinFacts wins = new WinFacts();
  private final int[] firstTransition = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorShanten = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorUkeire = new int[Limits.MAX_LEGAL_ACTIONS];
  private final float[] riichiExact = new float[Limits.MAX_LEGAL_ACTIONS * RIICHI_EXACTS.length];
  private final float[] declared = new float[DISCARD_EXACTS.length];
  private final byte[] tileCodes = new byte[TILES];
  private boolean thirteen;
  private long currentWaits;
  private long currentUkeire;
  private boolean ronLegal;
  private int passAction;
  private RiichiState selfRiichi;
  private boolean furitenTemporary;

  /**
   * O から 1 行を書き、書き手の行を確定する。
   *
   * @param o 判断の記録
   * @param out 行に付けた書き手
   * @throws IllegalStateException O が規則・エンジンの判定と合わないとき
   */
  public void write(Observation o, InputWriter out) {
    events.load(o);
    tiles.load(o, events);
    suits.load(o, events, tiles);
    seats.load(o, events, tiles);
    safety.load(events);
    ladder.load(o);
    projection.load(o);
    hands.load(o, tiles, events);
    shapes.reset();
    selfRiichi = o.riichi(0);
    furitenTemporary = o.furitenTemporary();
    AfterHand current = hands.current();
    thirteen = current.effectiveTileCount() == THIRTEEN;
    currentWaits = 0L;
    currentUkeire = 0L;
    if (thirteen) {
      shapes.analyze(current, tiles, ShapeAnalysis.CURRENT);
      currentWaits = shapes.waits();
      currentUkeire = shapes.ukeire();
    }
    writeRound(o, out);
    writeSeats(o, out);
    writeTiles(o, out);
    writeEvents(o, out);
    writeActions(o, out);
    out.commit(o.decisionKind());
  }

  /**
   * 直前に {@link #write} した O の色の正準の写しで、物理の牌を正準の牌にする（字牌はそのまま）。補助の頭の教師の牌の軸に、入力と同じ写しを
   * かけるための口。
   *
   * @param physicalTile 物理の牌（0〜33）
   * @return 正準の牌
   */
  public int canonicalTile(int physicalTile) {
    return suits.tile(Objects.checkIndex(physicalTile, TILES));
  }

  private void writeRound(Observation o, InputWriter out) {
    TokenWriter<RoundField> round = out.round();
    round.value(RoundField.DECISION_KIND, o.decisionKind());
    round.value(RoundField.ROUND_WIND, Wind.ofTile(o.roundWindTile()));
    round.id(RoundField.KYOKU, o.kyoku());
    round.count(RoundField.HONBA, Ids.saturated(o.honba(), InputSchema.STICK_SATURATION));
    round.count(RoundField.KYOTAKU, Ids.saturated(o.kyotaku(), InputSchema.STICK_SATURATION));
    round.count(RoundField.DRAWS_LEFT, drawsLeft(o));
    round.count(RoundField.KANS, events.kans());
    round.flag(RoundField.LAST_LIVE_TILE, o.remainingWall() == 0);
    round.flag(RoundField.FIRST_CYCLE, o.firstCycle());
    round.flag(RoundField.CALL_MADE, o.callMade());
    round.flag(RoundField.WEST_POSSIBLE, westPossible(o));
    round.flag(RoundField.FURITEN_DISCARD, (currentWaits & safety.river(0)) != 0L);
    boolean riichi = selfRiichi != RiichiState.NONE;
    round.flag(RoundField.FURITEN_TEMPORARY, furitenTemporary && !riichi);
    round.flag(RoundField.FURITEN_RIICHI, furitenTemporary && riichi);
    round.value(RoundField.LAST_EVENT, o.lastEvent());
    round.id(RoundField.LAST_EVENT_SEAT, o.lastEventSeat());
    round.id(RoundField.LAST_EVENT_TILE, suits.tile(o.lastEventTile()));
    round.flag(RoundField.LAST_EVENT_AKA, o.lastEventAka());
    round.flag(RoundField.LAST_EVENT_RINSHAN, o.lastEventRinshan());
    round.value(RoundField.LAST_KAN_KIND, o.lastKanKind());
    round.flag(RoundField.DEFERRED_DORA, o.deferredDora());
    out.scalar(RoundScalar.WALL, o.remainingWall());
    out.relation(Axis.ROW, 0).id(RelationAttribute.EVENT, events.publicEvents());
  }

  /**
   * 鳴きがないときの自分の残りツモ。次に山から引く席から数えて、自分の番が来る数。手番と鳴いた後の打牌では下家から、応答では打牌した席の下家から、
   * 槍槓では加槓した席の嶺上の 1 枚を山から引いた後のその席の下家から引く。
   */
  private static int drawsLeft(Observation o) {
    int wall = o.remainingWall();
    int ahead;
    switch (o.decisionKind()) {
      case TURN, POST_CALL_DAHAI -> ahead = SEATS;
      case RESPONSE -> ahead = Math.floorMod(-o.lastEventSeat(), SEATS);
      case CHANKAN -> {
        ahead = Math.floorMod(-o.lastEventSeat(), SEATS);
        wall--;
      }
      default -> throw new IllegalStateException("unknown decision kind: " + o.decisionKind());
    }
    return wall < ahead ? 0 : (wall - ahead) / SEATS + 1;
  }

  /** 南 4 局から西 3 局で、トップが 30000 点に届いていない（このままなら半荘が続く）。 */
  private static boolean westPossible(Observation o) {
    int kyoku = o.kyoku();
    if (kyoku < HanchanProgression.SOUTH_END_KYOKU || kyoku >= HanchanProgression.WEST_END_KYOKU) {
      return false;
    }
    int top = Integer.MIN_VALUE;
    for (int seat = 0; seat < SEATS; seat++) {
      top = Math.max(top, o.score(seat));
    }
    return top < HanchanProgression.TOP_THRESHOLD;
  }

  private void writeSeats(Observation o, InputWriter out) {
    int self = o.score(0);
    for (int seat = 0; seat < SEATS; seat++) {
      TokenWriter<SeatField> token = out.seat(seat);
      token.id(SeatField.RELATIVE_SEAT, seat);
      token.value(SeatField.WIND, WINDS[Math.floorMod(seat - o.dealer(), SEATS)]);
      token.id(SeatField.RANK, seats.rank(seat));
      token.value(SeatField.RIICHI, o.riichi(seat));
      token.flag(SeatField.IPPATSU, o.ippatsu(seat));
      token.flag(SeatField.MENZEN, seats.menzen(seat));
      token.flag(SeatField.BEFORE_FIRST_DISCARD, o.beforeFirstDiscard(seat));
      token.count(SeatField.MELDS, events.meldCount(seat));
      token.count(SeatField.KANS, seats.kans(seat));
      token.count(SeatField.DISCARDS, events.discards(seat));
      token.count(SeatField.OPEN_YAKUHAI_HAN, seats.yakuhaiHan(seat));
      token.value(SeatField.OPEN_SUIT, openSuit(seats.openSuit(seat)));
      token.flag(SeatField.OPEN_TANYAO, seats.tanyao(seat));
      token.flag(SeatField.OPEN_TRIPLETS, seats.triplets(seat));
      token.count(SeatField.OPEN_DORA, seats.openDora(seat));
      token.flag(SeatField.NAGASHI, seats.nagashi(seat));
      if (seat != 0) {
        for (ScoreLadder.Pass pass : PASSES) {
          token.value(pass.field(), ladder.grade(seat, pass));
        }
      }
      out.scalar(seat, SeatScalar.SCORE, o.score(seat));
      out.scalar(seat, SeatScalar.GAP_TO_SELF, o.score(seat) - self);
      TokenWriter<RelationAttribute> relation = out.relation(Axis.SEAT, seat);
      relation.id(RelationAttribute.OWNER, seat);
      relation.id(
          RelationAttribute.OWNER_RIICHI_EVENT, events.position(events.riichiDeclaration(seat)));
      relation.id(
          RelationAttribute.OWNER_LAST_DISCARD_EVENT, events.position(events.lastDiscard(seat)));
    }
  }

  private OpenSuit openSuit(int physicalSuit) {
    if (physicalSuit == SeatFacts.MIXED) {
      return null;
    }
    return physicalSuit == Tile.SUIT_HONOR ? OpenSuit.HONORS : OPEN_SUITS[suits.suit(physicalSuit)];
  }

  /**
   * 牌のトークン。枚数の欄は 34 枚の全てに書き、旗・安全・役牌の翻は牌のマスクから当たる牌にだけ書く（書かない欄は 0）。マスクは物理の牌で作り、
   * 正準の牌のマスクに写してから書く。
   */
  private void writeTiles(Observation o, InputWriter out) {
    for (int canonical = 0; canonical < TILES; canonical++) {
      int tile = suits.physical(canonical);
      TokenWriter<TileField> token = out.tile(canonical);
      token.id(TileField.TILE, canonical);
      token.count(TileField.HAND, tiles.hand(tile));
      token.count(TileField.VISIBLE, tiles.visible(tile));
      token.count(TileField.UNSEEN, tiles.unseen(tile));
      token.count(TileField.DORA, tiles.dora(tile));
      out.relation(Axis.TILE, canonical).id(RelationAttribute.TILE, canonical);
    }
    int drawn = o.drawnTile();
    flags(out, TileField.HAND_AKA, TileCounts.akaTiles(o.handAkaMask()));
    flags(out, TileField.DRAWN, drawn < 0 ? 0L : 1L << drawn);
    flags(out, TileField.UNSEEN_AKA, tiles.unseenAka());
    flags(out, TileField.OWN_RIVER, safety.river(0));
    flags(out, TileField.CURRENT_WAIT, currentWaits);
    flags(out, TileField.CURRENT_UKEIRE, currentUkeire);
    for (int seat = 1; seat < SEATS; seat++) {
      for (Safety safe : SAFETIES) {
        for (long rest = suits.mask(safety.mask(seat, safe)); rest != 0L; rest &= rest - 1) {
          int canonical = Long.numberOfTrailingZeros(rest);
          out.tile(canonical).value(TileField.SAFETY, seat - 1, safe);
          out.relation(Axis.TILE, canonical).value(RelationAttribute.SAFETY, seat - 1, safe);
        }
      }
    }
    int roundWind = o.roundWindTile();
    for (int seat = 0; seat < SEATS; seat++) {
      int seatWind = SeatFacts.seatWindTile(o, seat);
      for (int honor = Tile.TON; honor < TILES; honor++) {
        int han = SeatFacts.yakuhai(honor, seatWind, roundWind);
        if (han != 0) {
          out.tile(suits.tile(honor)).count(TileField.YAKUHAI, seat, han);
        }
      }
    }
  }

  /** 物理の牌のマスクに当たる牌のトークンの旗を立てる。 */
  private void flags(InputWriter out, TileField field, long physicalTiles) {
    for (long rest = suits.mask(physicalTiles); rest != 0L; rest &= rest - 1) {
      out.tile(Long.numberOfTrailingZeros(rest)).flag(field, true);
    }
  }

  private void writeEvents(Observation o, InputWriter out) {
    int count = events.publicEvents();
    for (int position = 0; position < count; position++) {
      TokenWriter<EventField> token = out.event(position);
      TokenWriter<RelationAttribute> relation = out.relation(Axis.EVENT, position);
      relation.id(RelationAttribute.EVENT, position);
      out.scalar(position, EventScalar.AGE, count - 1 - position);
      int e = events.entity(position);
      if (e < 0) {
        int kan = -1 - e;
        int indicator = suits.tile(o.indicatorTile(kan + 1));
        token.value(EventField.KIND, PublicEvent.DORA_INDICATOR);
        token.id(EventField.TILE, indicator);
        relation.id(RelationAttribute.TILE, indicator);
        relation.id(RelationAttribute.LINK, events.position(events.kanEvent(kan)));
        continue;
      }
      int author = events.author(e);
      int tile = suits.tile(events.tile(e));
      int declaration = events.riichiDeclaration(author);
      token.value(EventField.KIND, events.kind(e));
      token.id(EventField.AUTHOR, author);
      token.id(EventField.TILE, tile);
      token.flag(EventField.AUTHOR_IN_RIICHI, declaration >= 0 && declaration < e);
      relation.id(RelationAttribute.TILE, tile);
      relation.id(RelationAttribute.OWNER, author);
      relation.id(RelationAttribute.OWNER_RIICHI_EVENT, events.position(declaration));
      relation.id(
          RelationAttribute.OWNER_LAST_DISCARD_EVENT, events.position(events.lastDiscard(author)));
      relation.id(RelationAttribute.LINK, events.position(events.link(e)));
      Meld meld = events.meld(e);
      if (meld == null) {
        token.flag(EventField.AKA, events.aka(e));
        token.flag(EventField.TSUMOGIRI, events.tsumogiri(e));
        token.flag(EventField.RIICHI_DECLARATION, events.riichi(e));
        token.id(EventField.CALLED_BY, events.calledBy(e));
        token.id(EventField.DISCARD_INDEX, events.discardIndex(e));
      } else {
        int offset = meld.relativeSource().playerOffset();
        token.id(EventField.CALLED_TILE, suits.tile(meld.calledTileType()));
        token.id(EventField.SOURCE, offset < 0 ? NONE : (author + offset) % SEATS);
        token.value(EventField.AKA_SOURCE, meld.akaSource());
      }
    }
  }

  private void writeActions(Observation o, InputWriter out) {
    int actions = o.legalActions();
    ronLegal = false;
    passAction = NONE;
    for (int action = 0; action < actions; action++) {
      Action.Type type = o.action(action).type();
      ronLegal |= type == Action.Type.RON_AGARI;
      if (type == Action.Type.PASS) {
        passAction = action;
      }
    }
    boolean response =
        o.decisionKind() == EngineDecisionKind.RESPONSE
            || o.decisionKind() == EngineDecisionKind.CHANKAN;
    int transition = 0;
    for (int action = 0; action < actions; action++) {
      Action chosen = o.action(action);
      int representative = o.representative(action);
      int count = representative == action ? o.transitionsOf(action) : 0;
      if (representative == action) {
        firstTransition[action] = transition;
        for (int index = 0; index < count; index++) {
          writeTransition(o, out, action, chosen, index, transition++);
        }
      }
      ActionRole role = ActionRole.of(chosen.type());
      int base = response && action != passAction ? passAction : NONE;
      if (chosen.type() == Action.Type.RIICHI_DAHAI) {
        base = damaOf(o, chosen);
      }
      out.action(action, role, firstTransition[representative], count, base, representative);
      writeActionExact(o, out, action, chosen, representative);
      if (role == ActionRole.DISCARD || role == ActionRole.RIICHI) {
        out.prior(
            action,
            PriorFeatures.shanten(priorShanten[representative]),
            PriorFeatures.ukeire(priorUkeire[representative]));
      }
    }
  }

  private void writeActionExact(
      Observation o, InputWriter out, int action, Action chosen, int representative) {
    switch (chosen.type()) {
      case RON_AGARI, TSUMO_AGARI -> {
        wins.load(o, o.winOf(action));
        for (WinExact field : WIN_EXACTS) {
          out.exact(action, field, wins.value(field));
        }
      }
      case RIICHI_DAHAI -> {
        int at = representative * RIICHI_EXACTS.length;
        for (RiichiExact field : RIICHI_EXACTS) {
          out.exact(action, field, riichiExact[at + field.ordinal()]);
        }
      }
      default -> {}
    }
  }

  /** 立直 x の基準のダマ x（打牌の牌と赤・ツモ切りの選びが同じ DAHAI）。 */
  private static int damaOf(Observation o, Action riichi) {
    int identity = riichi.discardIdentityIndex();
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      Action candidate = o.action(action);
      if (candidate.type() == Action.Type.DAHAI && candidate.discardIdentityIndex() == identity) {
        return action;
      }
    }
    throw new IllegalStateException("riichi discard has no dama discard: " + riichi);
  }

  private void writeTransition(
      Observation o, InputWriter out, int action, Action chosen, int index, int transition) {
    Action.Type type = chosen.type();
    AfterHand after = hands.move(action, index);
    Action discard = hands.discard();
    int discarded = discard == null ? NONE : discard.tileType();
    TokenWriter<TransitionField> token = out.transition(transition);
    token.value(TransitionField.ACTION, type);
    token.id(TransitionField.MAIN_TILE, suits.tile(chosen.tileType()));
    token.value(TransitionField.TILE_SELECTION, chosen.tileSelection());
    if (discard != chosen && discard != null) {
      token.id(TransitionField.CALL_DISCARD, suits.tile(discarded));
      token.flag(TransitionField.CALL_DISCARD_AKA, discard.usesAkaTileFromHand());
    }
    token.flag(TransitionField.MENZEN, after.isMenzen());
    token.count(TransitionField.DORA, after.dora());
    long ukeire = 0L;
    long waitMask = 0L;
    Stance stance = Stance.DAMA;
    if (TransitionHands.terminal(type)) {
      writeWin(o, token, action, type);
    } else {
      shapes.analyze(after, tiles, hands.slot());
      ukeire = shapes.ukeire();
      waitMask = shapes.waits();
      token.id(TransitionField.SHANTEN, shapes.minimum());
      token.id(TransitionField.STANDARD_SHANTEN, shapes.standard());
      if (shapes.special()) {
        token.id(TransitionField.CHIITOI_SHANTEN, shapes.chiitoi());
        token.id(TransitionField.KOKUSHI_SHANTEN, shapes.kokushi());
      }
      token.count(TransitionField.UKEIRE_KINDS, Long.bitCount(ukeire));
      token.count(TransitionField.WAIT_KINDS, Long.bitCount(waitMask));
      Furiten furiten = furiten(type, discard, waitMask);
      token.value(TransitionField.FURITEN, furiten);
      int live = shapes.liveUkeire();
      out.scalar(transition, TransitionScalar.LIVE_UKEIRE, live);
      out.scalar(transition, TransitionScalar.HIT_WITHIN_1, shapes.hitWithin(1));
      out.scalar(transition, TransitionScalar.HIT_WITHIN_2, shapes.hitWithin(2));
      out.scalar(transition, TransitionScalar.HIT_WITHIN_3, shapes.hitWithin(3));
      if (index == 0) {
        priorShanten[action] = shapes.minimum();
        priorUkeire[action] = live;
      }
      if (waitMask != 0L) {
        RiichiState riichi = transitionRiichi(o, type, selfRiichi);
        stance = riichi == RiichiState.NONE ? Stance.DAMA : Stance.RIICHI;
        waits.score(
            after,
            waitMask,
            tiles,
            SeatFacts.seatWindTile(o, 0),
            o.roundWindTile(),
            riichi);
        out.scalar(
            transition,
            TransitionScalar.LIVE_RON_YAKU_WAITS,
            liveYakuWaits(WinMethod.RON));
        out.scalar(
            transition,
            TransitionScalar.LIVE_TSUMO_YAKU_WAITS,
            liveYakuWaits(WinMethod.TSUMO));
        boolean ronBlocked = furiten == Furiten.DISCARD || furiten == Furiten.RIICHI_PERSISTENT;
        boolean declares = type == Action.Type.RIICHI_DAHAI;
        projection.project(waits, stance, declares, ronBlocked, tiles, waitMask);
        for (DiscardExact field : DISCARD_EXACTS) {
          float value = projection.value(field);
          out.exact(transition, field, value);
          declared[field.ordinal()] = value;
        }
        if (declares) {
          projection.project(waits, Stance.DAMA, false, ronBlocked, tiles, waitMask);
          int at = action * RIICHI_EXACTS.length;
          for (RiichiExact field : RIICHI_EXACTS) {
            riichiExact[at + field.ordinal()] =
                declared[field.source().ordinal()] - projection.value(field.source());
          }
        }
      } else if (type == Action.Type.RIICHI_DAHAI) {
        // エンジンは形の向聴 0 で立直を許す。待ちが全て 5 枚目の牌（4 枚使いの単騎など）なら和了の道はなく、射影も差も 0
        int at = action * RIICHI_EXACTS.length;
        Arrays.fill(riichiExact, at, at + RIICHI_EXACTS.length, 0f);
      }
    }
    writeCodes(o, out, after, transition, type, discarded, ukeire, waitMask, stance);
  }

  /** 和了の遷移の欄（O の即時の和了から）。九種九牌は書かない。 */
  private static void writeWin(
      Observation o, TokenWriter<TransitionField> token, int action, Action.Type type) {
    if (type == Action.Type.KYUSHU_KYUHAI) {
      return;
    }
    int win = o.winOf(action);
    token.count(TransitionField.WIN_HAN, Ids.saturated(o.winHan(win), InputSchema.HAN_SATURATION));
    token.count(TransitionField.WIN_FU, o.winFuCode(win));
    token.count(TransitionField.WIN_YAKUMAN, o.winYakuman(win));
    token.count(
        TransitionField.URA_INDICATORS, o.winUraEligible(win) ? o.winUraIndicators(win) : 0);
  }

  /**
   * 遷移の後の振聴。待ちが自分の河（捨てる牌を含む）にあれば捨て牌の振聴。自分が打牌する遷移では同巡の見逃しが消え、立直後の見逃しだけが残る。
   * 打牌しない遷移（見送り・槓）では、今の見逃しと、ロンを見送ること（ロンが合法で和了しない）で見逃しの振聴になる。
   */
  private Furiten furiten(Action.Type type, Action discard, long waitMask) {
    long river = safety.river(0);
    if (discard != null) {
      river |= 1L << discard.tileType();
    }
    if ((waitMask & river) != 0L) {
      return Furiten.DISCARD;
    }
    boolean riichi = selfRiichi != RiichiState.NONE;
    if (discard != null) {
      return riichi && furitenTemporary ? Furiten.RIICHI_PERSISTENT : null;
    }
    if (furitenTemporary || ronLegal && type != Action.Type.RON_AGARI) {
      return riichi ? Furiten.RIICHI_PERSISTENT : Furiten.TEMPORARY;
    }
    return null;
  }

  /** 遷移の立直の状態（立直の打牌は第一打の前で鳴きがなければダブル立直、ほかは判断の時点の状態 current）。 */
  private static RiichiState transitionRiichi(Observation o, Action.Type type, RiichiState current) {
    if (type == Action.Type.RIICHI_DAHAI) {
      return o.beforeFirstDiscard(0) && !o.callMade()
          ? RiichiState.DOUBLE_RIICHI
          : RiichiState.RIICHI;
    }
    return current;
  }

  /** 立直なしで役がある待ちの生きた枚数。 */
  private int liveYakuWaits(WinMethod method) {
    long yaku = 0L;
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      if (waits.available(wait, Stance.DAMA, method, WinningTile.PLAIN)) {
        yaku |= 1L << waits.waitTile(wait);
      }
    }
    return tiles.liveCopies(yaku);
  }

  /** 遷移の鍵の符号（局 1・席 4・牌 34 の区画ごとにまとめて書く）。 */
  private void writeCodes(
      Observation o,
      InputWriter out,
      AfterHand after,
      int transition,
      Action.Type type,
      int discarded,
      long ukeire,
      long waitMask,
      Stance stance) {
    out.codes(transition, KeyBlock.ROUND, ROUND_CODES);
    int source =
        switch (type) {
          case CHI, PON, DAIMINKAN, RON_AGARI -> o.lastEventSeat();
          default -> NONE;
        };
    out.codes(transition, KeyBlock.SEAT, SEAT_CODES[source + 1]);
    // 手にない・受け入れでない・待ちでない・捨てない牌の符号は全て同じなので、当たる牌だけを作る。
    Arrays.fill(tileCodes, EMPTY_TILE_CODE);
    long tilesWithCode = after.concealedTileTypeMask() | ukeire | waitMask;
    if (discarded >= 0) {
      tilesWithCode |= 1L << discarded;
    }
    for (long rest = tilesWithCode; rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      long bit = 1L << tile;
      WaitClass wait = (waitMask & bit) != 0L ? waitClass(tile, stance) : null;
      tileCodes[suits.tile(tile)] =
          (byte)
              KeyBlock.tileCode(after.count(tile), (ukeire & bit) != 0L, wait, tile == discarded);
    }
    out.codes(transition, KeyBlock.TILE, tileCodes);
  }

  /** 席の区画の鍵の符号を、この手の元の席（なしは −1）ごとに作る（添字は元の席 + 1）。 */
  private static byte[][] seatCodes() {
    byte[][] codes = new byte[SEATS + 1][SEATS];
    for (int source = NONE; source < SEATS; source++) {
      for (int seat = 0; seat < SEATS; seat++) {
        codes[source + 1][seat] = (byte) KeyBlock.seatCode(seat, seat == source);
      }
    }
    return codes;
  }

  /** 待ちの類（その遷移の立直の構えのロンの翻、赤でない和了牌）。ロンに役がなくツモにあればツモだけ、どちらにもなければ待ちでない。 */
  private WaitClass waitClass(int tile, Stance stance) {
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      if (waits.waitTile(wait) != tile) {
        continue;
      }
      if (waits.available(wait, stance, WinMethod.RON, WinningTile.PLAIN)) {
        int han = waits.han(wait, stance, WinMethod.RON, WinningTile.PLAIN);
        if (han == 0 || han >= 6) {
          return WaitClass.HAN_6_PLUS;
        }
        return han >= 3 ? WaitClass.HAN_3_5 : WaitClass.HAN_1_2;
      }
      return waits.available(wait, stance, WinMethod.TSUMO, WinningTile.PLAIN)
          ? WaitClass.TSUMO_ONLY
          : null;
    }
    throw new IllegalStateException("tile " + tile + " is not a scored wait");
  }
}
