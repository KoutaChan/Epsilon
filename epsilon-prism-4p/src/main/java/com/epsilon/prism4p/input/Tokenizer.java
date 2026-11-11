package com.epsilon.prism4p.input;

import com.epsilon.agent.facts.SeatFacts;
import com.epsilon.agent.facts.TileCounts;
import com.epsilon.agent.facts.WaitScores.Method;
import com.epsilon.agent.facts.WaitScores.WinningTile;
import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.action.Meld;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.FourPlayerFormat;
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.EventTable;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputBlob;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.PriorFeatures;
import com.epsilon.prism.input.PublicSeatFacts;
import com.epsilon.prism.input.RowInputWriter.TokenWriter;
import com.epsilon.prism.input.ScoreLadder;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.SuitCanonicalizer;
import com.epsilon.prism.input.TileSafety;
import com.epsilon.prism.input.TileSafety.Safety;
import com.epsilon.prism.input.TransitionHands;
import com.epsilon.prism.input.WaitProjection;
import com.epsilon.prism.input.WaitScoring;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WinFacts;
import com.epsilon.prism4p.input.InputSchema.ActionRole;
import com.epsilon.prism4p.input.InputSchema.DiscardExact;
import com.epsilon.prism4p.input.InputSchema.EventField;
import com.epsilon.prism4p.input.InputSchema.EventScalar;
import com.epsilon.prism4p.input.InputSchema.Furiten;
import com.epsilon.prism4p.input.InputSchema.KeyBlock;
import com.epsilon.prism4p.input.InputSchema.OpenSuit;
import com.epsilon.prism4p.input.InputSchema.RelationAttribute;
import com.epsilon.prism4p.input.InputSchema.RiichiExact;
import com.epsilon.prism4p.input.InputSchema.RoundField;
import com.epsilon.prism4p.input.InputSchema.RoundScalar;
import com.epsilon.prism4p.input.InputSchema.SeatField;
import com.epsilon.prism4p.input.InputSchema.SeatScalar;
import com.epsilon.prism4p.input.InputSchema.TileField;
import com.epsilon.prism4p.input.InputSchema.TransitionField;
import com.epsilon.prism4p.input.InputSchema.TransitionScalar;
import com.epsilon.prism4p.input.InputSchema.WaitClass;
import com.epsilon.prism4p.input.InputSchema.WinExact;
import com.epsilon.prism4p.input.InputSchema.Wind;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の記録から、ネットワークの入力 1 行と、ルールベースの事前分布の特徴を書く。入力 1 行は {@link InputSchema} の全てのフィールド・
 * 数値・厳密なフィールド・インデックス・関係の属性・キーのコードからなる。値は判断の記録だけで決まるので、推論・学習・Reviewer で同じ値になる。
 * 符号化のスレッドごとに 1 つ持つ。途中の計算に使う作業領域はこのクラスの中だけに持ち、書くときにメモリを割り当てない。
 *
 * <p>トークンは局 1・席 4・牌 34・イベントの順に書く。席は観測した席から見た相対席の順、牌は正準化した色の順に並べる。イベントは公開された順に
 * 並べ、槓ドラの表示は {@link Observation#kanDoraEvent} の位置に置く。遷移は合法手の順に書き、代表の手だけが
 * {@link Observation#transitionsOf} の数の遷移を持つ。チー・ポンの遷移は鳴いた後の打牌の順に並び、代表でない同じ手は代表の遷移を指す。
 * 色の並べ替えは判断の初めに 1 回決め、牌の軸の全てにかける。
 *
 * <p>遷移の後の手は {@link TransitionHands} が作る。同じ後の手は 1 判断で 1 回だけ解析する（{@link ShapeAnalysis}）。待ちの採点と射影は、
 * 聴牌する遷移だけで行う。牌の枚数と席ごとの公開情報は agent の {@link TileCounts}・{@link SeatFacts} を使う。PRISM だけで使う事実は、
 * prism の入力の部品（{@link PublicSeatFacts}・{@link ScoreLadder}・{@link WaitProjection}・{@link WinFacts}）が持つ。
 */
public final class Tokenizer {

  private static final int SEATS = FourPlayerFormat.SEATS;
  private static final int TILES = Tiles.KINDS;
  private static final int THIRTEEN = 13;
  private static final int NONE = -1;
  private static final Wind[] WINDS = Wind.values();
  private static final Safety[] SAFETIES = Safety.values();
  private static final OpenSuit[] OPEN_SUITS = OpenSuit.values();
  private static final ScoreLadder.Pass[] PASSES = ScoreLadder.Pass.values();
  private static final SeatField[] PASS_FIELDS = passFields();
  private static final DiscardExact[] DISCARD_EXACTS = DiscardExact.values();
  private static final WinExact[] WIN_EXACTS = WinExact.values();
  private static final RiichiExact[] RIICHI_EXACTS = RiichiExact.values();
  private static final byte[] ROUND_CODES = {(byte) KeyBlock.roundCode()};
  private static final byte[][] SEAT_CODES = seatCodes();
  private static final byte EMPTY_TILE_CODE = (byte) KeyBlock.tileCode(0, false, null, false);
  private static final WaitClass[] WAIT_CLASSES = WaitClass.values();
  private static final byte[] TILE_CODES = tileCodes();

  private final Variant variant;
  private final EventTable events;
  private final TileCounts tiles;
  private final SeatFacts seatFacts;
  private final SuitCanonicalizer suits;
  private final PublicSeatFacts seats = new PublicSeatFacts(SEATS);
  private final TileSafety safety = new TileSafety(SEATS);
  private final ScoreLadder ladder;
  private final TransitionHands hands = new TransitionHands();
  private final ShapeAnalysis shapes;
  private final WaitScoring waits;
  private final WaitProjection projection = new WaitProjection(InputSchema.UTILITY);
  private final WinFacts wins = new WinFacts(InputSchema.UTILITY);
  private final int[] firstTransition = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorShanten = new int[Limits.MAX_LEGAL_ACTIONS];
  private final int[] priorUkeire = new int[Limits.MAX_LEGAL_ACTIONS];
  private final float[] riichiExact = new float[Limits.MAX_LEGAL_ACTIONS * RIICHI_EXACTS.length];
  private final float[] declared = new float[DISCARD_EXACTS.length];
  private final byte[] tileCodes = new byte[TILES];
  private boolean thirteen;
  private long currentWaits;
  private long currentUkeire;
  private boolean waitOffered;
  private int passAction;
  private RiichiState selfRiichi;
  private boolean furitenTemporary;

  public Tokenizer() {
    variant = FourPlayerFormat.VARIANT;
    events = new EventTable(variant);
    tiles = new TileCounts(variant);
    seatFacts = new SeatFacts(variant);
    suits = new SuitCanonicalizer(variant);
    ladder = new ScoreLadder(variant);
    shapes = new ShapeAnalysis(variant);
    waits = new WaitScoring(variant);
  }

  /**
   * 判断の記録から 1 行を書き、ライターの行を確定する。
   *
   * @param o 判断の記録
   * @param out 書き込む行のライター
   * @throws IllegalStateException 判断の記録がルールやエンジンの判定と合わないとき
   */
  public void write(Observation o, InputWriter out) {
    if (o.variant() != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + o.variant());
    }
    events.load(o, tiles, seatFacts);
    suits.load(o, events, tiles);
    seats.load(o, events, seatFacts, tiles);
    safety.load(events, seatFacts);
    ladder.load(o);
    projection.load(o);
    hands.load(o, tiles, seatFacts);
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
   * 直前に {@link #write} した判断の記録の色の並べ替えで、物理の牌を正準の牌にする（字牌はそのまま）。補助のヘッドの教師の牌の軸にも、入力と同じ
   * 並べ替えをかけるために使う。
   *
   * @param physicalTile 物理の牌の種類
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
   * 鳴きが入らないときに、自分があと何回ツモれるかを返す。次に山から引く席から順に数える。手番と鳴いた後の打牌では、下家が次に引く。
   * 打牌への応答では、打牌した席の下家が次に引く。槍槓では、加槓した席が嶺上牌を引くので山が 1 枚減り、その席の下家が次に引く。
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

  /** 南場の最後の局から西場の最後の局の手前までにいて、トップが返し点に届いていないか（このまま終われば半荘が続く）。 */
  private static boolean westPossible(Observation o) {
    Variant variant = o.variant();
    int kyoku = o.kyoku();
    if (kyoku < variant.lastRound() || kyoku >= variant.finalRound()) {
      return false;
    }
    int top = Integer.MIN_VALUE;
    for (int seat = 0; seat < SEATS; seat++) {
      top = Math.max(top, o.score(seat));
    }
    return top < variant.returnPoints();
  }

  private void writeSeats(Observation o, InputWriter out) {
    int self = o.score(0);
    for (int seat = 0; seat < SEATS; seat++) {
      TokenWriter<SeatField> token = out.seat(seat);
      token.id(SeatField.RELATIVE_SEAT, seat);
      token.value(SeatField.WIND, WINDS[Math.floorMod(seat - o.dealer(), SEATS)]);
      token.id(SeatField.RANK, seatFacts.rank(seat));
      token.value(SeatField.RIICHI, o.riichi(seat));
      token.flag(SeatField.IPPATSU, o.ippatsu(seat));
      token.flag(SeatField.MENZEN, seatFacts.openMelds(seat) == 0);
      token.flag(SeatField.BEFORE_FIRST_DISCARD, o.beforeFirstDiscard(seat));
      token.count(SeatField.MELDS, seatFacts.meldCount(seat));
      token.count(SeatField.KANS, seatFacts.kans(seat));
      token.count(SeatField.DISCARDS, events.discards(seat));
      token.count(SeatField.OPEN_YAKUHAI_HAN, seats.yakuhaiHan(seat));
      token.value(SeatField.OPEN_SUIT, openSuit(seats.openSuit(seat)));
      token.flag(SeatField.OPEN_TANYAO, seats.tanyao(seat));
      token.flag(SeatField.OPEN_TRIPLETS, seats.triplets(seat));
      token.count(SeatField.OPEN_DORA, seats.openDora(seat));
      token.flag(SeatField.NAGASHI, seats.nagashi(seat));
      if (seat != 0) {
        for (ScoreLadder.Pass pass : PASSES) {
          token.value(PASS_FIELDS[pass.ordinal()], ladder.grade(seat, pass));
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
    if (physicalSuit == PublicSeatFacts.MIXED) {
      return null;
    }
    return physicalSuit == PublicSeatFacts.HONORS ? OpenSuit.HONORS : OPEN_SUITS[suits.suit(physicalSuit)];
  }

  /**
   * 牌のトークンを書く。枚数のフィールドは 34 種の全ての牌に書く。フラグ・安全度・役牌の翻は、マスクに入っている牌にだけ書く（書かない
   * フィールドは 0）。マスクは物理の牌で作り、正準の牌のマスクに直してから書く。
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
    flags(out, TileField.HAND_AKA, tiles.handRed());
    flags(out, TileField.DRAWN, drawn < 0 ? 0L : Tiles.bit(drawn));
    flags(out, TileField.UNSEEN_AKA, tiles.unseenRed());
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
      int seatWind = PublicSeatFacts.seatWindTile(o, seat);
      for (int honor = Tiles.TON; honor < TILES; honor++) {
        int han = PublicSeatFacts.yakuhai(honor, seatWind, roundWind);
        if (han != 0) {
          out.tile(suits.tile(honor)).count(TileField.YAKUHAI, seat, han);
        }
      }
    }
  }

  /** 物理の牌のマスクに入っている牌について、そのトークンのフラグを立てる。 */
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
        Meld.RelativeSource source = meld.source();
        token.id(EventField.CALLED_TILE, suits.tile(meld.called()));
        token.id(
            EventField.SOURCE, source == Meld.RelativeSource.NONE ? NONE : source.seat(author, SEATS));
        token.value(EventField.AKA_SOURCE, meld.aka());
      }
    }
  }

  private void writeActions(Observation o, InputWriter out) {
    int actions = o.legalActions();
    passAction = NONE;
    for (int action = 0; action < actions; action++) {
      Action.Type type = o.action(action).type();
      if (type == Action.Type.PASS) {
        passAction = action;
      }
    }
    boolean response =
        o.decisionKind() == EngineDecisionKind.RESPONSE
            || o.decisionKind() == EngineDecisionKind.CHANKAN;
    waitOffered = response && (currentWaits & Tiles.bit(o.lastEventTile())) != 0L;
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
          out.exact(action, field, winValue(field));
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

  /** ロン・ツモのゲートのヘッドへ渡す厳密なフィールドの値を、直前に読んだ和了の事実から求める。 */
  private float winValue(WinExact field) {
    return switch (field) {
      case ENDS -> flag(wins.ends());
      case UTILITY -> wins.utility();
      case RANK_1 -> flag(wins.rank() == 0);
      case RANK_2 -> flag(wins.rank() == 1);
      case RANK_3 -> flag(wins.rank() == 2);
      case RANK_4 -> flag(wins.rank() == 3);
      case POINTS -> wins.points();
      case RENCHAN -> flag(wins.renchan());
      case PAO -> flag(wins.pao());
      case URA_ELIGIBLE -> flag(wins.uraEligible());
    };
  }

  private static float flag(boolean value) {
    return value ? 1f : 0f;
  }

  /** 牌 x を切る立直の基準にする、ダマで x を切る手を探す。同じ牌を赤・ツモ切りも同じ選び方で切る打牌が、合法手の何番目かを返す。 */
  private static int damaOf(Observation o, Action riichi) {
    int dama = Action.code(Action.Type.DAHAI, riichi.tile(), riichi.selection());
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (o.action(action).code() == dama) {
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
    int discarded = discard == null ? NONE : discard.tile();
    TokenWriter<TransitionField> token = out.transition(transition);
    token.value(TransitionField.ACTION, type);
    token.id(TransitionField.MAIN_TILE, suits.tile(chosen.tile()));
    token.value(TransitionField.TILE_SELECTION, chosen.selection());
    if (discard != chosen && discard != null) {
      token.id(TransitionField.CALL_DISCARD, suits.tile(discarded));
      token.flag(TransitionField.CALL_DISCARD_AKA, discard.selection().red());
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
      Furiten furiten = furiten(discard, waitMask);
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
        waits.score(after, waitMask, tiles, o, PublicSeatFacts.seatWindTile(o, 0), riichi);
        out.scalar(transition, TransitionScalar.LIVE_RON_YAKU_WAITS, liveYakuWaits(Method.RON));
        out.scalar(transition, TransitionScalar.LIVE_TSUMO_YAKU_WAITS, liveYakuWaits(Method.TSUMO));
        boolean ronBlocked = furiten == Furiten.DISCARD || furiten == Furiten.RIICHI_PERSISTENT;
        boolean declares = type == Action.Type.RIICHI_DAHAI;
        projection.project(waits, stance, declares, ronBlocked, tiles, waitMask);
        for (DiscardExact field : DISCARD_EXACTS) {
          float value = projection.value(field.projected());
          out.exact(transition, field, value);
          declared[field.ordinal()] = value;
        }
        if (declares) {
          projection.project(waits, Stance.DAMA, false, ronBlocked, tiles, waitMask);
          int at = action * RIICHI_EXACTS.length;
          for (RiichiExact field : RIICHI_EXACTS) {
            riichiExact[at + field.ordinal()] =
                declared[field.source().ordinal()] - projection.value(field.source().projected());
          }
        }
      } else if (type == Action.Type.RIICHI_DAHAI) {
        // エンジンは形が聴牌（向聴 0）なら立直を許す。待ちが全て 5 枚目の牌（4 枚使いの単騎など）なら和了できないので、射影も差も 0
        int at = action * RIICHI_EXACTS.length;
        Arrays.fill(riichiExact, at, at + RIICHI_EXACTS.length, 0f);
      }
    }
    writeCodes(o, out, after, transition, type, discarded, ukeire, waitMask, stance);
  }

  /** 和了する遷移のフィールドを、判断の記録にあるその場の和了の情報から書く。九種九牌では何も書かない。 */
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
   * 遷移の後の振聴を返す（振聴でなければ null）。待ちが自分の河（この遷移で捨てる牌を含む）にあれば、捨て牌の振聴。自分が打牌する遷移では
   * 同巡の見逃しが消え、立直後の見逃しだけが残る。打牌しない遷移（見送り・槓）では今の見逃しが残る。応答する他家の牌が今の手の形の上の
   * 待ちに入るときも、見逃しの振聴になる（役が無くてロンできない牌も含む）。
   */
  private Furiten furiten(Action discard, long waitMask) {
    long river = safety.river(0);
    if (discard != null) {
      river |= Tiles.bit(discard.tile());
    }
    if ((waitMask & river) != 0L) {
      return Furiten.DISCARD;
    }
    boolean riichi = selfRiichi != RiichiState.NONE;
    if (discard != null) {
      return riichi && furitenTemporary ? Furiten.RIICHI_PERSISTENT : null;
    }
    if (furitenTemporary || waitOffered) {
      return riichi ? Furiten.RIICHI_PERSISTENT : Furiten.TEMPORARY;
    }
    return null;
  }

  /** 遷移の後の立直の状態を返す。立直の打牌は、第一打の前で鳴きがなければダブル立直、ほかは立直。立直以外の手は判断の時点の状態のまま。 */
  private static RiichiState transitionRiichi(Observation o, Action.Type type, RiichiState current) {
    if (type == Action.Type.RIICHI_DAHAI) {
      return o.beforeFirstDiscard(0) && !o.callMade()
          ? RiichiState.DOUBLE_RIICHI
          : RiichiState.RIICHI;
    }
    return current;
  }

  /** 立直しなくても役がある待ちの、生きている枚数。 */
  private int liveYakuWaits(Method method) {
    long yaku = 0L;
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      if (waits.available(wait, Stance.DAMA, method, WinningTile.PLAIN)) {
        yaku |= Tiles.bit(waits.waitTile(wait));
      }
    }
    return tiles.live(yaku);
  }

  /** 遷移のキーのコードを、局 1・席 4・牌 34 の領域ごとにまとめて書く。 */
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
    // 手になく、受け入れでも待ちでもなく、捨てもしない牌のコードは全て同じなので、それ以外の牌のコードだけを作る。
    Arrays.fill(tileCodes, EMPTY_TILE_CODE);
    long tilesWithCode = after.concealedKinds() | ukeire | waitMask;
    if (discarded >= 0) {
      tilesWithCode |= 1L << discarded;
    }
    for (long rest = tilesWithCode; rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      long bit = 1L << tile;
      WaitClass wait = (waitMask & bit) != 0L ? waitClass(tile, stance) : null;
      tileCodes[suits.tile(tile)] =
          TILE_CODES[tileCodeIndex(after.count(tile), (ukeire & bit) != 0L, wait, tile == discarded)];
    }
    out.codes(transition, KeyBlock.TILE, tileCodes);
  }

  /** 牌の領域のキーのコードを、{@link KeyBlock#tileCode} の引数の組み合わせごとに前もって作る（インデックスは {@link #tileCodeIndex}）。 */
  private static byte[] tileCodes() {
    byte[] codes = new byte[tileCodeIndex(Tiles.COPIES + 1, false, null, false)];
    for (int count = 0; count <= Tiles.COPIES; count++) {
      for (int ukeire = 0; ukeire < 2; ukeire++) {
        for (int wait = -1; wait < WAIT_CLASSES.length; wait++) {
          WaitClass waitClass = wait < 0 ? null : WAIT_CLASSES[wait];
          for (int discarded = 0; discarded < 2; discarded++) {
            codes[tileCodeIndex(count, ukeire == 1, waitClass, discarded == 1)] =
                (byte) KeyBlock.tileCode(count, ukeire == 1, waitClass, discarded == 1);
          }
        }
      }
    }
    return codes;
  }

  private static int tileCodeIndex(int afterCount, boolean ukeire, WaitClass wait, boolean discarded) {
    int index = afterCount * 2 + (ukeire ? 1 : 0);
    index = index * (WAIT_CLASSES.length + 1) + (wait == null ? 0 : wait.ordinal() + 1);
    return index * 2 + (discarded ? 1 : 0);
  }

  /** 追い越し方ごとに、席のトークンのどのフィールドに書くかを返す（インデックスは追い越し方の ordinal）。 */
  private static SeatField[] passFields() {
    SeatField[] fields = new SeatField[PASSES.length];
    for (ScoreLadder.Pass pass : PASSES) {
      fields[pass.ordinal()] =
          switch (pass) {
            case BY_DIRECT_RON -> SeatField.PASS_BY_DIRECT_RON;
            case BY_OTHER_RON -> SeatField.PASS_BY_OTHER_RON;
            case BY_TSUMO -> SeatField.PASS_BY_TSUMO;
            case PASSED_BY_RON -> SeatField.PASSED_BY_RON;
            case PASSED_BY_TSUMO -> SeatField.PASSED_BY_TSUMO;
          };
    }
    return fields;
  }

  /** 席の領域のキーのコードを、この手で鳴く・ロンする牌を出した席（無ければ −1）ごとに作る（インデックスはその席 + 1）。 */
  private static byte[][] seatCodes() {
    byte[][] codes = new byte[SEATS + 1][SEATS];
    for (int source = NONE; source < SEATS; source++) {
      for (int seat = 0; seat < SEATS; seat++) {
        codes[source + 1][seat] = (byte) KeyBlock.seatCode(seat, seat == source);
      }
    }
    return codes;
  }

  /**
   * 待ちのクラスを返す。その遷移の構え（立直かダマか）で、赤でない和了牌をロンしたときの翻で分ける（翻 0 は 6 翻以上の側に入れる）。
   * ロンに役がなくツモにあればツモだけのクラス、どちらにも役がなければ null。
   */
  private WaitClass waitClass(int tile, Stance stance) {
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      if (waits.waitTile(wait) != tile) {
        continue;
      }
      if (waits.available(wait, stance, Method.RON, WinningTile.PLAIN)) {
        int han = waits.han(wait, stance, Method.RON, WinningTile.PLAIN);
        if (han == 0 || han >= 6) {
          return WaitClass.HAN_6_PLUS;
        }
        return han >= 3 ? WaitClass.HAN_3_5 : WaitClass.HAN_1_2;
      }
      return waits.available(wait, stance, Method.TSUMO, WinningTile.PLAIN)
          ? WaitClass.TSUMO_ONLY
          : null;
    }
    throw new IllegalStateException("tile " + tile + " is not a scored wait");
  }
}
