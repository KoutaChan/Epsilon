package com.epsilon.prism3p.input;

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
import com.epsilon.prism.Limits;
import com.epsilon.prism.input.AfterHand;
import com.epsilon.prism.input.EventTable;
import com.epsilon.prism.input.EventTable.PublicEvent;
import com.epsilon.prism.input.Ids;
import com.epsilon.prism.input.InputBlob.Axis;
import com.epsilon.prism.input.PriorFeatures;
import com.epsilon.prism.input.PublicSeatFacts;
import com.epsilon.prism.input.RowInputWriter.TokenWriter;
import com.epsilon.prism.input.ScoreLadder;
import com.epsilon.prism.input.ShapeAnalysis;
import com.epsilon.prism.input.SuitCanonicalizer;
import com.epsilon.prism.input.TileSafety;
import com.epsilon.prism.input.TileSafety.Safety;
import com.epsilon.prism.input.TileTokens;
import com.epsilon.prism.input.TransitionHands;
import com.epsilon.prism.input.WaitProjection;
import com.epsilon.prism.input.WaitScoring;
import com.epsilon.prism.input.WaitScoring.Stance;
import com.epsilon.prism.input.WinFacts;
import com.epsilon.prism3p.input.InputSchema.ActionRole;
import com.epsilon.prism3p.input.InputSchema.DiscardExact;
import com.epsilon.prism3p.input.InputSchema.EventField;
import com.epsilon.prism3p.input.InputSchema.EventScalar;
import com.epsilon.prism3p.input.InputSchema.Furiten;
import com.epsilon.prism3p.input.InputSchema.KeyBlock;
import com.epsilon.prism3p.input.InputSchema.OpenSuit;
import com.epsilon.prism3p.input.InputSchema.RelationAttribute;
import com.epsilon.prism3p.input.InputSchema.RiichiExact;
import com.epsilon.prism3p.input.InputSchema.RoundField;
import com.epsilon.prism3p.input.InputSchema.RoundScalar;
import com.epsilon.prism3p.input.InputSchema.SeatField;
import com.epsilon.prism3p.input.InputSchema.SeatScalar;
import com.epsilon.prism3p.input.InputSchema.TileField;
import com.epsilon.prism3p.input.InputSchema.TransitionField;
import com.epsilon.prism3p.input.InputSchema.TransitionScalar;
import com.epsilon.prism3p.input.InputSchema.WaitClass;
import com.epsilon.prism3p.input.InputSchema.WinExact;
import com.epsilon.prism3p.input.InputSchema.Wind;
import java.util.Arrays;
import java.util.Objects;

/**
 * 判断の記録から、三人打ちのネットワークの入力を 1 行書く。書くのは {@link InputSchema} の全てのフィールド（カテゴリ・数値・厳密な値）、
 * インデックス、関係の属性、キーのコードと、ルールベースの事前分布の特徴。判断の記録だけで決まる純関数なので、推論・学習・Reviewer で同じ値に
 * なる。符号化のスレッドごとに 1 つ持ち、途中の計算に使う作業領域はこのクラスの中にだけ持つ。書くときにメモリを確保しない。
 *
 * <p>トークンは、局 1 個、席 3 個（観測席から見た相対席の順）、牌 27 個（色を正準化した後の牌のトークンの順）、イベント（公開された順。
 * 槓ドラの表示は {@link Observation#kanDoraEvent} の位置）の順に書く。遷移は合法手の順に書き、代表の手だけが
 * {@link Observation#transitionsOf} の数の遷移を持つ。ポンの遷移は鳴いた後の打牌の順に並ぶ。遷移が同じになる手のうち、代表でない手は
 * 代表の遷移を指す。色の正準化（筒子と索子の入れ替え）は判断の初めに 1 回決め、牌を表す全ての軸にかけてから牌のトークンの番号にする。
 *
 * <p>遷移後の手は {@link TransitionHands} が作る。同じ形の遷移後の手は、1 つの判断の中で 1 回だけ解析する（{@link ShapeAnalysis}）。
 * 待ちの採点と射影は、聴牌になる遷移だけで行う。点の移動は三人打ちの精算（ツモ損あり）で求め、抜いた北は手のドラと待ちの翻に数える。
 */
public final class Tokenizer {

  private static final Variant VARIANT = ThreePlayerFormat.VARIANT;
  private static final int SEATS = ThreePlayerFormat.SEATS;
  private static final int TILES = ThreePlayerFormat.TILES;
  private static final TileTokens TOKENS = TileTokens.of(VARIANT);
  private static final int NONE = -1;
  private static final Wind[] WINDS = Wind.values();
  private static final Safety[] SAFETIES = Safety.values();

  /** 正準化した色の番号（萬子 0・1 つ目の色 1・2 つ目の色 2）ごとの副露の色。萬子は入れ替えないので番号は 0 のまま。 */
  private static final OpenSuit[] OPEN_SUITS = {OpenSuit.MANZU, OpenSuit.FIRST, OpenSuit.SECOND};

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

  private final EventTable events = new EventTable(VARIANT);
  private final TileCounts tiles = new TileCounts(VARIANT);
  private final SeatFacts seatFacts = new SeatFacts(VARIANT);
  private final SuitCanonicalizer suits = new SuitCanonicalizer(VARIANT);
  private final PublicSeatFacts seats = new PublicSeatFacts(SEATS);
  private final TileSafety safety = new TileSafety(SEATS);
  private final ScoreLadder ladder = new ScoreLadder(VARIANT);
  private final TransitionHands hands = new TransitionHands();
  private final ShapeAnalysis shapes = new ShapeAnalysis(VARIANT);
  private final WaitScoring waits = new WaitScoring(VARIANT);
  private final WaitProjection projection = new WaitProjection(InputSchema.UTILITY);
  private final WinFacts wins = new WinFacts(InputSchema.UTILITY);

  /** 物理の牌の種類ごとの、正準化した牌のトークン。判断ごとに作る。三人打ちで使わない 2〜8 萬は −1 のまま。 */
  private final int[] tokenOf = new int[Tiles.KINDS];

  /** 正準化した牌のトークンごとの、物理の牌の種類。 */
  private final int[] physicalOf = new int[TILES];

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
  private boolean tsumoLegal;
  private int passAction;
  private RiichiState selfRiichi;
  private boolean furitenTemporary;

  public Tokenizer() {
    Arrays.fill(tokenOf, NONE);
  }

  /**
   * 記録から 1 行を書き、ライターの行を確定する。
   *
   * @param o 判断の記録
   * @param out 書く行に結び付けたライター
   * @throws IllegalArgumentException 判断の記録が三人打ちのものでないとき
   * @throws IllegalStateException 判断の記録がルールやエンジンの判定と合わないとき
   */
  public void write(Observation o, InputWriter out) {
    if (o.variant() != VARIANT) {
      throw new IllegalArgumentException("Variant mismatch: expected " + VARIANT + " but found " + o.variant());
    }
    events.load(o, tiles, seatFacts);
    suits.load(o, events, tiles);
    for (int token = 0; token < TILES; token++) {
      int physical = suits.physical(TOKENS.kind(token));
      physicalOf[token] = physical;
      tokenOf[physical] = token;
    }
    seats.load(o, events, seatFacts, tiles);
    safety.load(events, seatFacts);
    ladder.load(o);
    projection.load(o);
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
    writeRound(o, out);
    writeSeats(o, out);
    writeTiles(o, out);
    writeEvents(o, out);
    writeActions(o, out);
    out.commit(o.decisionKind());
  }

  /**
   * 直前に {@link #write} した判断の記録と同じ色の正準化で、物理の牌を正準化した牌のトークンに変える。補助のヘッドの教師の牌の軸に、入力と同じ
   * 正準化をかけるために使う。
   *
   * @param physicalTile 物理の牌の種類
   * @return 正準化した牌のトークンの番号
   * @throws IllegalStateException 三人打ちで使わない牌の種類のとき
   */
  public int canonicalTile(int physicalTile) {
    return token(Objects.checkIndex(physicalTile, Tiles.KINDS));
  }

  /** 物理の牌の種類を、正準化した牌のトークンに変える。負の値（牌なし）には「なし」を返す。 */
  private int token(int physicalTile) {
    if (physicalTile < 0) {
      return NONE;
    }
    int token = tokenOf[physicalTile];
    if (token < 0) {
      throw new IllegalStateException(VARIANT + " has no tile kind " + physicalTile);
    }
    return token;
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
    round.id(RoundField.LAST_EVENT_TILE, token(o.lastEventTile()));
    round.flag(RoundField.LAST_EVENT_AKA, o.lastEventAka());
    round.flag(RoundField.LAST_EVENT_RINSHAN, o.lastEventRinshan());
    round.value(RoundField.LAST_KAN_KIND, o.lastKanKind());
    round.flag(RoundField.DEFERRED_DORA, o.deferredDora());
    out.scalar(RoundScalar.WALL, o.remainingWall());
    out.relation(Axis.ROW, 0).id(RelationAttribute.EVENT, events.publicEvents());
  }

  /**
   * この先に鳴きと北抜きがないときの、自分の残りツモの回数。次に山から引く席から数えて、自分の番が何回来るかを数える。手番と鳴いた後の打牌
   * では下家が次に引き、他家の応答では打牌した席の下家が次に引く。槍槓と、抜いた北への応答では、宣言した席が嶺上牌を 1 枚引いて山が 1 枚
   * 減った後に、その席の下家が次に引く。
   */
  private static int drawsLeft(Observation o) {
    EngineDecisionKind kind = o.decisionKind();
    int ahead =
        switch (kind) {
          case TURN, POST_CALL_DAHAI -> SEATS;
          case RESPONSE, CHANKAN, NUKI_RESPONSE -> Math.floorMod(-o.lastEventSeat(), SEATS);
        };
    int wall = o.remainingWall();
    if (kind == EngineDecisionKind.CHANKAN || kind == EngineDecisionKind.NUKI_RESPONSE) {
      wall--;
    }
    return wall < ahead ? 0 : (wall - ahead) / SEATS + 1;
  }

  /** 南場の最後の局から、西場の最後の局の 1 つ前までで、トップが返し点に届いていない（このままなら半荘が続く）。 */
  private static boolean westPossible(Observation o) {
    int kyoku = o.kyoku();
    if (kyoku < VARIANT.lastRound() || kyoku >= VARIANT.finalRound()) {
      return false;
    }
    int top = Integer.MIN_VALUE;
    for (int seat = 0; seat < SEATS; seat++) {
      top = Math.max(top, o.score(seat));
    }
    return top < VARIANT.returnPoints();
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
      token.count(SeatField.NUKI, o.nuki(seat));
      token.count(SeatField.DISCARDS, events.discards(seat));
      token.count(SeatField.OPEN_YAKUHAI_HAN, seats.yakuhaiHan(seat));
      token.value(SeatField.OPEN_SUIT, openSuit(seats.openSuit(seat)));
      token.flag(SeatField.OPEN_TANYAO, seats.tanyao(seat));
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
      relation.id(RelationAttribute.OWNER_RIICHI_EVENT, events.position(events.riichiDeclaration(seat)));
      relation.id(RelationAttribute.OWNER_LAST_DISCARD_EVENT, events.position(events.lastDiscard(seat)));
    }
  }

  private OpenSuit openSuit(int physicalSuit) {
    if (physicalSuit == PublicSeatFacts.MIXED) {
      return null;
    }
    return physicalSuit == PublicSeatFacts.HONORS ? OpenSuit.HONORS : OPEN_SUITS[suits.suit(physicalSuit)];
  }

  /**
   * 牌のトークンを書く。枚数のフィールドは 27 個の全てに書き、フラグ・安全度・役牌の翻は、物理の牌のマスクに当たるトークンにだけ書く
   * （書かないフィールドは 0）。北のドラの数は、表示牌の数に抜きドラの 1 を足す。
   */
  private void writeTiles(Observation o, InputWriter out) {
    for (int token = 0; token < TILES; token++) {
      int tile = physicalOf[token];
      TokenWriter<TileField> field = out.tile(token);
      field.id(TileField.TILE, token);
      field.count(TileField.HAND, tiles.hand(tile));
      field.count(TileField.VISIBLE, tiles.visible(tile));
      field.count(TileField.UNSEEN, tiles.unseen(tile));
      field.count(TileField.DORA, tiles.dora(tile) + (tile == Tiles.PEI ? 1 : 0));
      out.relation(Axis.TILE, token).id(RelationAttribute.TILE, token);
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
        for (long rest = safety.mask(seat, safe); rest != 0L; rest &= rest - 1) {
          int token = token(Long.numberOfTrailingZeros(rest));
          out.tile(token).value(TileField.SAFETY, seat - 1, safe);
          out.relation(Axis.TILE, token).value(RelationAttribute.SAFETY, seat - 1, safe);
        }
      }
    }
    int roundWind = o.roundWindTile();
    for (int seat = 0; seat < SEATS; seat++) {
      int seatWind = PublicSeatFacts.seatWindTile(o, seat);
      for (int honor = Tiles.TON; honor < Tiles.KINDS; honor++) {
        int han = PublicSeatFacts.yakuhai(honor, seatWind, roundWind);
        if (han != 0) {
          out.tile(token(honor)).count(TileField.YAKUHAI, seat, han);
        }
      }
    }
  }

  /** 物理の牌のマスクに当たる牌のトークンに、フラグを立てる。 */
  private void flags(InputWriter out, TileField field, long physicalTiles) {
    for (long rest = physicalTiles; rest != 0L; rest &= rest - 1) {
      out.tile(token(Long.numberOfTrailingZeros(rest))).flag(field, true);
    }
  }

  private void writeEvents(Observation o, InputWriter out) {
    int count = events.publicEvents();
    for (int position = 0; position < count; position++) {
      TokenWriter<EventField> field = out.event(position);
      TokenWriter<RelationAttribute> relation = out.relation(Axis.EVENT, position);
      relation.id(RelationAttribute.EVENT, position);
      out.scalar(position, EventScalar.AGE, count - 1 - position);
      int e = events.entity(position);
      if (e < 0) {
        int kan = -1 - e;
        int indicator = token(o.indicatorTile(kan + 1));
        field.value(EventField.KIND, PublicEvent.DORA_INDICATOR);
        field.id(EventField.TILE, indicator);
        relation.id(RelationAttribute.TILE, indicator);
        relation.id(RelationAttribute.LINK, events.position(events.kanEvent(kan)));
        continue;
      }
      int author = events.author(e);
      int tile = token(events.tile(e));
      int declaration = events.riichiDeclaration(author);
      PublicEvent kind = events.kind(e);
      field.value(EventField.KIND, kind);
      field.id(EventField.AUTHOR, author);
      field.id(EventField.TILE, tile);
      field.flag(EventField.AUTHOR_IN_RIICHI, declaration >= 0 && declaration < e);
      relation.id(RelationAttribute.TILE, tile);
      relation.id(RelationAttribute.OWNER, author);
      relation.id(RelationAttribute.OWNER_RIICHI_EVENT, events.position(declaration));
      relation.id(RelationAttribute.OWNER_LAST_DISCARD_EVENT, events.position(events.lastDiscard(author)));
      relation.id(RelationAttribute.LINK, events.position(events.link(e)));
      Meld meld = events.meld(e);
      if (meld != null) {
        Meld.RelativeSource source = meld.source();
        field.id(EventField.CALLED_TILE, token(meld.called()));
        field.id(EventField.SOURCE, source == Meld.RelativeSource.NONE ? NONE : source.seat(author, SEATS));
        field.value(EventField.AKA_SOURCE, meld.aka());
      } else if (kind == PublicEvent.DISCARD) {
        field.flag(EventField.AKA, events.aka(e));
        field.flag(EventField.TSUMOGIRI, events.tsumogiri(e));
        field.flag(EventField.RIICHI_DECLARATION, events.riichi(e));
        field.id(EventField.CALLED_BY, events.calledBy(e));
        field.id(EventField.DISCARD_INDEX, events.discardIndex(e));
      }
    }
  }

  private void writeActions(Observation o, InputWriter out) {
    int actions = o.legalActions();
    tsumoLegal = false;
    passAction = NONE;
    for (int action = 0; action < actions; action++) {
      Action.Type type = o.action(action).type();
      tsumoLegal |= type == Action.Type.TSUMO_AGARI;
      if (type == Action.Type.PASS) {
        passAction = action;
      }
    }
    EngineDecisionKind kind = o.decisionKind();
    boolean response =
        kind == EngineDecisionKind.RESPONSE
            || kind == EngineDecisionKind.CHANKAN
            || kind == EngineDecisionKind.NUKI_RESPONSE;
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

  private void writeActionExact(Observation o, InputWriter out, int action, Action chosen, int representative) {
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

  /** ロン・ツモのゲートに渡すフィールドの値を返す（直前に読み込んだ和了の情報から求める）。 */
  private float winValue(WinExact field) {
    return switch (field) {
      case ENDS -> flag(wins.ends());
      case UTILITY -> wins.utility();
      case RANK_1 -> flag(wins.rank() == 0);
      case RANK_2 -> flag(wins.rank() == 1);
      case RANK_3 -> flag(wins.rank() == 2);
      case POINTS -> wins.points();
      case RENCHAN -> flag(wins.renchan());
      case PAO -> flag(wins.pao());
      case URA_ELIGIBLE -> flag(wins.uraEligible());
    };
  }

  private static float flag(boolean value) {
    return value ? 1f : 0f;
  }

  /** 立直の打牌の基準になるダマの打牌を返す（同じ牌を、同じ赤・ツモ切りの選び方で切る DAHAI）。 */
  private static int damaOf(Observation o, Action riichi) {
    int dama = Action.code(Action.Type.DAHAI, riichi.tile(), riichi.selection());
    for (int action = 0, actions = o.legalActions(); action < actions; action++) {
      if (o.action(action).code() == dama) {
        return action;
      }
    }
    throw new IllegalStateException("riichi discard has no dama discard: " + riichi);
  }

  private void writeTransition(Observation o, InputWriter out, int action, Action chosen, int index, int transition) {
    Action.Type type = chosen.type();
    AfterHand after = hands.move(action, index);
    Action discard = hands.discard();
    int discarded = discard == null ? NONE : discard.tile();
    TokenWriter<TransitionField> field = out.transition(transition);
    field.value(TransitionField.ACTION, type);
    field.id(TransitionField.MAIN_TILE, token(type == Action.Type.NUKI ? Tiles.PEI : chosen.tile()));
    field.value(TransitionField.TILE_SELECTION, chosen.selection());
    if (discard != chosen && discard != null) {
      field.id(TransitionField.CALL_DISCARD, token(discarded));
      field.flag(TransitionField.CALL_DISCARD_AKA, discard.selection().red());
    }
    field.flag(TransitionField.MENZEN, after.isMenzen());
    field.count(TransitionField.DORA, after.dora());
    long ukeire = 0L;
    long waitMask = 0L;
    Stance stance = Stance.DAMA;
    if (TransitionHands.terminal(type)) {
      writeWin(o, field, action, type);
    } else {
      shapes.analyze(after, tiles, hands.slot());
      ukeire = shapes.ukeire();
      waitMask = shapes.waits();
      field.id(TransitionField.SHANTEN, shapes.minimum());
      field.id(TransitionField.STANDARD_SHANTEN, shapes.standard());
      if (shapes.special()) {
        field.id(TransitionField.CHIITOI_SHANTEN, shapes.chiitoi());
        field.id(TransitionField.KOKUSHI_SHANTEN, shapes.kokushi());
      }
      field.count(TransitionField.UKEIRE_KINDS, Long.bitCount(ukeire));
      field.count(TransitionField.WAIT_KINDS, Long.bitCount(waitMask));
      Furiten furiten = furiten(type, discard, waitMask);
      field.value(TransitionField.FURITEN, furiten);
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
        for (DiscardExact exact : DISCARD_EXACTS) {
          float value = projection.value(exact.projected());
          out.exact(transition, exact, value);
          declared[exact.ordinal()] = value;
        }
        if (declares) {
          projection.project(waits, Stance.DAMA, false, ronBlocked, tiles, waitMask);
          int at = action * RIICHI_EXACTS.length;
          for (RiichiExact exact : RIICHI_EXACTS) {
            riichiExact[at + exact.ordinal()] =
                declared[exact.source().ordinal()] - projection.value(exact.source().projected());
          }
        }
      } else if (type == Action.Type.RIICHI_DAHAI) {
        // エンジンは形の上で聴牌なら立直を許す。待ちが全て 5 枚目の牌（4 枚使いの単騎など）なら和了できないので、射影も差も 0
        int at = action * RIICHI_EXACTS.length;
        Arrays.fill(riichiExact, at, at + RIICHI_EXACTS.length, 0f);
      }
    }
    writeCodes(o, out, after, transition, type, discarded, ukeire, waitMask, stance);
  }

  /** 和了の遷移のフィールドを、記録にあるこの手で和了したときの値から書く。九種九牌では何も書かない。 */
  private static void writeWin(Observation o, TokenWriter<TransitionField> field, int action, Action.Type type) {
    if (type == Action.Type.KYUSHU_KYUHAI) {
      return;
    }
    int win = o.winOf(action);
    field.count(TransitionField.WIN_HAN, Ids.saturated(o.winHan(win), InputSchema.HAN_SATURATION));
    field.count(TransitionField.WIN_FU, o.winFuCode(win));
    field.count(TransitionField.WIN_YAKUMAN, o.winYakuman(win));
    field.count(TransitionField.URA_INDICATORS, o.winUraEligible(win) ? o.winUraIndicators(win) : 0);
  }

  /**
   * 遷移の後の振聴を返す（振聴でなければ null）。待ちが自分の河（この手で捨てる牌を含む）にあれば捨て牌の振聴。自分が打牌する遷移では同巡の
   * 見逃しが消え、立直後の見逃しだけが残る。打牌しない遷移（見送り・槓・北抜き）では、今の見逃しが続く。応答する他家の牌が今の手の形の上の
   * 待ちに入るときも、見逃しの振聴になる（役が無くてロンできない牌も含む）。立直中にツモ和了できる北を抜いても、立直後の見逃しになる。
   */
  private Furiten furiten(Action.Type type, Action discard, long waitMask) {
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
    if (furitenTemporary || waitOffered || riichi && tsumoLegal && type == Action.Type.NUKI) {
      return riichi ? Furiten.RIICHI_PERSISTENT : Furiten.TEMPORARY;
    }
    return null;
  }

  /**
   * 遷移の後の立直の状態を返す。立直の打牌は、第一打の前で鳴きも北抜きもなければダブル立直、それ以外は立直。ほかの手は判断の時点の状態の
   * まま。エンジンと同じく、北抜きは第一巡を途切れさせる（記録の鳴きのフラグは北抜きを数えないので、抜いた数を別に見る）。
   */
  private static RiichiState transitionRiichi(Observation o, Action.Type type, RiichiState current) {
    if (type == Action.Type.RIICHI_DAHAI) {
      return o.beforeFirstDiscard(0) && !o.callMade() && !nukiMade(o) ? RiichiState.DOUBLE_RIICHI : RiichiState.RIICHI;
    }
    return current;
  }

  /** この局で誰かが北を抜いた。 */
  private static boolean nukiMade(Observation o) {
    for (int seat = 0; seat < SEATS; seat++) {
      if (o.nuki(seat) > 0) {
        return true;
      }
    }
    return false;
  }

  /** 立直しなくても役がつく待ち牌の、生きている枚数。 */
  private int liveYakuWaits(Method method) {
    long yaku = 0L;
    for (int wait = 0, count = waits.waits(); wait < count; wait++) {
      if (waits.available(wait, Stance.DAMA, method, WinningTile.PLAIN)) {
        yaku |= Tiles.bit(waits.waitTile(wait));
      }
    }
    return tiles.live(yaku);
  }

  /** 遷移のキーのコードを、局 1・席 3・牌 27 の領域ごとにまとめて書く。 */
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
          case PON, DAIMINKAN, RON_AGARI -> o.lastEventSeat();
          default -> NONE;
        };
    out.codes(transition, KeyBlock.SEAT, SEAT_CODES[source + 1]);
    // 手になく、受け入れ牌でも待ち牌でもなく、捨てもしない牌のコードは全て同じなので、それ以外の牌だけを作る。
    Arrays.fill(tileCodes, EMPTY_TILE_CODE);
    long tilesWithCode = after.concealedKinds() | ukeire | waitMask;
    if (discarded >= 0) {
      tilesWithCode |= 1L << discarded;
    }
    for (long rest = tilesWithCode; rest != 0L; rest &= rest - 1) {
      int tile = Long.numberOfTrailingZeros(rest);
      long bit = 1L << tile;
      WaitClass wait = (waitMask & bit) != 0L ? waitClass(tile, stance) : null;
      tileCodes[token(tile)] =
          TILE_CODES[tileCodeIndex(after.count(tile), (ukeire & bit) != 0L, wait, tile == discarded)];
    }
    out.codes(transition, KeyBlock.TILE, tileCodes);
  }

  /** 牌の領域のキーのコードを、{@link KeyBlock#tileCode} の引数の組ごとに前もって作る（添字は {@link #tileCodeIndex}）。 */
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

  /** 追い越し方ごとの席のフィールド（添字は {@code ScoreLadder.Pass} の ordinal）。 */
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

  /** 席の領域のキーのコードを、この手で鳴く・ロンする牌を出した席（なければ −1）ごとに作る（添字はその席 + 1）。 */
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
   * 待ちの分類を返す。その遷移の立直の状態で、赤でない和了牌でロンしたときの翻で分ける。ロンに役がなくツモにあればツモだけ、どちらにも
   * なければ待ちでない（null）。
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
      return waits.available(wait, stance, Method.TSUMO, WinningTile.PLAIN) ? WaitClass.TSUMO_ONLY : null;
    }
    throw new IllegalStateException("tile " + tile + " is not a scored wait");
  }
}
