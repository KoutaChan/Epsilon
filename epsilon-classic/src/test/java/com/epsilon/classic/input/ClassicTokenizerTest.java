package com.epsilon.classic.input;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertThrows;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.fail;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationSchema.EventKind;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.Architecture;
import com.epsilon.classic.RandomPlay;
import com.epsilon.classic.input.Buckets.Bucket;
import com.epsilon.classic.input.ClassicFields.ActionFloat;
import com.epsilon.classic.input.ClassicFields.ActionInt;
import com.epsilon.classic.input.ClassicFields.ActionRoute;
import com.epsilon.classic.input.ClassicFields.ActionTransitionFloat;
import com.epsilon.classic.input.ClassicFields.ActionTransitionInt;
import com.epsilon.classic.input.ClassicFields.ActionTransitionWaitFloat;
import com.epsilon.classic.input.ClassicFields.BoundaryFloat;
import com.epsilon.classic.input.ClassicFields.MeldInt;
import com.epsilon.classic.input.ClassicFields.PlayerInt;
import com.epsilon.classic.input.ClassicFields.RiverInt;
import com.epsilon.classic.input.ClassicFields.RoundInt;
import com.epsilon.classic.input.ClassicFields.TileFlag;
import com.epsilon.classic.input.ClassicFields.TileInt;
import com.epsilon.classic.input.ClassicFields.WinFact;
import com.epsilon.classic.input.ClassicSchema.FieldLayout;
import com.epsilon.classic.input.ClassicSchema.StateToken;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.engine.EngineDecisionKind;
import com.epsilon.mahjong.engine.SeatView;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.RiichiState;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Set;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.testng.annotations.Test;

/**
 * ランダムな自己対局の全ての判断で、全ての席について判断の記録から Nano と Major の両方の入力を書き、次のことを確かめる。例外が
 * 出ない。整数のフィールドは幅の範囲内、実数のフィールドは有限、パディングは 0。実在するトークン・合法手・遷移・待ちは「値 ＋ 1」と
 * {@code PRESENT} の決まりどおり。同じ判断の記録を 2 つの行に書くと同じ値になる。全体を通して、どのフィールドも 1 回は 0 でない値を
 * 持つ（ランダムな対局では 0 のままのこともある、まれなフィールドを除く）。立直を宣言した席は、宣言牌がまだ通っていない判断でも立直と
 * 一発の印を持つ。
 */
public class ClassicTokenizerTest {

  private static final Logger log = LoggerFactory.getLogger(ClassicTokenizerTest.class);
  private static final Variant VARIANT = Variant.TENHOU_4P;
  private static final int STEPS = 100_000;

  /** 0 のままのこともあるフィールド。和了者と自分の点差はいつも 0 で、役満・包・槍槓はランダムな自己対局ではまれ。 */
  private static final Set<InputField> RARE =
      Set.of(
          ActionFloat.NORMALIZED_SOLE_WIN_SCORE_GAP_SELF,
          WinFact.YAKUMAN_MULTIPLIER,
          WinFact.REQUIRES_PAO_CORRECTION,
          RoundInt.EVENT_KAN_KIND);

  private final ObservationWriter writer = new ObservationWriter(VARIANT);
  private final ObservationBatch batch = new ObservationBatch(VARIANT, 1_024);
  private final Observation o = new Observation(VARIANT);
  private final ClassicExtras extras = ClassicExtras.empty();
  private final float[] prior = prior();
  private final List<Check> checks = List.of(new Check(Architecture.NANO), new Check(Architecture.MAJOR));
  private long decisions;
  private long riichiDiscards;
  private long chankans;
  private long callTransitions;
  private long waitingTransitions;
  private long frontierTransitions;
  private long pendingDeclarations;

  @Test(timeOut = 1_800_000)
  public void everyDecisionOfRandomGamesFillsTheInputsWithinRange() {
    RandomPlay play = new RandomPlay(VARIANT, 71);
    for (int step = 0; step < STEPS; step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        visit(play);
      }
      play.step();
    }
    log.info(
        String.format(
            Locale.ROOT,
            "CLASSIC tokens: decisions %d riichiDiscards %d chankans %d callTransitions %d waitingTransitions %d"
                + " frontierTransitions %d pendingDeclarations %d buckets %s",
            decisions,
            riichiDiscards,
            chankans,
            callTransitions,
            waitingTransitions,
            frontierTransitions,
            pendingDeclarations,
            checks.get(0).inputs.keySet()));
    assertTrue(decisions > 50_000, "decisions " + decisions);
    assertTrue(riichiDiscards > 0, "no riichi discard");
    assertTrue(callTransitions > 0, "no call with several discards");
    assertTrue(waitingTransitions > 0, "no transition with waits");
    assertTrue(frontierTransitions > 0, "no transition one draw from a tenpai with yaku");
    assertTrue(pendingDeclarations > 0, "no decision before a riichi declaration passed");
    for (Check check : checks) {
      List<String> empty = new ArrayList<>();
      for (InputField field : check.fields) {
        if (!check.filled.contains(field) && !RARE.contains(field)) {
          empty.add(((Enum<?>) field).getDeclaringClass().getSimpleName() + "." + field);
        }
      }
      assertEquals(empty, List.of(), check.architecture + " fields that stayed 0");
    }
  }

  @Test
  public void rejectsRowsThatDoNotFit() {
    RandomPlay play = new RandomPlay(VARIANT, 5);
    while (play.boundary() != Boundary.DECISIONS || play.engine.decisions().legalCount(0) < 2) {
      play.step();
    }
    Decisions boundary = play.engine.decisions();
    batch.clear();
    o.bind(batch, writer.capture(play.engine.view(boundary.seat(0)), boundary, 0, batch));
    ClassicTokenizer tokenizer = new ClassicTokenizer(Architecture.NANO, VARIANT);
    ClassicInputs small = new ClassicInputs(ClassicSchema.of(Architecture.NANO), new Bucket(1, 1), 1);
    assertThrows(IllegalArgumentException.class, () -> tokenizer.write(o, extras, small, 0));
    ClassicInputs major = new ClassicInputs(ClassicSchema.of(Architecture.MAJOR), Buckets.required(o), 1);
    assertThrows(IllegalArgumentException.class, () -> tokenizer.write(o, extras, major, 0));
    ClassicInputs nano = new ClassicInputs(ClassicSchema.of(Architecture.NANO), Buckets.required(o), 1);
    assertThrows(IllegalArgumentException.class, () -> tokenizer.write(o, extras, new float[3], nano, 0));
    assertThrows(IllegalArgumentException.class, () -> new ClassicTokenizer(Architecture.NANO, Variant.TENHOU_3P));
  }

  private void visit(RandomPlay play) {
    Decisions boundary = play.engine.decisions();
    batch.clear();
    for (int slot = 0; slot < boundary.size(); slot++) {
      SeatView view = play.engine.view(boundary.seat(slot));
      o.bind(batch, writer.capture(view, boundary, slot, batch));
      extras.capture(view);
      boolean withPrior = decisions % 2 == 0;
      for (Check check : checks) {
        check.visit(withPrior);
      }
      count();
      decisions++;
    }
  }

  private void count() {
    chankans += o.decisionKind() == EngineDecisionKind.CHANKAN ? 1 : 0;
    for (int action = 0; action < o.legalActions(); action++) {
      Action.Type type = o.action(action).type();
      riichiDiscards += type == Action.Type.RIICHI_DAHAI ? 1 : 0;
      callTransitions += Buckets.transitions(o, action) > 1 ? 1 : 0;
    }
  }

  /** 一様でない GRP の事前分布（値は 0 以上で有限）。 */
  private static float[] prior() {
    float[] prior = new float[BoundaryFloat.GRP_RANK.count()];
    for (int index = 0; index < prior.length; index++) {
      prior[index] = (index % 5 + 1) / 10f;
    }
    return prior;
  }

  /** Nano か Major の一方の入力を書いて確かめる。 */
  private final class Check {

    private final Architecture architecture;
    private final ClassicSchema schema;
    private final ClassicTokenizer tokenizer;
    private final Map<Bucket, ClassicInputs> inputs = new HashMap<>();
    private final Set<InputField> filled = new HashSet<>();
    private final List<InputField> fields = new ArrayList<>();
    private final InputField[] stateCategories;
    private final InputField[] stateNumerics;
    private final InputField[] actionCategories;
    private final InputField[] actionNumerics;
    private final InputField[] actionRoutes;
    private final InputField[] actionFacts;
    private final InputField[] transitionCategories;
    private final InputField[] transitionNumerics;
    private final InputField[] waitScores;
    private final InputField[] waitFacts;
    private final InputField[] boundary;

    Check(Architecture architecture) {
      this.architecture = architecture;
      schema = ClassicSchema.of(architecture);
      tokenizer = new ClassicTokenizer(architecture, VARIANT);
      List<InputField> categorical = new ArrayList<>();
      List<InputField> numeric = new ArrayList<>();
      for (StateToken token : StateToken.values()) {
        InputField[] categories = expand(token.categorical());
        InputField[] numbers = expand(token.numeric());
        for (int index = 0; index < token.count(); index++) {
          categorical.addAll(Arrays.asList(categories));
          numeric.addAll(Arrays.asList(numbers));
        }
      }
      stateCategories = categorical.toArray(InputField[]::new);
      stateNumerics = numeric.toArray(InputField[]::new);
      actionCategories = expand(ActionInt.class);
      actionNumerics = expand(ActionFloat.class);
      actionRoutes = expand(ActionRoute.class);
      actionFacts = expand(WinFact.class);
      transitionCategories = expand(ActionTransitionInt.class);
      transitionNumerics = expand(ActionTransitionFloat.class);
      waitScores = expand(ActionTransitionWaitFloat.class);
      waitFacts = expand(WinFact.class);
      boundary = expand(BoundaryFloat.class);
      for (Class<? extends Enum<?>> type : ClassicSchema.fieldTypes()) {
        for (Object field : schema.layoutOf(type).fields()) {
          fields.add((InputField) field);
        }
      }
    }

    void visit(boolean withPrior) {
      Bucket bucket = Buckets.required(o);
      ClassicInputs in = inputs.computeIfAbsent(bucket, size -> new ClassicInputs(schema, size, 2));
      for (int row = 0; row < 2; row++) {
        if (withPrior) {
          tokenizer.write(o, extras, prior, in, row);
        } else {
          tokenizer.write(o, extras, in, row);
        }
      }
      for (Tensor tensor : Tensor.values()) {
        int width = in.rowWidth(tensor);
        int first = in.start(tensor, 0);
        int second = in.start(tensor, 1);
        boolean same =
            tensor.slab() == Tensor.Slab.CATEGORICAL
                ? Arrays.equals(in.categorical(), first, first + width, in.categorical(), second, second + width)
                : Arrays.equals(in.numeric(), first, first + width, in.numeric(), second, second + width);
        assertTrue(same, architecture + " " + tensor + " differs between two writes of one O");
      }
      state(in);
      actions(in);
      ledgerAndBoundary(in, withPrior);
    }

    private void state(ClassicInputs in) {
      short[] categorical = in.categorical();
      int base = in.start(Tensor.STATE_CATEGORIES, 0);
      units(categorical, base, stateCategories, true);
      int numbers = in.start(Tensor.STATE_NUMERICS, 0);
      units(in.numeric(), numbers, stateNumerics, true);
      int round = base + schema.stateStart(StateToken.ROUND, Tensor.Slab.CATEGORICAL);
      FieldLayout<RoundInt> roundInt = schema.layout(RoundInt.class);
      assertEquals(categorical[round + roundInt.offset(RoundInt.PLAYER_SEAT)], o.observer() + 1);
      assertEquals(categorical[round + roundInt.offset(RoundInt.KYOKU_INDEX)], o.kyoku() + 1);
      assertEquals(categorical[round + roundInt.offset(RoundInt.DEALER_RELATIVE_SEAT)], o.dealer() + 1);
      assertEquals(categorical[round + roundInt.offset(RoundInt.WALL_REMAINING)], o.remainingWall());
      assertEquals(categorical[round + roundInt.offset(RoundInt.HONBA)], o.honba());
      FieldLayout<PlayerInt> playerInt = schema.layout(PlayerInt.class);
      int players = base + schema.stateStart(StateToken.PLAYER, Tensor.Slab.CATEGORICAL);
      for (int seat = 0; seat < ClassicSchema.SEATS; seat++) {
        int token = players + seat * playerInt.stride();
        assertEquals(categorical[token + playerInt.offset(PlayerInt.RELATIVE_SEAT)], seat + 1);
        assertEquals(categorical[token + playerInt.offset(PlayerInt.ABSOLUTE_SEAT)], o.absoluteSeat(seat) + 1);
        if (!declared(seat)) {
          continue;
        }
        assertTrue(categorical[token + playerInt.offset(PlayerInt.RIICHI_STATUS)] > 1, "a declared seat is in riichi");
        if (o.riichi(seat) == RiichiState.NONE) {
          assertEquals(categorical[token + playerInt.offset(PlayerInt.IPPATSU)], 1, "a pending declaration has ippatsu");
          pendingDeclarations += architecture == Architecture.NANO ? 1 : 0;
        }
      }
      FieldLayout<TileInt> tileInt = schema.layout(TileInt.class);
      int tiles = base + schema.stateStart(StateToken.TILE, Tensor.Slab.CATEGORICAL);
      for (int kind = 0; kind < 34; kind++) {
        int token = tiles + kind * tileInt.stride();
        assertEquals(categorical[token + tileInt.offset(TileInt.TILE_TYPE)], kind + 1);
        assertEquals(categorical[token + tileInt.offset(TileInt.SELF_HAND_COUNT)], o.handCount(kind));
      }
      FieldLayout<RiverInt> riverInt = schema.layout(RiverInt.class);
      int rivers = base + schema.stateStart(StateToken.RIVER, Tensor.Slab.CATEGORICAL);
      for (int seat = 0; seat < ClassicSchema.SEATS; seat++) {
        boolean ended = false;
        for (int index = 0; index < ClassicSchema.RIVER_TOKENS; index++) {
          int token = rivers + (seat * ClassicSchema.RIVER_TOKENS + index) * riverInt.stride();
          boolean present = categorical[token + riverInt.offset(RiverInt.PRESENT)] == 1;
          if (!present) {
            ended = true;
            assertZero(categorical, token, riverInt.stride(), "river token");
            continue;
          }
          assertTrue(!ended, "a river token follows padding");
          assertEquals(categorical[token + riverInt.offset(RiverInt.INDEX)], index + 1);
          assertEquals(categorical[token + riverInt.offset(RiverInt.RELATIVE_PLAYER)], seat + 1);
        }
      }
      FieldLayout<MeldInt> meldInt = schema.layout(MeldInt.class);
      int melds = base + schema.stateStart(StateToken.MELD, Tensor.Slab.CATEGORICAL);
      for (int seat = 0; seat < ClassicSchema.SEATS; seat++) {
        for (int index = 0; index < ClassicSchema.MELD_TOKENS; index++) {
          int token = melds + (seat * ClassicSchema.MELD_TOKENS + index) * meldInt.stride();
          if (categorical[token + meldInt.offset(MeldInt.PRESENT)] == 1) {
            assertEquals(categorical[token + meldInt.offset(MeldInt.INDEX)], index + 1);
          } else {
            assertZero(categorical, token, meldInt.stride(), "meld token");
          }
        }
      }
      if (architecture.publicHistory()) {
        history(categorical, base, rivers, riverInt, melds, meldInt);
      }
    }

    /** 公開の履歴の値を確かめる。値を持つのは実在するトークンだけで、関係は 0〜11、席はクエリの席から見た相対席。 */
    private void history(
        short[] categorical, int base, int rivers, FieldLayout<RiverInt> riverInt, int melds, FieldLayout<MeldInt> meldInt) {
      int river = ClassicSchema.SEATS * ClassicSchema.RIVER_TOKENS;
      for (int query = 0; query < ClassicSchema.SEATS; query++) {
        for (int token = 0; token < ClassicSchema.HISTORY_TOKENS; token++) {
          int at = base + schema.historyStart() + (query * ClassicSchema.HISTORY_TOKENS + token) * 2;
          int packed = categorical[at] & 0xFFFF | (categorical[at + 1] & 0xFFFF) << 16;
          int author;
          boolean present;
          if (token < river) {
            author = token / ClassicSchema.RIVER_TOKENS;
            present = categorical[rivers + token * riverInt.stride() + riverInt.offset(RiverInt.PRESENT)] == 1;
          } else {
            author = (token - river) / ClassicSchema.MELD_TOKENS;
            int meld = melds + (token - river) * meldInt.stride();
            present = categorical[meld + meldInt.offset(MeldInt.PRESENT)] == 1;
          }
          if (!present) {
            assertEquals(packed, 0, "history of padding");
            continue;
          }
          for (PublicHistory.Nibble nibble : PublicHistory.Nibble.values()) {
            int value = packed >>> nibble.shift() & 0xF;
            if (nibble == PublicHistory.Nibble.QUERY_RELATIVE_AUTHOR) {
              assertEquals(value, Math.floorMod(author - query, ClassicSchema.SEATS));
            } else {
              assertTrue(value < PublicHistory.Relation.values().length, nibble + " " + value);
            }
          }
          assertEquals(packed >>> 28, 0);
        }
      }
    }

    private void actions(ClassicInputs in) {
      Bucket bucket = in.bucket();
      short[] categorical = in.categorical();
      float[] numeric = in.numeric();
      FieldLayout<ActionInt> actionInt = schema.layout(ActionInt.class);
      FieldLayout<ActionTransitionInt> transitionInt = schema.layout(ActionTransitionInt.class);
      FieldLayout<ActionTransitionFloat> transitionFloat = schema.layout(ActionTransitionFloat.class);
      for (int action = 0; action < bucket.actions(); action++) {
        boolean real = action < o.legalActions();
        int categories = in.start(Tensor.ACTION_CATEGORIES, 0) + action * actionCategories.length;
        units(categorical, categories, actionCategories, real);
        units(numeric, in.start(Tensor.ACTION_NUMERICS, 0) + action * actionNumerics.length, actionNumerics, real);
        units(categorical, in.start(Tensor.ACTION_ROUTES, 0) + action * actionRoutes.length, actionRoutes, real);
        units(categorical, in.start(Tensor.ACTION_WIN_FACTS, 0) + action * actionFacts.length, actionFacts, real);
        if (real) {
          int id = VocabularyCodec.actionIndex(o.action(action)) + 1;
          assertEquals(categorical[categories + actionInt.offset(ActionInt.ID)], id);
          assertTrue(categorical[categories + actionInt.offset(ActionInt.TYPE)] >= 1);
          assertTrue(categorical[categories + actionInt.offset(ActionInt.WIN_CONTEXT)] >= 1);
          assertTrue(categorical[categories + actionInt.offset(ActionInt.FOLLOW_UP_KIND)] >= 1);
        }
        int transitions = real ? Buckets.transitions(o, action) : 0;
        for (int transition = 0; transition < bucket.transitions(); transition++) {
          boolean present = transition < transitions;
          int index = action * bucket.transitions() + transition;
          int base = in.start(Tensor.TRANSITION_CATEGORIES, 0) + index * transitionCategories.length;
          units(categorical, base, transitionCategories, present);
          int values = in.start(Tensor.TRANSITION_NUMERICS, 0) + index * transitionNumerics.length;
          units(numeric, values, transitionNumerics, present);
          if (present) {
            assertEquals(categorical[base + transitionInt.offset(ActionTransitionInt.PRESENT)], 1);
            assertTrue(categorical[base + transitionInt.offset(ActionTransitionInt.KIND)] >= 1);
            assertTrue(categorical[base + transitionInt.offset(ActionTransitionInt.DISCARD_CONTEXT)] >= 1);
            float frontier =
                numeric[
                    values
                        + transitionFloat.offset(
                            ActionTransitionFloat.NORMALIZED_NEXT_FURITEN_FREE_RON_YAKU_TENPAI_UKEIRE_KINDS)];
            frontierTransitions += frontier > 0 && architecture == Architecture.NANO ? 1 : 0;
            for (ActionTransitionFloat probability :
                List.of(
                    ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_1_SELF_DRAW,
                    ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_2_SELF_DRAWS,
                    ActionTransitionFloat.CURRENT_UKEIRE_HIT_WITHIN_3_SELF_DRAWS)) {
              float value = numeric[values + transitionFloat.offset(probability)];
              assertTrue(value >= 0 && value <= 1, probability + " " + value);
            }
          }
          transitionTiles(in, index, present);
          waits(in, index, present);
        }
      }
    }

    private void transitionTiles(ClassicInputs in, int index, boolean present) {
      int base = in.start(Tensor.TRANSITION_TILES, 0) + index * 34;
      for (int kind = 0; kind < 34; kind++) {
        int code = in.categorical()[base + kind];
        if (present) {
          assertTrue(code >= 1 && code < TileFlag.CODES, "tile code " + code);
        } else {
          assertEquals(code, 0, "tile code of padding");
        }
      }
    }

    /** 待ちの種類が先頭から詰めて置かれ、待ちごとの値は実在する待ちだけが持つことを確かめる。 */
    private void waits(ClassicInputs in, int index, boolean present) {
      short[] categorical = in.categorical();
      int ids = in.start(Tensor.WAIT_TILE_IDS, 0) + index * ClassicSchema.MAX_WAITS;
      int count = 0;
      while (count < ClassicSchema.MAX_WAITS && categorical[ids + count] != 0) {
        assertTrue(categorical[ids + count] < ClassicSchema.WAIT_TILE_VALUES);
        count++;
      }
      assertZero(categorical, ids + count, ClassicSchema.MAX_WAITS - count, "wait ids after the waits");
      if (!present) {
        assertEquals(count, 0, "waits of padding");
      }
      if (count > 0 && architecture == Architecture.NANO) {
        waitingTransitions++;
      }
      for (int wait = 0; wait < ClassicSchema.MAX_WAITS; wait++) {
        boolean real = wait < count;
        int slot = index * ClassicSchema.MAX_WAITS + wait;
        units(in.numeric(), in.start(Tensor.WAIT_SCORES, 0) + slot * waitScores.length, waitScores, real);
        int pairs = in.start(Tensor.WAIT_WIN_FACTS, 0) + slot * 2 * waitFacts.length;
        units(categorical, pairs, waitFacts, real);
        units(categorical, pairs + waitFacts.length, waitFacts, real);
        if (in.rowWidth(Tensor.WAIT_YAKUS) == 0) {
          continue;
        }
        int chunks = in.start(Tensor.WAIT_YAKUS, 0) + slot * 2 * ClassicSchema.YAKU_CHUNKS;
        for (int chunk = 0; chunk < 2 * ClassicSchema.YAKU_CHUNKS; chunk++) {
          int value = categorical[chunks + chunk] & 0xFFFF;
          if (real) {
            assertTrue(value >= 1 && value < ClassicSchema.YAKU_CHUNK_VALUES, "yaku chunk " + value);
          } else {
            assertEquals(value, 0, "yaku chunk of padding");
          }
        }
      }
    }

    private void ledgerAndBoundary(ClassicInputs in, boolean withPrior) {
      if (in.rowWidth(Tensor.POINT_LEDGER_100) > 0) {
        int ledger = in.start(Tensor.POINT_LEDGER_100, 0);
        for (int seat = 0; seat < ClassicSchema.SEATS; seat++) {
          assertEquals(in.categorical()[ledger + o.absoluteSeat(seat)], o.score(seat) / 100);
        }
      }
      int base = in.start(Tensor.BOUNDARY_CONTEXT, 0);
      units(in.numeric(), base, boundary, true);
      FieldLayout<BoundaryFloat> layout = schema.layout(BoundaryFloat.class);
      int grp = base + layout.offset(BoundaryFloat.GRP_RANK);
      for (int index = 0; index < BoundaryFloat.GRP_RANK.count(); index++) {
        assertEquals(in.numeric()[grp + index], withPrior ? prior[index] : 1f / ClassicSchema.SEATS, 0f);
      }
      assertEquals(in.numeric()[base + layout.offset(BoundaryFloat.PRESENT)], withPrior ? 1f : 0f, 0f);
    }

    /** 整数のフィールドが幅の範囲内にあり、パディングなら 0 であることを確かめる。0 でない値を持ったフィールドを覚えておく。 */
    private void units(short[] values, int base, InputField[] layout, boolean real) {
      for (int index = 0; index < layout.length; index++) {
        int value = values[base + index];
        if (!real) {
          if (value != 0) {
            fail(architecture + " padding " + layout[index] + " is " + value);
          }
          continue;
        }
        CategoricalField field = (CategoricalField) layout[index];
        if (value < 0 || value >= field.width()) {
          fail(architecture + " " + field + " is " + value + ", outside [0, " + field.width() + ")");
        }
        if (value != 0) {
          filled.add(field);
        }
      }
    }

    /** 実数のフィールドが有限で、パディングなら 0 であることを確かめる。0 でない値を持ったフィールドを覚えておく。 */
    private void units(float[] values, int base, InputField[] layout, boolean real) {
      for (int index = 0; index < layout.length; index++) {
        float value = values[base + index];
        if (!Float.isFinite(value) || !real && value != 0) {
          fail(architecture + " " + layout[index] + " is " + value + (real ? "" : " in padding"));
        }
        if (value != 0) {
          filled.add(layout[index]);
        }
      }
    }

    /** フィールドの enum を、1 つのトークンの要素ごとに並べる（要素が 2 つ以上のフィールドはその数だけ繰り返す）。 */
    private InputField[] expand(Class<?> type) {
      FieldLayout<?> layout = schema.layoutOf(type);
      InputField[] expanded = new InputField[layout.stride()];
      for (Object each : layout.fields()) {
        InputField field = (InputField) each;
        int offset = layout.offsetOf((Enum<?>) each);
        Arrays.fill(expanded, offset, offset + field.count(), field);
      }
      return expanded;
    }
  }

  /** その席がこの局で立直を宣言したか（宣言牌がまだ通っていなくてもよい）。 */
  private boolean declared(int seat) {
    for (int event = 0; event < o.events(); event++) {
      if (o.eventKind(event) == EventKind.DISCARD && o.eventAuthor(event) == seat && o.discardRiichi(event)) {
        return true;
      }
    }
    return false;
  }

  private static void assertZero(short[] values, int from, int length, String what) {
    for (int index = from; index < from + length; index++) {
      if (values[index] != 0) {
        fail(what + " holds " + values[index] + " at " + (index - from));
      }
    }
  }
}
