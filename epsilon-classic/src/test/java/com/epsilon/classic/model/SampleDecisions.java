package com.epsilon.classic.model;

import com.epsilon.agent.observe.Observation;
import com.epsilon.agent.observe.ObservationBatch;
import com.epsilon.agent.observe.ObservationWriter;
import com.epsilon.classic.RandomPlay;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.VocabularyCodec;
import com.epsilon.mahjong.engine.Boundary;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import java.util.EnumMap;
import java.util.Map;

/**
 * テスト用の判断のサンプル。ランダムな自己対局から判断を集め、手の種類（ロン・ツモ・九種九牌・鳴き・槓・立直・打牌だけ）ごとに
 * 決めた数まで残す。集まる判断は、シードとステップ数だけで決まる。
 */
public final class SampleDecisions {

  /** 判断を分類するときの手の種類。 */
  public enum Kind {
    RON,
    TSUMO,
    KYUSHU,
    CALL,
    KAN,
    RIICHI,
    DISCARD_ONLY
  }

  public static final Variant VARIANT = Variant.TENHOU_4P;

  private final ObservationBatch store = new ObservationBatch(VARIANT, 1 << 16);
  private final Map<Kind, Integer> counts = new EnumMap<>(Kind.class);

  /**
   * @param seed 自己対局のシード
   * @param perKind 種類ごとに集める判断の最大数
   * @param steps 進めるステップ数の上限
   */
  public SampleDecisions(long seed, int perKind, int steps) {
    RandomPlay play = new RandomPlay(VARIANT, seed);
    ObservationWriter writer = new ObservationWriter(VARIANT);
    ObservationBatch scratch = new ObservationBatch(VARIANT, 1 << 12);
    Observation o = new Observation(VARIANT);
    for (Kind kind : Kind.values()) {
      counts.put(kind, 0);
    }
    for (int step = 0; step < steps && !full(perKind); step++) {
      if (play.boundary() == Boundary.DECISIONS) {
        Decisions decisions = play.engine.decisions();
        for (int slot = 0; slot < decisions.size(); slot++) {
          scratch.clear();
          o.bind(scratch, writer.capture(play.engine.view(decisions.seat(slot)), decisions, slot, scratch));
          Kind kind = kind(o);
          if (counts.get(kind) < perKind) {
            store.add(o);
            counts.merge(kind, 1, Integer::sum);
          }
        }
      }
      play.step();
    }
  }

  public int size() {
    return store.size();
  }

  /** 集めた判断のうち、指定した番号のものを渡した {@code Observation} で読めるようにして返す。 */
  public Observation bind(Observation o, int index) {
    return o.bind(store, index);
  }

  public Map<Kind, Integer> counts() {
    return counts;
  }

  private boolean full(int perKind) {
    for (int count : counts.values()) {
      if (count < perKind) {
        return false;
      }
    }
    return true;
  }

  /** 判断の種類を返す。合法手が複数の種類にまたがるときは、列挙で前にある種類にする（ロンが最優先）。 */
  private static Kind kind(Observation o) {
    Kind kind = Kind.DISCARD_ONLY;
    for (int action = 0; action < o.legalActions(); action++) {
      Kind found =
          switch (VocabularyCodec.actionType(o.action(action).type())) {
            case RON_AGARI -> Kind.RON;
            case TSUMO_AGARI -> Kind.TSUMO;
            case KYUSHU_KYUHAI -> Kind.KYUSHU;
            case CHI, PON, DAIMINKAN -> Kind.CALL;
            case ANKAN, KAKAN -> Kind.KAN;
            case RIICHI_DAHAI -> Kind.RIICHI;
            default -> Kind.DISCARD_ONLY;
          };
      if (found.ordinal() < kind.ordinal()) {
        kind = found;
      }
    }
    return kind;
  }

  /** 判断の合法手の 1 つについて、その手の種類を返す。 */
  public static ActionType type(Observation o, int action) {
    return VocabularyCodec.actionType(o.action(action).type());
  }
}
