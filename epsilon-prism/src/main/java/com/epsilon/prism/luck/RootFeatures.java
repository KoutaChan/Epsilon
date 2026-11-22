package com.epsilon.prism.luck;

import com.epsilon.agent.observe.Observation;
import com.epsilon.mahjong.action.Action;
import com.epsilon.mahjong.engine.Decisions;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.rules.Vocabulary;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.prism.auxiliary.OracleCodec;
import com.epsilon.prism.noise.RootCoefficients;
import java.util.Arrays;

/** 判断の時点で分かる卓と手牌の値。配列の手は受け取り側が指定した候補順に並ぶ。 */
record RootFeatures(
    double[] common,
    int[] actions,
    boolean hands,
    LedgerCell.Standing standing,
    Vocabulary<Action.Type> types) {

  static final int ACTION_FIELDS = 3;

  int actionCount() {
    return actions.length / ACTION_FIELDS;
  }

  int points(int action) {
    return actions[action * ACTION_FIELDS + 2];
  }

  enum Field {
    CONSTANT,
    ROUND,
    ROUND_SQUARED,
    WALL,
    WALL_SQUARED,
    GAP,
    ABS_GAP,
    RANK,
    DEALER,
    CAN_WIN,
    LEGAL,
    HONBA,
    DEPOSITS,
    ACTION,
    TENPAI,
    WAITS,
    SHANTEN,
    WAITING_OPPONENTS,
    DEAL_IN
  }

  /** ルールと情報集合から定まる、係数と特徴の共通の列順。 */
  static final class Layout {
    private final int[] offsets = new int[Field.values().length + 1];
    final int common, types, hand, sharedHands, width;

    Layout(Variant v, boolean candidates, boolean hands) {
      for (Field f : Field.values()) {
        int count =
            switch (f) {
              case GAP, ABS_GAP, RANK -> v.seats() - 1;
              case ACTION -> candidates ? Vocabulary.of(Action.Type.class, v).size() : 0;
              case TENPAI, WAITS, SHANTEN -> hands ? 1 : 0;
              case WAITING_OPPONENTS, DEAL_IN -> hands && candidates ? 1 : 0;
              default -> 1;
            };
        offsets[f.ordinal() + 1] = at(f) + count;
      }
      common = at(Field.ACTION);
      types = at(Field.TENPAI) - common;
      hand = hands ? at(Field.TENPAI) : -1;
      sharedHands = at(Field.WAITING_OPPONENTS) - at(Field.TENPAI);
      width = offsets[Field.values().length];
    }

    static Layout of(Variant v, RootCoefficients cfg) {
      return new Layout(
          v,
          cfg.family() == RootCoefficients.Family.PER_CANDIDATE,
          cfg.information() == RootCoefficients.Information.ROOT_HANDS);
    }

    int at(Field field) {
      return offsets[field.ordinal()];
    }
  }

  Action.Type type(int action) {
    return types.get(actions[action * ACTION_FIELDS]);
  }

  int actionType(int action) {
    return actions[action * ACTION_FIELDS];
  }

  int opponentsWaiting(int action) {
    return actions[action * ACTION_FIELDS + 1];
  }

  double dealInPoints(int action) {
    return (double) points(action) / Variant.DEPOSIT;
  }

  /** 候補に対応する特徴を、学習した係数と同じ順で書く。 */
  void fill(int action, Layout layout, double[] out) {
    System.arraycopy(common, 0, out, 0, layout.common);
    if (layout.types > 0) {
      Arrays.fill(out, layout.common, layout.common + layout.types, 0);
      out[layout.common + actionType(action)] = 1;
    }
    if (layout.hand >= 0) {
      System.arraycopy(common, layout.common, out, layout.hand, layout.sharedHands);
      if (layout.types > 0) {
        out[layout.at(Field.WAITING_OPPONENTS)] = opponentsWaiting(action);
        out[layout.at(Field.DEAL_IN)] = dealInPoints(action);
      }
    }
  }

  /** 観測の復号と、相手ごとの放銃牌の集計に使う作業領域。 */
  static final class Reader {
    final Variant variant;
    final Layout commonLayout;
    final Observation observation;
    final OracleCodec oracle;
    final Vocabulary<Action.Type> types;
    final int[] danger = new int[Tiles.KINDS], points = new int[Tiles.KINDS];

    Reader(Variant variant) {
      this.variant = variant;
      commonLayout = new Layout(variant, false, true);
      observation = new Observation(variant);
      oracle = OracleCodec.of(variant);
      types = Vocabulary.of(Action.Type.class, variant);
    }

    RootFeatures read(
        long[] words, int base, long[] hidden, int[] measured, int count, boolean useHands) {
      var o = observation.bind(words, base);
      var standing = LedgerCell.of(variant).standing(o);
      int others = variant.seats() - 1;
      Layout layout = commonLayout;
      double[] x = new double[layout.width];
      double k = (double) o.kyoku() / (variant.lastRound() + 1),
          w = (double) o.remainingWall() / variant.liveWall();
      x[layout.at(Field.CONSTANT)] = 1;
      x[layout.at(Field.ROUND)] = k;
      x[layout.at(Field.ROUND_SQUARED)] = k * k;
      x[layout.at(Field.WALL)] = w;
      x[layout.at(Field.WALL_SQUARED)] = w * w;
      int gaps = layout.at(Field.GAP);
      for (int j = 0; j < others; j++)
        x[gaps + j] = (double) (o.score(0) - o.score(j + 1)) / variant.startPoints();
      Arrays.sort(x, gaps, gaps + others);
      for (int j = 0; j < others; j++) {
        x[layout.at(Field.ABS_GAP) + j] = Math.abs(x[gaps + j]);
        x[layout.at(Field.RANK) + j] = standing.rank() == j + 1 ? 1 : 0;
      }
      x[layout.at(Field.DEALER)] = o.dealer() == 0 ? 1 : 0;
      x[layout.at(Field.CAN_WIN)] = o.wins() > 0 ? 1 : 0;
      x[layout.at(Field.LEGAL)] = (double) o.legalActions() / Decisions.MAX_LEGAL;
      x[layout.at(Field.HONBA)] = o.honba();
      x[layout.at(Field.DEPOSITS)] = o.kyotaku();
      boolean hands = useHands && hidden.length > 0;
      if (hands) {
        Arrays.fill(danger, 0);
        Arrays.fill(points, 0);
        int pos = 0;
        for (int j = 1; j <= others; j++) {
          int sh = oracle.shanten(hidden, 0, j);
          if (sh == 0) x[layout.at(Field.TENPAI)]++;
          x[layout.at(Field.WAITS)] += Long.bitCount(oracle.waits(hidden, 0, j));
          x[layout.at(Field.SHANTEN)] += sh;
          for (long bits = oracle.danger(hidden, 0, j); bits != 0; bits &= bits - 1) {
            int tile = Long.numberOfTrailingZeros(bits);
            danger[tile]++;
            points[tile] += oracle.points(hidden, 0, pos++);
          }
        }
      }
      int[] features = new int[count * ACTION_FIELDS];
      for (int j = 0; j < count; j++) {
        Action a = o.action(measured == null ? j : measured[j]);
        int off = j * ACTION_FIELDS;
        features[off] = types.index(a.type());
        if (hands && (a.type() == Action.Type.DAHAI || a.type() == Action.Type.RIICHI_DAHAI)) {
          features[off + 1] = danger[a.tile()];
          features[off + 2] = points[a.tile()];
        }
      }
      return new RootFeatures(x, features, hands, standing, types);
    }
  }
}
