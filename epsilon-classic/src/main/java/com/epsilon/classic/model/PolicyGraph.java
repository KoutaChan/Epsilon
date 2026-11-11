package com.epsilon.classic.model;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import com.epsilon.classic.input.ClassicInputs;
import com.epsilon.classic.input.ClassicVocabulary.ActionGroup;
import com.epsilon.classic.input.ClassicVocabulary.ActionType;
import com.epsilon.classic.input.Tensor;

/**
 * 方策のスコアを、麻雀の判断の流れに沿って合法手の確率にする。
 *
 * <p>他家の応答では、ロンか見送るか → 鳴くか見送るか → 鳴きの種類 → 種類の中の候補、の順に分ける。手番では、ツモ和了か続けるか →
 * 九種九牌か続けるか → 槓か打牌か、の順に分ける。槓なら槓の種類 → 候補へ、打牌なら物理の打牌 → ダマか立直か、へ進む。二択では、選ぶ側の
 * ロジット x から、選ぶ側に log σ(x)、反対側に log σ(−x) を割り当てる。片側の手しか無い二択は分けない。種類・候補・物理の打牌は、それぞれの
 * 中で softmax を取る。手の対数確率は、通ったノードでの条件付き対数確率の和になる。そのため、選んだ手の損失は通ったノードにだけ勾配を流す。
 * 鳴き全体と槓全体の確率は二択だけで決まり、候補の数には依らない。
 *
 * <p>貪欲に選んだ手（{@link Greedy}）は、ノードごとに確率の大きい側をたどった手で、確率が最大の手とは限らない。
 */
public final class PolicyGraph {

  /** 合法手の無い位置の対数確率。 */
  private static final float ILLEGAL = -1e9f;

  private static final ActionType[] TYPES = ActionType.values();
  private static final ActionGroup[] GROUPS = ActionGroup.values();
  private static final ActionType[] MELDS = {ActionType.CHI, ActionType.PON, ActionType.DAIMINKAN};
  private static final ActionType[] KANS = {ActionType.ANKAN, ActionType.KAKAN};
  private static final int MELD_BITS = bits(MELDS);
  private static final int KAN_BITS = bits(KANS);
  private static final int DISCARD_BITS = bits(ActionType.DAHAI, ActionType.RIICHI_DAHAI);

  private PolicyGraph() {}

  /**
   * 合法手の対数確率を求める。
   *
   * @param scores 方策のスコア
   * @param batch スコアを作った行のバッチ（合法手の種別と位置の対応を読む）
   * @return [行, 合法手]（FP32）。合法手の無い位置は −10^9
   */
  public static NDArray logProbabilities(PolicyScores scores, ClassicBatch batch) {
    try (ArrayScope scope = new ArrayScope()) {
      ActionColumns columns = ActionColumns.of(batch.schema());
      NDArray categories = batch.tensor(Tensor.ACTION_CATEGORIES);
      NDArray alternatives = toFloat(scores.alternatives());
      NDArray actions = toFloat(scores.actions());
      NDArray riichi = toFloat(scores.riichi());
      long rows = actions.getShape().get(0);
      NDArray types = categories.get("..., {}", columns.type());
      NDArray groups = categories.get("..., {}", columns.group());
      NDArray legal = indicator(categories.get("..., {}", columns.id()).neq(0));
      NDArray response = null;
      for (ActionGroup group : GROUPS) {
        if (group.response()) {
          NDArray member = indicator(groups.eq(group.ordinal() + 1));
          response = response == null ? member : response.add(member);
        }
      }
      NDArray responseRow = present(response.mul(legal), rows);
      NDArray zero = actions.getManager().zeros(new Shape(rows, 1));

      Branch answer = new Branch(actions.zerosLike(), legal.mul(responseRow), zero);
      answer = gate(answer, alternatives, actions, types, ActionType.RON_AGARI, Alternative.RON, rows);
      NDArray assigned = respond(answer, alternatives, actions, types, rows);

      Branch turn = new Branch(assigned, legal.mul(responseRow.neg().add(1f)), zero);
      turn = gate(turn, alternatives, actions, types, ActionType.TSUMO_AGARI, Alternative.TSUMO, rows);
      turn = gate(turn, alternatives, actions, types, ActionType.KYUSHU_KYUHAI, Alternative.KYUSHU, rows);
      turn = kan(turn, alternatives, actions, types, rows);
      assigned = discard(turn, actions, riichi, batch.tensor(Tensor.ACTION_ROUTES), types, columns);
      return scope.keep(assigned.add(legal.neg().add(1f).mul(ILLEGAL)));
    }
  }

  /**
   * 合法手の確率を求める。
   *
   * @param logProbabilities {@link #logProbabilities} の値
   * @param batch 同じ行のバッチ
   * @return [行, 合法手]（FP32）。行ごとに和が 1 で、合法手の無い位置は 0
   */
  public static NDArray probabilities(NDArray logProbabilities, ClassicBatch batch) {
    ActionColumns columns = ActionColumns.of(batch.schema());
    NDArray legal = indicator(batch.tensor(Tensor.ACTION_CATEGORIES).get("..., {}", columns.id()).neq(0));
    return logProbabilities.exp().mul(legal);
  }

  /**
   * 手を貪欲に選ぶ。グラフのノードごとに、確率の和が大きい側をたどる。和が同じなら、二択では選ぶ側を、種類・候補・物理の打牌では前の方を
   * 選ぶ。手の種別は合法手の整数の列から読む。選ぶときにメモリを確保しない。作業用の状態を持つのでスレッドセーフではなく、スレッドごとに
   * 1 つずつ使う。
   */
  public static final class Greedy {

    private short[] categorical;
    private ActionColumns columns;
    private int actions;
    private int routes;
    private float[] probabilities;
    private int offset;
    private int legal;

    /**
     * 1 行の中で、手を貪欲に選ぶ。
     *
     * @param inputs 判断を書いた入力
     * @param row 行の番号
     * @param probabilities 合法手の確率（{@link PolicyGraph#probabilities} の行）
     * @param offset 行の先頭の位置
     * @return 選んだ手が合法手の何番目か
     * @throws IllegalArgumentException 行に合法手が無いとき
     */
    public int select(ClassicInputs inputs, int row, float[] probabilities, int offset) {
      categorical = inputs.categorical();
      columns = ActionColumns.of(inputs.schema());
      actions = inputs.start(Tensor.ACTION_CATEGORIES, row);
      routes = inputs.start(Tensor.ACTION_ROUTES, row);
      this.probabilities = probabilities;
      this.offset = offset;
      legal = 0;
      while (legal < inputs.bucket().actions() && field(legal, columns.id()) != 0) {
        legal++;
      }
      if (legal == 0) {
        throw new IllegalArgumentException("The row has no legal action");
      }
      boolean response = false;
      for (int slot = 0; slot < legal; slot++) {
        response |= GROUPS[field(slot, columns.group()) - 1].response();
      }
      int all = (1 << TYPES.length) - 1;
      if (response) {
        int ron = bit(ActionType.RON_AGARI);
        if (has(ron) && mass(ron) >= mass(all & ~ron)) {
          return best(ron);
        }
        int pass = bit(ActionType.PASS);
        if (has(MELD_BITS) && (!has(pass) || mass(MELD_BITS) >= mass(pass))) {
          return best(heaviest(MELDS));
        }
        return best(pass);
      }
      int tsumo = bit(ActionType.TSUMO_AGARI);
      if (has(tsumo) && mass(tsumo) >= mass(all & ~tsumo)) {
        return best(tsumo);
      }
      int kyushu = bit(ActionType.KYUSHU_KYUHAI);
      if (has(kyushu) && mass(kyushu) >= mass(all & ~tsumo & ~kyushu)) {
        return best(kyushu);
      }
      if (has(KAN_BITS) && (!has(DISCARD_BITS) || mass(KAN_BITS) >= mass(DISCARD_BITS))) {
        return best(heaviest(KANS));
      }
      return discard();
    }

    /** 確率の和が最大の物理の打牌を選び、その中でダマと立直のうち確率の大きい側を選ぶ。 */
    private int discard() {
      int chosen = -1;
      float chosenMass = -1f;
      for (int slot = 0; slot < legal; slot++) {
        if (route(slot, columns.representative()) == 0) {
          continue;
        }
        float identity = slotMass(route(slot, columns.damaSlot())) + slotMass(route(slot, columns.riichiSlot()));
        if (identity > chosenMass) {
          chosenMass = identity;
          chosen = slot;
        }
      }
      if (chosen < 0) {
        throw new IllegalStateException("A turn without a discard, kan, tsumo or kyushu");
      }
      int dama = route(chosen, columns.damaSlot()) - 1;
      int riichi = route(chosen, columns.riichiSlot()) - 1;
      if (riichi >= 0 && (dama < 0 || probability(riichi) >= probability(dama))) {
        return riichi;
      }
      return dama;
    }

    /** 渡した種別のうち、確率の和が最大の種別のビットを返す（同じなら前の方）。 */
    private int heaviest(ActionType[] types) {
      int heaviest = 0;
      float heaviestMass = -1f;
      for (ActionType type : types) {
        int bit = bit(type);
        if (has(bit) && mass(bit) > heaviestMass) {
          heaviestMass = mass(bit);
          heaviest = bit;
        }
      }
      return heaviest;
    }

    private boolean has(int types) {
      for (int slot = 0; slot < legal; slot++) {
        if ((types & typeBit(slot)) != 0) {
          return true;
        }
      }
      return false;
    }

    private float mass(int types) {
      float sum = 0f;
      for (int slot = 0; slot < legal; slot++) {
        if ((types & typeBit(slot)) != 0) {
          sum += probability(slot);
        }
      }
      return sum;
    }

    /** ビットで指定した種別の手のうち、確率が最大の手を返す（同じなら前の方）。 */
    private int best(int types) {
      int best = -1;
      for (int slot = 0; slot < legal; slot++) {
        if ((types & typeBit(slot)) != 0 && (best < 0 || probability(slot) > probability(best))) {
          best = slot;
        }
      }
      if (best < 0) {
        throw new IllegalStateException("No legal action of the chosen kind");
      }
      return best;
    }

    private float slotMass(int storedSlot) {
      return storedSlot == 0 ? 0f : probability(storedSlot - 1);
    }

    private float probability(int slot) {
      return probabilities[offset + slot];
    }

    private int typeBit(int slot) {
      return 1 << (field(slot, columns.type()) - 1);
    }

    private int field(int slot, int column) {
      return categorical[actions + slot * columns.stride() + column];
    }

    private int route(int slot, int column) {
      return categorical[routes + slot * columns.routeStride() + column];
    }
  }

  /** 途中まで分けた枝。割り当て済みの対数確率、まだ分けていない手のマスク、この枝に至るまでの対数確率を持つ。 */
  private record Branch(NDArray assigned, NDArray unresolved, NDArray mass) {}

  /** 二択で、選ぶ側の種別の手とそれ以外の手に分ける。 */
  private static Branch gate(
      Branch branch,
      NDArray alternatives,
      NDArray actions,
      NDArray types,
      ActionType accepted,
      Alternative alternative,
      long rows) {
    NDArray acceptedMask = typeMask(types, branch.unresolved(), accepted);
    NDArray declinedMask = branch.unresolved().sub(acceptedMask);
    NDArray active = present(acceptedMask, rows).mul(present(declinedMask, rows));
    NDArray logit = alternatives.get(":, {}", alternative.ordinal()).reshape(rows, 1);
    NDArray assigned =
        assign(branch.assigned(), actions, acceptedMask, branch.mass().add(logSigmoid(logit).mul(active)));
    return new Branch(assigned, declinedMask, branch.mass().add(logSigmoid(logit.neg()).mul(active)));
  }

  /** 他家の応答で、ロンを分けた後に残った手を、見送るか鳴くかに分ける。鳴くなら種類と候補を割り当てる。 */
  private static NDArray respond(Branch branch, NDArray alternatives, NDArray actions, NDArray types, long rows) {
    NDArray passMask = typeMask(types, branch.unresolved(), ActionType.PASS);
    NDArray[] meldMasks = typeMasks(types, branch.unresolved(), MELDS);
    NDArray meldMask = union(meldMasks);
    NDArray active = present(passMask, rows).mul(present(meldMask, rows));
    NDArray logit = alternatives.get(":, {}", Alternative.CALL.ordinal()).reshape(rows, 1);
    NDArray assigned =
        assign(branch.assigned(), actions, passMask, branch.mass().add(logSigmoid(logit.neg()).mul(active)));
    NDArray meldMass = branch.mass().add(logSigmoid(logit).mul(active));
    return assignTypes(
        assigned,
        alternatives,
        actions,
        meldMasks,
        new Alternative[] {Alternative.CHI, Alternative.PON, Alternative.DAIMINKAN},
        meldMass,
        rows);
  }

  /** 手番で、ツモ和了と九種九牌を分けた後に残った手を、槓か打牌かに分ける。槓なら種類と候補を割り当て、打牌の枝を返す。 */
  private static Branch kan(Branch branch, NDArray alternatives, NDArray actions, NDArray types, long rows) {
    NDArray discardMask =
        typeMask(types, branch.unresolved(), ActionType.DAHAI)
            .add(typeMask(types, branch.unresolved(), ActionType.RIICHI_DAHAI));
    NDArray[] kanMasks = typeMasks(types, branch.unresolved(), KANS);
    NDArray kanMask = union(kanMasks);
    NDArray active = present(discardMask, rows).mul(present(kanMask, rows));
    NDArray logit = alternatives.get(":, {}", Alternative.KAN.ordinal()).reshape(rows, 1);
    NDArray assigned =
        assignTypes(
            branch.assigned(),
            alternatives,
            actions,
            kanMasks,
            new Alternative[] {Alternative.ANKAN, Alternative.KAKAN},
            branch.mass().add(logSigmoid(logit).mul(active)),
            rows);
    return new Branch(assigned, discardMask, branch.mass().add(logSigmoid(logit.neg()).mul(active)));
  }

  /** 種類の softmax を取り、続けて種類の中の候補の softmax を割り当てる。 */
  private static NDArray assignTypes(
      NDArray assigned,
      NDArray alternatives,
      NDArray actions,
      NDArray[] masks,
      Alternative[] kinds,
      NDArray mass,
      long rows) {
    NDList scores = new NDList(kinds.length);
    NDList presence = new NDList(kinds.length);
    for (int index = 0; index < kinds.length; index++) {
      scores.add(alternatives.get(":, {}", kinds[index].ordinal()).reshape(rows, 1));
      presence.add(present(masks[index], rows));
    }
    NDArray kindScores = NDArrays.concat(scores, 1);
    NDArray normalizer = NDArrays.maskedLogSumExp(kindScores, NDArrays.concat(presence, 1), 1);
    NDArray result = assigned;
    for (int index = 0; index < kinds.length; index++) {
      NDArray kind = kindScores.get(":, {}", index).reshape(rows, 1).sub(normalizer);
      result = assign(result, actions, masks[index], mass.add(kind));
    }
    return result;
  }

  /** 打牌の枝を割り当てる。物理の打牌で softmax を取り、ダマか立直かの二択で分ける。 */
  private static NDArray discard(
      Branch branch, NDArray actions, NDArray riichi, NDArray routes, NDArray types, ActionColumns columns) {
    DiscardChoices choices =
        DiscardChoices.of(
            routes,
            typeMask(types, branch.unresolved(), ActionType.DAHAI),
            typeMask(types, branch.unresolved(), ActionType.RIICHI_DAHAI),
            columns);
    NDArray normalizer = NDArrays.maskedLogSumExp(actions, choices.representativeMask(), 1);
    NDArray representative =
        actions.sub(normalizer).add(branch.mass()).mul(choices.representativeMask());
    NDArray identity = NDArrays.paddedBatchGather(representative, choices.representativeSlots());
    NDArray logit = NDArrays.paddedBatchGather(riichi, choices.riichiSlots());
    NDArray active = choices.damaPresent().mul(choices.riichiPresent());
    NDArray variant =
        logSigmoid(logit)
            .mul(choices.riichiMask())
            .add(logSigmoid(logit.neg()).mul(choices.damaMask()))
            .mul(active);
    return branch.assigned().add(identity.add(variant).mul(choices.discardMask()));
  }

  /** マスクが 1 の手に、それらの手の中で取ったスコアの log softmax と、枝の対数確率の和を割り当てる。 */
  private static NDArray assign(NDArray assigned, NDArray scores, NDArray mask, NDArray mass) {
    NDArray normalizer = NDArrays.maskedLogSumExp(scores, mask, 1);
    return assigned.add(scores.sub(normalizer).add(mass).mul(mask));
  }

  private static NDArray[] typeMasks(NDArray types, NDArray mask, ActionType[] kinds) {
    NDArray[] masks = new NDArray[kinds.length];
    for (int index = 0; index < kinds.length; index++) {
      masks[index] = typeMask(types, mask, kinds[index]);
    }
    return masks;
  }

  private static NDArray typeMask(NDArray types, NDArray mask, ActionType type) {
    return indicator(types.eq(ActionColumns.stored(type))).mul(mask);
  }

  private static NDArray union(NDArray[] masks) {
    NDArray union = masks[0];
    for (int index = 1; index < masks.length; index++) {
      union = union.add(masks[index]);
    }
    return union;
  }

  /** 行にマスクが 1 の手があれば 1 の配列 [行, 1] を返す。 */
  private static NDArray present(NDArray mask, long rows) {
    return indicator(mask.sum(new int[] {1}).gt(0.5f)).reshape(rows, 1);
  }

  /** log σ(x) = −softplus(−x)。 */
  private static NDArray logSigmoid(NDArray value) {
    return Activation.softPlus(value.neg()).neg();
  }

  private static NDArray indicator(NDArray condition) {
    return condition.toType(DataType.FLOAT32, false).stopGradient();
  }

  private static NDArray toFloat(NDArray array) {
    return array.getNDArrayInternal().differentiableCast(DataType.FLOAT32);
  }

  private static int bit(ActionType type) {
    return 1 << type.ordinal();
  }

  private static int bits(ActionType... types) {
    int bits = 0;
    for (ActionType type : types) {
      bits |= bit(type);
    }
    return bits;
  }
}
