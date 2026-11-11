package com.epsilon.classic.campaign;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import java.io.IOException;
import java.util.List;
import java.util.Objects;
import java.util.SplittableRandom;

/**
 * 学習器の相手の候補。macro ごとに相手を 1 人選び、その macro の全ての相手の席に座らせる。
 *
 * @param champion 今の王者
 * @param past 過去の王者（今の王者を除き、王者になった順）
 */
public record OpponentPool(Champion champion, List<Champion> past) {

  public OpponentPool {
    Objects.requireNonNull(champion, "champion");
    past = List.copyOf(past);
  }

  /**
   * 王者の保存先の今の王者と過去の王者。
   *
   * @throws IllegalStateException 保存先に王者がいない
   * @throws IOException 保存先が読めない
   */
  public static OpponentPool of(ChampionStore store) throws IOException {
    Champion champion =
        store.current().orElseThrow(() -> new IllegalStateException("The champion store holds no champion"));
    return new OpponentPool(champion, store.past());
  }

  /**
   * macro の相手を選ぶ。確率 p で今の王者を、残りは過去の王者から一様に 1 人を選ぶ。過去の王者がいなければ今の王者。種だけで決まる。
   *
   * @param championProbability 今の王者を選ぶ確率 p（0 以上 1 以下）
   * @param seed 選びの種
   */
  public Champion draw(double championProbability, long seed) {
    if (!(championProbability >= 0 && championProbability <= 1)) {
      throw new IllegalArgumentException("The champion probability must be in [0, 1]: " + championProbability);
    }
    if (past.isEmpty()) {
      return champion;
    }
    SplittableRandom random = new SplittableRandom(seed);
    return random.nextDouble() < championProbability ? champion : past.get(random.nextInt(past.size()));
  }
}
