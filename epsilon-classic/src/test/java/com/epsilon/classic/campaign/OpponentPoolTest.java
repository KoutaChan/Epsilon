package com.epsilon.classic.campaign;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertSame;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.eval.ChampionStore;
import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.StoreEvent.Initial;
import com.epsilon.agent.eval.StoreEvent.Promoted;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Comparator;
import java.util.List;
import java.util.stream.Stream;
import org.testng.annotations.Test;

/**
 * 学習器の相手の選び: 確率 p で今の王者、残りは過去の王者から一様に 1 人。過去の王者がいなければいつも今の王者。選びは種だけで決まり、王者の
 * 保存先の今の王者と過去の王者から候補を作る。
 */
public class OpponentPoolTest {

  private static final int DRAWS = 60_000;

  /** 二項分布の標準誤差のこの倍まで、頻度が確率からずれてよい。 */
  private static final double SIGMAS = 4.5;

  private static Champion champion(String label, int version) {
    return new Champion(label, Path.of("/checkpoints", label).toAbsolutePath(), version, null);
  }

  @Test
  public void theChampionComesWithItsProbabilityAndThePastShareTheRest() {
    Champion now = champion("v00030", 30);
    List<Champion> past = List.of(champion("v00010", 10), champion("v00020", 20), champion("v00025", 25));
    OpponentPool pool = new OpponentPool(now, past);
    double p = 0.7;
    int[] counts = new int[past.size()];
    int champions = 0;
    for (long seed = 0; seed < DRAWS; seed++) {
      Champion drawn = pool.draw(p, seed);
      if (drawn == now) {
        champions++;
      } else {
        counts[past.indexOf(drawn)]++;
      }
    }
    assertWithin(champions, p);
    for (int count : counts) {
      assertWithin(count, (1 - p) / past.size());
    }
  }

  @Test
  public void theEdgesOfTheProbabilityAndAnEmptyPast() {
    Champion now = champion("v00002", 2);
    Champion old = champion("v00001", 1);
    OpponentPool alone = new OpponentPool(now, List.of());
    OpponentPool pool = new OpponentPool(now, List.of(old));
    for (long seed = 0; seed < 200; seed++) {
      assertSame(alone.draw(0, seed), now);
      assertSame(pool.draw(1, seed), now);
      assertSame(pool.draw(0, seed), old);
      assertSame(pool.draw(0.7, seed), pool.draw(0.7, seed));
    }
    expectThrows(IllegalArgumentException.class, () -> pool.draw(-0.1, 1));
    expectThrows(IllegalArgumentException.class, () -> pool.draw(1.1, 1));
    expectThrows(IllegalArgumentException.class, () -> pool.draw(Double.NaN, 1));
  }

  @Test
  public void thePoolHoldsTheStoresChampionAndItsPastChampions() throws IOException {
    Path root = Files.createTempDirectory("opponent-pool");
    try {
      Champion first = Champion.of("v00001", checkpoint(root, "v1", "a"), 1);
      Champion second = Champion.of("v00011", checkpoint(root, "v11", "b"), 11);
      Champion third = Champion.of("v00021", checkpoint(root, "v21", "c"), 21);
      ChampionStore store = ChampionStore.open(root.resolve("champion"));
      expectThrows(IllegalStateException.class, () -> OpponentPool.of(store));
      store.append(new Initial(first));
      assertEquals(OpponentPool.of(store), new OpponentPool(first, List.of()));
      store.append(new Promoted(second, "v00011-champion"));
      store.append(new Promoted(third, "v00021-champion"));
      assertEquals(OpponentPool.of(store), new OpponentPool(third, List.of(first, second)));
    } finally {
      try (Stream<Path> files = Files.walk(root)) {
        for (Path file : files.sorted(Comparator.reverseOrder()).toList()) {
          Files.delete(file);
        }
      }
    }
  }

  private static void assertWithin(int count, double probability) {
    double error = Math.sqrt(probability * (1 - probability) / DRAWS);
    double frequency = (double) count / DRAWS;
    assertTrue(Math.abs(frequency - probability) <= SIGMAS * error, frequency + " vs " + probability);
  }

  private static Path checkpoint(Path root, String name, String weights) throws IOException {
    Path directory = root.resolve(name);
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("weights.bin"), weights, StandardCharsets.UTF_8);
    Files.writeString(directory.resolve("model.json"), "{}", StandardCharsets.UTF_8);
    return directory;
  }
}
