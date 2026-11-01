package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import com.epsilon.prism.core.CrnSeeds;
import com.epsilon.util.SeedMixer;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.List;

/**
 * 当てはめに使う運の網 Ω の標本の表（メモリの上）。封印したコホートの標本（{@link LuckSampleStore}）を読んだ順に並べ、O° は 1 本の
 * {@code long[]} に続けて置く。作った後は変えないので、複数のスレッドから読んでよい。
 *
 * <p>保留は半荘ごとに決める（{@link #heldOut}）: 半荘 ID のハッシュの一様値が割合より小さい半荘の標本を全て保留にする。同じ半荘の標本（同じ
 * 牌山と兄弟の分岐）が当てはめと保留に分かれないので、保留の R² が当てはめの標本に寄らない。鍵は半荘 ID だけなので、同じ標本ならどの網・
 * どの設定で当てても同じ保留になる。
 */
public final class LuckSamples {

  private static final int SEATS = GameState.NUM_PLAYERS;

  /** 保留の硬貨の塩（"HOLD"。標本の硬貨 {@link LuckSampling} とは別）。 */
  private static final long HOLDOUT_SALT = 0x484f4c44L;

  private long[] words = new long[1 << 16];
  private int[] start = new int[1024 + 1];
  private float[] y = new float[1024 * SEATS];
  private float[] weight = new float[1024];
  private long[] game = new long[1024];
  private int size;

  private LuckSamples() {}

  /**
   * 封印したコホートの標本を読む。
   *
   * @param cohorts コホートのディレクトリ（標本は luck/）
   * @return 表
   * @throws IOException 封印されていないか、壊れているとき
   */
  public static LuckSamples load(List<Path> cohorts) throws IOException {
    LuckSamples samples = new LuckSamples();
    for (Path cohort : cohorts) {
      LuckSampleStore store = LuckSampleStore.open(cohort);
      samples.reserve(Math.toIntExact(samples.size + store.samples()));
      try (LuckSampleStore.Cursor cursor = store.cursor()) {
        while (cursor.next()) {
          samples.add(cursor);
        }
      }
    }
    return samples;
  }

  /**
   * 半荘の標本を保留にするかを返す。
   *
   * @param game 半荘 ID
   * @param fraction 保留の割合
   * @return 保留なら true
   */
  public static boolean heldOut(long game, double fraction) {
    return CrnSeeds.unit(SeedMixer.mix64(game ^ HOLDOUT_SALT)) < fraction;
  }

  public int size() {
    return size;
  }

  /** 標本 {@code sample} の半荘 ID。 */
  public long game(int sample) {
    return game[sample];
  }

  /**
   * 標本を塊の 1 行に書く（入力・プール・結果・Y・重み）。
   *
   * @param sample 標本
   * @param rows 書く先
   * @param row 行
   * @param tokenizer 符号化（このスレッドのもの）
   * @param observation 読み手（このスレッドのもの）
   */
  void put(
      int sample, LuckRows rows, int row, LuckTokenizer tokenizer, LuckObservation observation) {
    observation.bind(words, start[sample]);
    rows.put(row, tokenizer, observation, y, sample * SEATS, weight[sample], game[sample]);
  }

  private void add(LuckSampleStore.Cursor cursor) {
    long[] observation = cursor.words();
    int longs = cursor.observation().longs();
    int at = start[size];
    if (at + longs > words.length) {
      words = Arrays.copyOf(words, Math.max(words.length * 2, at + longs));
    }
    System.arraycopy(observation, 0, words, at, longs);
    start[size + 1] = at + longs;
    for (int seat = 0; seat < SEATS; seat++) {
      y[size * SEATS + seat] = cursor.y(seat);
    }
    weight[size] = cursor.weight();
    game[size] = cursor.game();
    size++;
  }

  private void reserve(int capacity) {
    if (capacity <= weight.length) {
      return;
    }
    start = Arrays.copyOf(start, capacity + 1);
    y = Arrays.copyOf(y, capacity * SEATS);
    weight = Arrays.copyOf(weight, capacity);
    game = Arrays.copyOf(game, capacity);
  }
}
