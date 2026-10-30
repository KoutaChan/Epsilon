package com.epsilon.prism.luck;

import com.epsilon.core.GameState;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.file.Path;

/**
 * 運の網の試験の道具: {@link LuckGames} の打ち方で半荘を打ち、主経路の出来事を捕捉して、封印した標本のコホートを書く（{@link #write}。Y は局末の
 * 点（精算の後）から 25000 点を引いて 1 万点で割ったもの）。結果の軸の並びを試すために、Y を結果の関数に替えたコホートも作れる（{@link
 * #outcomeTarget}）。
 */
final class LuckCohorts {

  private static final double START = 25_000;
  private static final double UNIT = 10_000;

  private LuckCohorts() {}

  /**
   * 標本のコホートを書く。
   *
   * @param cohort コホートのディレクトリ
   * @param seed 最初の半荘の種（硬貨の種も同じ）
   * @param games 半荘の数（半荘 ID は 0 から）
   * @param rate 標本の率
   * @return コホートのディレクトリ
   * @throws IOException 書けないとき
   */
  static Path write(Path cohort, long seed, int games, double rate) throws IOException {
    LuckCapture capture = new LuckCapture(new LuckSampling(seed, rate));
    try (LuckSampleWriter writer = new LuckSampleWriter(cohort, 1)) {
      LuckGames.play(
          seed,
          games,
          null,
          new LuckGames.Visitor() {
            @Override
            public LuckSink sink() {
              return capture;
            }

            @Override
            public void roundStarted(int game, int round) {
              capture.root(game, LuckSampling.MAIN_PATH, LuckSampling.handRoot(round));
            }

            @Override
            public void roundEnded(int game, int round, GameState state) {
              LuckBatch batch = capture.detach();
              if (batch == null) {
                return;
              }
              double[] phi = new double[GameState.NUM_PLAYERS];
              for (int seat = 0; seat < phi.length; seat++) {
                phi[seat] = (state.score(seat) - START) / UNIT;
              }
              try {
                batch.settle(phi, writer);
              } catch (IOException failure) {
                throw new UncheckedIOException(failure);
              }
            }
          });
      writer.seal();
    }
    return cohort;
  }

  /**
   * 同じ標本の Y を出来事の前の状態と結果だけの関数に替えたコホートを書く: 行為者は結果の牌の手の枚数（赤なら + 2）、下家は結果の牌の手の
   * 枚数、ほかは 0。後の運を含まないので、結果の軸の並びが正しく当てられた Ω の保留の R² は 1 に近づく。
   *
   * @param source 元のコホート
   * @param cohort 書く先のコホート
   * @return 書く先のコホート
   * @throws IOException 読み書きできないとき
   */
  static Path outcomeTarget(Path source, Path cohort) throws IOException {
    LuckObservation read = new LuckObservation();
    float[] y = new float[GameState.NUM_PLAYERS];
    try (LuckSampleWriter writer = new LuckSampleWriter(cohort, 1);
        LuckSampleStore.Cursor cursor = LuckSampleStore.open(source).cursor()) {
      while (cursor.next()) {
        read.bind(cursor.words(), 0);
        int tile = LuckSchema.outcomeTile(read.outcome());
        y[0] = read.handCount(0, tile) + (LuckSchema.outcomeAka(read.outcome()) ? 2f : 0f);
        y[1] = read.handCount(1, tile);
        writer.append(cursor.game(), cursor.words(), 0, y, cursor.stratum(), cursor.weight());
      }
      writer.seal();
    }
    return cohort;
  }
}
