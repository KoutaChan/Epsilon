package com.epsilon.classic.learn;

import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.ConcurrentLinkedQueue;

/**
 * 1 回の macro（収集と学習を 1 回ずつ行う反復）で集めた軌跡。半荘を半荘 ID の順に並べ、判断に通しの番号を振る。番号は半荘の順で、
 * 半荘の中は判断が起きた順。並びは集めた順に依らないので、同じ半荘を集めれば同じ学習になる。作った後は変わらないので、どのスレッド
 * から読んでもよい。
 */
public final class TrajectoryBatch {

  private final Variant variant;
  private final List<GameTrajectory> games;

  /** [半荘] → 最初の判断の通しの番号。末尾は判断の総数。 */
  private final int[] starts;

  private TrajectoryBatch(Variant variant, List<GameTrajectory> games) {
    this.variant = variant;
    this.games = games;
    starts = new int[games.size() + 1];
    for (int game = 0; game < games.size(); game++) {
      starts[game + 1] = Math.addExact(starts[game], games.get(game).size());
    }
  }

  /**
   * 半荘の軌跡を半荘 ID の順に並べる。
   *
   * @throws IllegalArgumentException ルールの違う半荘がある。または半荘 ID が同じ半荘が 2 つある
   */
  public static TrajectoryBatch of(Variant variant, List<GameTrajectory> games) {
    Objects.requireNonNull(variant, "variant");
    List<GameTrajectory> sorted = new ArrayList<>(games);
    sorted.sort(Comparator.comparingLong(GameTrajectory::id));
    for (int game = 0; game < sorted.size(); game++) {
      GameTrajectory trajectory = sorted.get(game);
      if (trajectory.variant() != variant) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + variant + " but found " + trajectory.variant());
      }
      if (game > 0 && sorted.get(game - 1).id() == trajectory.id()) {
        throw new IllegalArgumentException("Game " + trajectory.id() + " appears twice");
      }
    }
    return new TrajectoryBatch(variant, List.copyOf(sorted));
  }

  public Variant variant() {
    return variant;
  }

  /** 半荘 ID の順に並べた半荘。 */
  public List<GameTrajectory> games() {
    return games;
  }

  /** 判断の総数。 */
  public int size() {
    return starts[games.size()];
  }

  /** 半荘の最初の判断の通しの番号。 */
  public int start(int game) {
    return starts[Objects.checkIndex(game, games.size())];
  }

  /** 判断が属する半荘の番号（{@link #games} の何番目か）。 */
  public int gameIndex(int decision) {
    Objects.checkIndex(decision, size());
    int found = Arrays.binarySearch(starts, 0, games.size(), decision);
    if (found >= 0) {
      while (found + 1 < games.size() && starts[found + 1] == decision) {
        found++;
      }
      return found;
    }
    return -found - 2;
  }

  /**
   * 卓を回す処理が終えた半荘を集めていく受け取り先。どのスレッドから足してもよい。
   */
  public static final class Collector {

    private final Variant variant;
    private final ConcurrentLinkedQueue<GameTrajectory> games = new ConcurrentLinkedQueue<>();

    public Collector(Variant variant) {
      this.variant = Objects.requireNonNull(variant, "variant");
    }

    /** 終えた半荘を足す。 */
    public void add(GameTrajectory game) {
      if (game.variant() != variant) {
        throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + game.variant());
      }
      games.add(game);
    }

    /** 足した半荘の数。 */
    public int games() {
      return games.size();
    }

    /** 足し終えた半荘を半荘 ID の順に並べる。 */
    public TrajectoryBatch build() {
      return of(variant, new ArrayList<>(games));
    }
  }
}
