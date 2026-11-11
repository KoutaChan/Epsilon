package com.epsilon.grp.train;

import com.epsilon.grp.GrpState;
import com.epsilon.mahjong.rules.Variant;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.stream.IntStream;

/**
 * 記録した全ての遷移が、epsilon-mahjong のルール（Settlement と Progression）で再現できるかを数える。遷移は半荘ごとに東 1 局の
 * 始まりの卓からたどり、局が終わるたびに次の局の卓へ進み、最後は最終の点で終わる。遷移は 3 つに分ける。局の結果のネットワークが
 * 表せる結果で再現できたもの、包か流し満貫でだけ再現できたもの、どの結果でも再現できなかったもの（外れ）。外れがあれば、記録を
 * 作ったエンジンと天鳳のルールが違うか、ルールの実装が誤っている。どちらなのかは外れの例を見て決める。
 */
public final class RuleCheck {

  /**
   * 外れた遷移。
   *
   * @param gameId 半荘の ID
   * @param hand 半荘の中で何局目か（0 が東 1 局）
   * @param from その局の始まりの卓
   * @param next 半荘が続くときの、次の局の卓。終わるときは null
   * @param finalScores 半荘が終わるときの最終の点。続くときは null
   */
  public record Mismatch(long gameId, int hand, GrpState from, GrpState next, int[] finalScores) {}

  /**
   * 数えた結果。
   *
   * @param transitions 遷移の数
   * @param outcomes 局の結果のネットワークが表せる結果で再現できた数
   * @param extras 包か流し満貫でだけ再現できた数
   * @param mismatches 外れの数
   * @param examples 外れの例。半荘の順に、先頭から上限の数まで
   */
  public record Count(long transitions, long outcomes, long extras, long mismatches, List<Mismatch> examples) {}

  private RuleCheck() {}

  /**
   * 半荘の遷移を数える。半荘ごとに並列に数え、結果は半荘の順に足す。そのため結果はスレッドの実行順に依らない。
   *
   * @param examples 残す外れの例の数の上限
   */
  public static Count count(Variant variant, List<GameRecord> games, int examples) {
    Objects.requireNonNull(variant, "variant");
    if (examples < 0) {
      throw new IllegalArgumentException("Negative example count " + examples);
    }
    ThreadLocal<TransitionSearch> searches = ThreadLocal.withInitial(() -> new TransitionSearch(variant));
    List<Count> counts = IntStream.range(0, games.size())
        .parallel()
        .mapToObj(index -> count(searches.get(), games.get(index), variant))
        .toList();
    long transitions = 0;
    long outcomes = 0;
    long extras = 0;
    long mismatches = 0;
    List<Mismatch> kept = new ArrayList<>();
    for (Count count : counts) {
      transitions += count.transitions();
      outcomes += count.outcomes();
      extras += count.extras();
      mismatches += count.mismatches();
      for (Mismatch mismatch : count.examples()) {
        if (kept.size() < examples) {
          kept.add(mismatch);
        }
      }
    }
    return new Count(transitions, outcomes, extras, mismatches, List.copyOf(kept));
  }

  private static Count count(TransitionSearch search, GameRecord game, Variant variant) {
    if (game.variant() != variant) {
      throw new IllegalArgumentException("A " + game.variant() + " game among " + variant + " games");
    }
    List<GrpState> handEnds = game.handEnds();
    long outcomes = 0;
    long extras = 0;
    List<Mismatch> mismatches = new ArrayList<>();
    for (int hand = 0; hand <= handEnds.size(); hand++) {
      GrpState from = hand == 0 ? null : handEnds.get(hand - 1);
      if (from == null) {
        search.fromStart();
      } else {
        search.from(from);
      }
      boolean last = hand == handEnds.size();
      int found = last ? search.last(game.finalScores()) : search.next(handEnds.get(hand));
      if (found > 0) {
        outcomes++;
      } else if (search.extras()) {
        extras++;
      } else {
        GrpState start = from != null ? from : first(variant);
        mismatches.add(new Mismatch(
            game.gameId(), hand, start, last ? null : handEnds.get(hand), last ? game.finalScores() : null));
      }
    }
    return new Count(handEnds.size() + 1, outcomes, extras, mismatches.size(), mismatches);
  }

  /** 半荘の始めの卓（東 1 局、全員が配給原点）を返す。 */
  private static GrpState first(Variant variant) {
    int[] scores = new int[variant.seats()];
    Arrays.fill(scores, variant.startPoints());
    return new GrpState(variant, 0, 0, 0, scores);
  }
}
