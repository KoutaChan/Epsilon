package com.epsilon.mahjong.record;

import com.epsilon.mahjong.rules.Variant;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/**
 * 牌譜の 1 半荘。牌譜の形式には依らない。ルールは牌譜を読む側が決める。配列はコピーして持つ。
 *
 * @param rounds 半荘の局を、行った順に並べたもの
 * @param finalScores 終局の点（絶対席の順）。残りの供託はトップへ渡した後の点
 */
public record RecordedGame(Variant variant, List<RecordedRound> rounds, int[] finalScores, Meta meta) {

  /** 牌譜を読んだ形式。 */
  public enum Source {
    /** 天鳳の牌譜の XML（mjloggm）。 */
    TENHOU,
    /** mjai のイベントの列。 */
    MJAI,
    /** 雀魂の牌譜を復号した JSON。 */
    MAJSOUL
  }

  /**
   * 牌譜の出どころ。
   *
   * @param id 牌譜 id
   * @param source 牌譜を読んだ形式
   * @param names 打ち手の名前（絶対席の順）。名前の無い席は空の文字列
   */
  public record Meta(String id, Source source, List<String> names) {

    public Meta {
      Objects.requireNonNull(id, "id");
      Objects.requireNonNull(source, "source");
      names = List.copyOf(names);
    }
  }

  public RecordedGame {
    Objects.requireNonNull(variant, "variant");
    Objects.requireNonNull(meta, "meta");
    if (meta.names().size() != variant.seats()) {
      throw new IllegalArgumentException(
          variant + " needs " + variant.seats() + " names, got " + meta.names().size());
    }
    rounds = List.copyOf(rounds);
    if (rounds.isEmpty()) {
      throw new IllegalArgumentException("A game has at least one round");
    }
    for (RecordedRound round : rounds) {
      if (round.start().variant() != variant) {
        throw new IllegalArgumentException(
            "Variant mismatch: expected " + variant + " but found " + round.start().variant());
      }
    }
    if (finalScores.length != variant.seats()) {
      throw new IllegalArgumentException(
          variant + " needs " + variant.seats() + " final scores, got " + finalScores.length);
    }
    finalScores = finalScores.clone();
  }

  /** コピーを返す。 */
  @Override
  public int[] finalScores() {
    return finalScores.clone();
  }

  @Override
  public boolean equals(Object other) {
    return other instanceof RecordedGame game
        && variant == game.variant
        && rounds.equals(game.rounds)
        && Arrays.equals(finalScores, game.finalScores)
        && meta.equals(game.meta);
  }

  @Override
  public int hashCode() {
    return Objects.hash(variant, rounds, Arrays.hashCode(finalScores), meta);
  }

  @Override
  public String toString() {
    return "RecordedGame[" + variant + " " + meta.source() + " " + meta.id() + " names=" + meta.names() + " rounds="
        + rounds + " final=" + Arrays.toString(finalScores) + "]";
  }
}
