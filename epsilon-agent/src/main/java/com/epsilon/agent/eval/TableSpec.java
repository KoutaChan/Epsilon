package com.epsilon.agent.eval;

import com.epsilon.agent.settings.InvalidSetting;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Range;
import com.epsilon.mahjong.engine.DrawOrder;
import java.nio.file.InvalidPathException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * 評価の卓 1 つの設定。キーは {@code onPromotion} を除いて全て必須で、既定値が無い。そのため、どの値で測ったかが設定だけで分かる。
 *
 * <p>相手の書き方: {@code heuristic}、{@code nano:<絶対パス>}、{@code major:<絶対パス>}、{@code prism:<絶対パス>}、
 * {@code champion}（ジョブを始めるときの王者）、{@code version:<番号>}（その番号のバージョン）、{@code version:-<距離>}（測るバージョンより
 * 距離の分だけ前のバージョン）。
 *
 * <p>相手が {@code version:<番号>} の卓は系譜の基準の卓で、どのバージョンでも同じ牌山を打つ。この卓には昇格したバージョンも測るか
 * （{@code onPromotion}）を必ず書き、ほかの卓には書けない。
 *
 * <p>王者を替える卓は、良いのときに打つ確かめの卓を書ける。確かめの卓は幅を書き、その決まりは {@link EvaluationSettings} が確かめる。
 *
 * @param opponent 相手
 * @param seating 席の決め方
 * @param draws 局のツモの順
 * @param wallSet 牌山の集め方
 * @param walls 牌山の数。途中で止まりうる止め方では上限
 * @param stop 止め方
 * @param score 判定に使う量
 * @param onPromotion 王者との決闘で昇格したバージョンも測るか（基準の卓だけ。ほかの卓では null）。相手のバージョンより新しいバージョンなら、
 *     最初のバージョンより前でも測る
 * @param confirm 「良い」のときに打つ確かめの卓の名前（王者を替える卓だけ。無ければ確かめずに昇格させる）
 * @param confirmMargin 確かめの卓の幅（判定に使う量の単位）。候補 − 王者の対の差の区間の下端が、この幅にマイナスを付けた値を上回れば昇格させる
 *     （確かめの卓だけ。ほかの卓では null）
 * @param every 測るバージョンの間隔（0 なら定期的には測らない）
 * @param from 測る最初のバージョン
 */
public record TableSpec(
    @Key(doc = "相手（heuristic、nano:絶対パス、major:絶対パス、prism:絶対パス、champion、version:番号、version:-距離）",
            presence = Presence.REQUIRED)
        String opponent,
    @Key(doc = "席の決め方", presence = Presence.REQUIRED) SeatingSettings seating,
    @Key(doc = "局のツモの順", presence = Presence.REQUIRED) DrawOrder draws,
    @Key(doc = "牌山の集め方", presence = Presence.REQUIRED) WallSet wallSet,
    @Key(doc = "牌山の数（途中で止まりうる止め方では上限）", presence = Presence.REQUIRED) @Range(min = 2) int walls,
    @Key(doc = "止め方", presence = Presence.REQUIRED) StopSettings stop,
    @Key(doc = "判定に使う量", presence = Presence.REQUIRED) Score score,
    @Key(doc = "王者との決闘で昇格したバージョンも測るか（相手が version:番号 の卓だけに書く）", presence = Presence.ABSENT)
        Boolean onPromotion,
    @Key(doc = "「良い」のときに打つ確かめの卓（王者を替える卓だけに書く）", presence = Presence.ABSENT) String confirm,
    @Key(doc = "確かめの卓の幅（候補 − 王者の対の差の区間の下端が、この幅にマイナスを付けた値を上回れば昇格させる。確かめの卓だけに書く）",
            presence = Presence.ABSENT)
        @Range(min = 0)
        Double confirmMargin,
    @Key(doc = "測るバージョンの間隔（0 なら定期的には測らない）", presence = Presence.REQUIRED) @Range(min = 0) int every,
    @Key(doc = "測る最初のバージョン", presence = Presence.REQUIRED) @Range(min = 0) int from) {

  /** 判定の時点の数の上限。 */
  public static final int MAX_LOOKS = 32;

  private static final Pattern CHECKPOINT = Pattern.compile("(nano|major|prism):(.+)");
  private static final Pattern VERSION = Pattern.compile("version:(-?)([0-9]+)");

  /** 相手の書き方の種類。 */
  public enum Opponent {
    /** ルールベースの打ち手。 */
    HEURISTIC,
    /** 絶対パスで指定したチェックポイント。 */
    CHECKPOINT,
    /** ジョブを始めるときの王者。 */
    CHAMPION,
    /** 番号で指定したバージョン。 */
    VERSION,
    /** 測るバージョンより距離の分だけ前のバージョン。 */
    DISTANCE
  }

  public TableSpec {
    Objects.requireNonNull(opponent, "opponent");
    Objects.requireNonNull(seating, "seating");
    Objects.requireNonNull(draws, "draws");
    Objects.requireNonNull(wallSet, "wallSet");
    Objects.requireNonNull(stop, "stop");
    Objects.requireNonNull(score, "score");
    Opponent kind = kind(opponent);
    if (kind == Opponent.VERSION) {
      int version = versionNumber(opponent);
      if (version < 1 || version >= from) {
        throw new InvalidSetting("opponent", "version:N needs 1 <= N < from = " + from + ": " + opponent);
      }
    }
    if (kind == Opponent.DISTANCE) {
      int distance = versionNumber(opponent);
      if (distance < 1 || from - distance < 1) {
        throw new InvalidSetting(
            "opponent", "version:-d needs d >= 1 and from - d >= 1 (from = " + from + "): " + opponent);
      }
    }
    if (stop instanceof StopSettings.Test && wallSet != WallSet.FRESH) {
      throw new InvalidSetting("wallSet", "a TEST table plays fresh walls for every job");
    }
    if (stop instanceof StopSettings.Test test && test.promote() && kind != Opponent.CHAMPION) {
      throw new InvalidSetting("opponent", "a table that promotes plays the champion, not " + opponent);
    }
    int checkEvery = stop.checkEvery();
    if (checkEvery > 0) {
      if (walls % checkEvery != 0) {
        throw new InvalidSetting("walls", "walls " + walls + " is not a multiple of checkEvery " + checkEvery);
      }
      if (walls / checkEvery > MAX_LOOKS) {
        throw new InvalidSetting(
            "walls", "walls " + walls + " / checkEvery " + checkEvery + " exceeds " + MAX_LOOKS + " looks");
      }
    }
    if (score == Score.RANK && stop instanceof StopSettings.Test test && test.evenMargin() > 0.5) {
      throw new InvalidSetting("stop.evenMargin", "a rank margin must not exceed 0.5: " + test.evenMargin());
    }
    if (confirm != null && !(stop instanceof StopSettings.Test test && test.promote())) {
      throw new InvalidSetting("confirm", "only a table that promotes the champion names a confirm table");
    }
    if (score == Score.RANK && confirmMargin != null && confirmMargin > 0.5) {
      throw new InvalidSetting("confirmMargin", "a rank margin must not exceed 0.5: " + confirmMargin);
    }
    if (kind == Opponent.VERSION) {
      if (onPromotion == null) {
        throw new InvalidSetting(
            "onPromotion", "a version:N table says whether it also measures the promoted versions");
      }
      if (wallSet != WallSet.FIXED) {
        throw new InvalidSetting(
            "wallSet", "a version:N table is a baseline and plays the same walls for every version");
      }
    } else if (onPromotion != null) {
      throw new InvalidSetting("onPromotion", "only a version:N table measures the promoted versions, not " + opponent);
    }
  }

  /**
   * 相手の書き方から種類を求める。
   *
   * @throws InvalidSetting 書き方が誤っているとき
   */
  public static Opponent kind(String opponent) {
    if (opponent.equals("heuristic")) {
      return Opponent.HEURISTIC;
    }
    if (opponent.equals("champion")) {
      return Opponent.CHAMPION;
    }
    Matcher version = VERSION.matcher(opponent);
    if (version.matches()) {
      return version.group(1).isEmpty() ? Opponent.VERSION : Opponent.DISTANCE;
    }
    Matcher checkpoint = CHECKPOINT.matcher(opponent);
    if (checkpoint.matches() && absolute(checkpoint.group(2))) {
      return Opponent.CHECKPOINT;
    }
    throw new InvalidSetting(
        "opponent",
        "expected heuristic, nano:/path, major:/path, prism:/path, champion, version:N or version:-d, not " + opponent);
  }

  private static boolean absolute(String path) {
    try {
      return Path.of(path).isAbsolute();
    } catch (InvalidPathException invalid) {
      return false;
    }
  }

  /** version:N の N か、version:-d の d。 */
  private static int versionNumber(String opponent) {
    Matcher matcher = VERSION.matcher(opponent);
    if (!matcher.matches()) {
      throw new IllegalArgumentException("Not a version opponent: " + opponent);
    }
    try {
      return Integer.parseInt(matcher.group(2));
    } catch (NumberFormatException overflow) {
      throw new InvalidSetting("opponent", "the version number is too large: " + opponent);
    }
  }

  /**
   * @param version バージョン
   * @return このバージョンを定期的に測るか（間隔が正で、最初のバージョン以後にある間隔の倍数のバージョンなら測る）
   */
  public boolean scheduled(int version) {
    return every > 0 && version >= Math.max(from, 1) && version % every == 0;
  }

  /**
   * @param version 測るバージョン
   * @return 相手のバージョン（{@code version:N} なら N、{@code version:-d} なら version − d）
   * @throws IllegalStateException 相手が系譜のバージョンでないとき
   */
  public int opponentVersion(int version) {
    return switch (kind(opponent)) {
      case VERSION -> versionNumber(opponent);
      case DISTANCE -> version - versionNumber(opponent);
      case HEURISTIC, CHECKPOINT, CHAMPION ->
          throw new IllegalStateException("The opponent " + opponent + " is not a fixed version of the lineage");
    };
  }

  /**
   * 公開した最新のバージョンまでに定期的に測るバージョンを、古い順に返す。
   *
   * @param latest 公開した最新のバージョン
   */
  public List<Integer> versions(int latest) {
    List<Integer> versions = new ArrayList<>();
    if (every > 0) {
      for (int version = Math.ceilDiv(Math.max(from, 1), every) * every; version <= latest; version += every) {
        versions.add(version);
      }
    }
    return versions;
  }
}
