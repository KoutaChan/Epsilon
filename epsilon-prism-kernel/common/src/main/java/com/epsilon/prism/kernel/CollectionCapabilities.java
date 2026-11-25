package com.epsilon.prism.kernel;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;

/** 収集カーネルが対応する幅とルールを、配布物の宣言から読み取る。 */
public final class CollectionCapabilities {
  private record Rule(int seats, int tiles) {}

  private static final java.util.Set<Rule> RULES = new java.util.HashSet<>();

  public static boolean supportsRule(int seats, int tiles) {
    return RULES.contains(new Rule(seats, tiles));
  }

  private static final Map<String, Width> NAMED_WIDTHS = readWidths();
  private static final Map<Integer, Width> WIDTHS =
      NAMED_WIDTHS.values().stream()
          .collect(Collectors.toUnmodifiableMap(Width::hidden, Function.identity()));

  /** 入力の埋め込みと層 0 の射影を別々に実行するカーネルの幅。 */
  public static final int PROJECTED_WIDTH = NAMED_WIDTHS.get("Projected").hidden();

  private static final Map<String, Integer> LIMITS = readLimits();
  static final int STANDARD_WIDTH =
      WIDTHS.values().stream().filter(Width::standard).mapToInt(Width::hidden).min().orElseThrow();
  static final int U8_CODES = 1 << Byte.SIZE;

  /** ΣC の牌の射影の幅。 */
  public static final int COUNT_WIDTH = limit("CountWidth");

  /** 入力の埋め込みと射影を別に計算する幅で、レーンに保存する射影の行数の上限。 */
  public static int projectionRowLimit(int hidden) {
    Width width = width(hidden);
    return width != null && width.core() && !width.folded() ? limit("ProjectionRows") : 0;
  }

  private CollectionCapabilities() {}

  /** 幅、層 0 の折り込み、基本候補層、読み直し、ΣC と K を持つ候補層の対応。 */
  public record Width(int hidden, boolean folded, boolean standard, boolean reread, boolean core) {
    /** 候補層の部品の組を使えるなら true。 */
    public boolean supports(boolean segments, boolean reread) {
      return (segments ? core : standard) && (!reread || (!segments && this.reread));
    }
  }

  /** 宣言された幅の能力。宣言されていない幅なら null。 */
  public static Width width(int hidden) {
    return WIDTHS.get(hidden);
  }

  /** 指定の形の候補層を実行できるなら true。 */
  public static boolean supports(int hidden, boolean segments, boolean reread) {
    Width width = width(hidden);
    return width != null && width.supports(segments, reread);
  }

  /** この幅では入力の埋め込みに層 0 の射影を折り込む。 */
  public static boolean foldsLayerZero(int hidden) {
    Width width = width(hidden);
    return width != null && width.folded();
  }

  private static Map<String, Width> readWidths() {
    try (var reader = open("/collection_widths.def")) {
      Map<String, Width> widths = new HashMap<>();
      for (String line; (line = reader.readLine()) != null; ) {
        if (line.startsWith("PRISM_RULES(")) {
          var fields = line.substring(line.indexOf('(') + 1, line.length() - 1).split(", ");
          RULES.add(new Rule(Integer.parseInt(fields[0]), Integer.parseInt(fields[1])));
          continue;
        }
        boolean nativeWidth = line.startsWith("PRISM_WIDTH(");
        if (!nativeWidth && !line.startsWith("PRISM_EAGER_WIDTH(")) continue;
        String[] fields = line.substring(line.indexOf('(') + 1, line.length() - 1).split(", ");
        int hidden = Integer.parseInt(fields[1]);
        var width =
            nativeWidth
                ? new Width(
                    hidden, flag(fields[2]), flag(fields[3]), flag(fields[4]), flag(fields[5]))
                : new Width(hidden, false, false, false, false);
        if (widths.putIfAbsent(fields[0], width) != null) {
          throw new IllegalStateException("duplicate PRISM width name " + fields[0]);
        }
      }
      return Map.copyOf(widths);
    } catch (IOException e) {
      throw new IllegalStateException("cannot read native collection capabilities", e);
    }
  }

  static int limit(String name) {
    Integer value = LIMITS.get(name);
    if (value == null) throw new IllegalStateException("missing native collection limit " + name);
    return value;
  }

  private static Map<String, Integer> readLimits() {
    try (var reader = open("/collection_limits.def")) {
      Map<String, Integer> limits = new HashMap<>();
      for (String line; (line = reader.readLine()) != null; ) {
        if (!line.startsWith("PRISM_LIMIT(")) continue;
        String[] fields = line.substring("PRISM_LIMIT(".length(), line.length() - 1).split(", ");
        if (limits.putIfAbsent(fields[0], Integer.parseInt(fields[1])) != null) {
          throw new IllegalStateException("duplicate native collection limit " + fields[0]);
        }
      }
      return Map.copyOf(limits);
    } catch (IOException e) {
      throw new IllegalStateException("cannot read native collection limits", e);
    }
  }

  private static BufferedReader open(String resource) {
    var stream = CollectionCapabilities.class.getResourceAsStream(resource);
    if (stream == null)
      throw new IllegalStateException("missing native collection resource " + resource);
    return new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
  }

  private static boolean flag(String value) {
    return switch (value) {
      case "0" -> false;
      case "1" -> true;
      default -> throw new IllegalArgumentException("invalid collection capability " + value);
    };
  }
}
