package com.epsilon.prism.hip;

import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.HashMap;
import java.util.Map;

/** 専用の HIP カーネルの対応表。native の宣言を、GPU がない環境でも読める。 */
public final class HipCapabilities {
  private static final Map<Integer, Width> WIDTHS = readWidths();
  private static final Map<String, Integer> LIMITS = readLimits();
  static final int STANDARD_WIDTH =
      WIDTHS.values().stream().filter(Width::standard).mapToInt(Width::hidden).min().orElseThrow();
  static final int U8_CODES = 1 << Byte.SIZE;

  /** ΣC の牌の射影の幅。 */
  public static final int COUNT_WIDTH = limit("CountWidth");

  private HipCapabilities() {}

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

  private static Map<Integer, Width> readWidths() {
    try (var reader = open("/prism_hip_capabilities.def")) {
      Map<Integer, Width> widths = new HashMap<>();
      for (String line; (line = reader.readLine()) != null; ) {
        boolean nativeWidth = line.startsWith("PRISM_WIDTH(");
        if (!nativeWidth && !line.startsWith("PRISM_EAGER_WIDTH(")) continue;
        String[] fields = line.substring(line.indexOf('(') + 1, line.length() - 1).split(", ");
        int hidden = Integer.parseInt(fields[1]);
        var width =
            nativeWidth
                ? new Width(
                    hidden, flag(fields[2]), flag(fields[3]), flag(fields[4]), flag(fields[5]))
                : new Width(hidden, false, false, false, false);
        if (widths.putIfAbsent(hidden, width) != null) {
          throw new IllegalStateException("duplicate PRISM width " + hidden);
        }
      }
      return Map.copyOf(widths);
    } catch (IOException e) {
      throw new IllegalStateException("cannot read native HIP capabilities", e);
    }
  }

  static int limit(String name) {
    Integer value = LIMITS.get(name);
    if (value == null) throw new IllegalStateException("missing native HIP limit " + name);
    return value;
  }

  private static Map<String, Integer> readLimits() {
    try (var reader = open("/prism_hip_limits.def")) {
      Map<String, Integer> limits = new HashMap<>();
      for (String line; (line = reader.readLine()) != null; ) {
        if (!line.startsWith("PRISM_LIMIT(")) continue;
        String[] fields = line.substring("PRISM_LIMIT(".length(), line.length() - 1).split(", ");
        if (limits.putIfAbsent(fields[0], Integer.parseInt(fields[1])) != null) {
          throw new IllegalStateException("duplicate native HIP limit " + fields[0]);
        }
      }
      return Map.copyOf(limits);
    } catch (IOException e) {
      throw new IllegalStateException("cannot read native HIP limits", e);
    }
  }

  private static BufferedReader open(String resource) {
    var stream = HipCapabilities.class.getResourceAsStream(resource);
    if (stream == null) throw new IllegalStateException("missing native HIP resource " + resource);
    return new BufferedReader(new InputStreamReader(stream, StandardCharsets.UTF_8));
  }

  private static boolean flag(String value) {
    return switch (value) {
      case "0" -> false;
      case "1" -> true;
      default -> throw new IllegalArgumentException("invalid HIP capability " + value);
    };
  }
}
