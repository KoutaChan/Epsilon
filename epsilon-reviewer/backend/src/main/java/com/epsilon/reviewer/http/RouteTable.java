package com.epsilon.reviewer.http;

import com.epsilon.reviewer.api.Command;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** 起動時に、操作（{@link Command}）ごとの HTTP のパスを / で区切っておく。リクエストのメソッドとパスから、操作と、パスに埋めた引数を探す。 */
final class RouteTable {
  private final List<Template> templates = new ArrayList<>();

  RouteTable() {
    for (Command command : Command.values())
      if (command.httpPath() != null) {
        String[] segments = command.httpPath().substring(1).split("/");
        templates.add(new Template(command, command.httpMethod(), segments));
      }
  }

  /** @param rawPath URL デコードする前のパス（デコードするのはパスに埋めた引数の値だけ） */
  Optional<Route> match(String method, String rawPath) {
    String[] segments = rawPath.substring(1).split("/", -1);
    for (Template template : templates) {
      if (!template.method().equals(method) || template.segments().length != segments.length)
        continue;
      Map<String, String> parameters = new HashMap<>();
      boolean matches = true;
      for (int i = 0; i < segments.length && matches; i++) {
        String pattern = template.segments()[i];
        if (pattern.startsWith("{"))
          parameters.put(pattern.substring(1, pattern.length() - 1), decode(segments[i]));
        else matches = pattern.equals(segments[i]);
      }
      if (matches) return Optional.of(new Route(template.command(), parameters));
    }
    return Optional.empty();
  }

  /** / で区切ったパスの 1 区間を URL デコードする（+ はそのまま残す）。デコードできなければそのまま返し、形の正しくない id として扱わせる。 */
  private static String decode(String segment) {
    try {
      return URLDecoder.decode(segment.replace("+", "%2B"), StandardCharsets.UTF_8);
    } catch (IllegalArgumentException failure) {
      return segment;
    }
  }

  /** 見つけたルート。 */
  record Route(Command command, Map<String, String> parameters) {}

  private record Template(Command command, String method, String[] segments) {}
}
