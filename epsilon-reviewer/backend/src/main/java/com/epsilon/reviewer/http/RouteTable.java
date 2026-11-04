package com.epsilon.reviewer.http;

import com.epsilon.reviewer.api.Command;
import java.net.URLDecoder;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;

/** Command の HTTP の道を起動時に節へ分けておき、方法と道から操作と道の欄を探す。 */
final class RouteTable {
  private final List<Template> templates = new ArrayList<>();

  RouteTable() {
    for (Command command : Command.values())
      if (command.httpPath() != null) {
        String[] segments = command.httpPath().substring(1).split("/");
        templates.add(new Template(command, command.httpMethod(), segments));
      }
  }

  /** @param rawPath URL を戻す前の道（欄の値だけを戻す） */
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

  /** 道の節を戻す（+ はそのまま）。戻せない節は形の違う id として扱われるのでそのまま返す。 */
  private static String decode(String segment) {
    try {
      return URLDecoder.decode(segment.replace("+", "%2B"), StandardCharsets.UTF_8);
    } catch (IllegalArgumentException failure) {
      return segment;
    }
  }

  /** 見つけた道。 */
  record Route(Command command, Map<String, String> parameters) {}

  private record Template(Command command, String method, String[] segments) {}
}
