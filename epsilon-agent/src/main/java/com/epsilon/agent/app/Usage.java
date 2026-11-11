package com.epsilon.agent.app;

import com.epsilon.agent.settings.KeyInfo;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Settings;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/** 使い方の表示。コマンドの注釈と、引数のキーの注釈から作る。 */
final class Usage {

  private static final String LAYERS =
      "Every command accepts (except the commands that drive a run):\n"
          + "  --settings FILE  settings file\n"
          + "  --preset NAME    bundled preset (repeatable, layered after the presets at the top of the file)\n"
          + "  --set PATH=VALUE override of one key (repeatable)";
  private Usage() {}

  /** コマンドの一覧。 */
  static String commands(Application<?> application, Catalog catalog) {
    int width = catalog.entries().stream().mapToInt(entry -> entry.name().length()).max().orElse(0);
    StringBuilder out = new StringBuilder();
    out.append("Usage: ").append(application.name()).append(" <command> [--name value ...]\n");
    out.append("Arguments: ").append(application.name()).append(" help <command>\n\n");
    out.append("Commands:\n");
    for (Catalog.Entry entry : catalog.entries()) {
      out.append("  ").append(pad(entry.name(), width)).append("  ").append(entry.named().doc()).append('\n');
    }
    out.append('\n').append(LAYERS).append('\n');
    return out.toString();
  }

  /** 1 つのコマンドの引数の表。 */
  static String command(Application<?> application, Catalog.Entry entry) {
    List<String[]> rows = new ArrayList<>();
    StringBuilder synopsis = new StringBuilder(application.name()).append(' ').append(entry.name());
    Map<String, KeyInfo> keys = new LinkedHashMap<>();
    for (KeyInfo info : Settings.describe(entry.type())) {
      keys.put(info.path(), info);
    }
    entry.arguments().forEach(
        (argument, component) -> {
          KeyInfo info = keys.get(component.getName());
          boolean repeated = info.type().endsWith(" list");
          String value = "<" + (repeated ? info.type().substring(0, info.type().length() - 5) : info.type()) + ">";
          String form = "--" + argument + " " + value;
          boolean required = info.presence() == Presence.REQUIRED;
          synopsis.append(' ').append(required ? form : "[" + form + "]").append(repeated ? " ..." : "");
          List<String> notes = new ArrayList<>();
          notes.add(
              switch (info.presence()) {
                case REQUIRED -> "required";
                case ABSENT -> "optional";
                case DEFAULT -> "default " + (info.value().isEmpty() ? "none" : info.value());
              });
          if (!info.range().isEmpty()) {
            notes.add(info.range());
          }
          if (repeated) {
            notes.add("repeatable");
          }
          if (info.secret()) {
            notes.add("settings file only");
          }
          rows.add(new String[] {form, String.join(", ", notes), info.doc()});
        });
    StringBuilder out = new StringBuilder(synopsis).append("\n  ").append(entry.named().doc()).append('\n');
    if (!rows.isEmpty()) {
      int formWidth = rows.stream().mapToInt(row -> row[0].length()).max().orElse(0);
      int noteWidth = rows.stream().mapToInt(row -> row[1].length()).max().orElse(0);
      out.append("\nArguments:\n");
      for (String[] row : rows) {
        out.append("  ").append(pad(row[0], formWidth)).append("  ").append(pad(row[1], noteWidth)).append("  ")
            .append(row[2]).append('\n');
      }
    }
    out.append('\n')
        .append(
            entry.named().reads() == Reads.RUN
                ? "Reads the settings recorded in the run. Change them only with " + application.name() + " settings change --run DIR."
                : LAYERS + "\nThe arguments can also be written in the settings file, in the command section [" + entry.section() + "].")
        .append('\n');
    return out.toString();
  }


  private static String pad(String text, int width) {
    return text.length() >= width ? text : text + " ".repeat(width - text.length());
  }
}
