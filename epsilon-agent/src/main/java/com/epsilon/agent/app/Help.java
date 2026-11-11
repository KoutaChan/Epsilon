package com.epsilon.agent.app;

import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.SettingsException;
import java.util.List;

/**
 * コマンドの一覧か、1 つのコマンドの引数の表を出す。
 *
 * @param command 引数の表を出すコマンド。無ければ一覧
 */
@Named(value = "help", doc = "List the commands, or show the arguments of one command")
public record Help(
    @Key(doc = "Command to show the arguments of (the name may be written in parts, as in help settings change)", presence = Presence.ABSENT)
        String command)
    implements BuiltinCommand {

  @Override
  public Outcome run(Context<Record> context) {
    Application<Record> application = context.application();
    Catalog catalog = Catalog.of(application);
    if (command == null) {
      context.out().print(Usage.commands(application, catalog));
      return Outcome.ok(catalog.entries().size() + " commands");
    }
    Catalog.Entry entry = catalog.named(command);
    if (entry == null) {
      throw new SettingsException(
          List.of(
              "Unknown command "
                  + command
                  + Catalog.suggestion(
                      command, catalog.entries().stream().map(Catalog.Entry::name).toList(), " (did you mean ", "?)")));
    }
    context.out().print(Usage.command(application, entry));
    return Outcome.ok(entry.arguments().size() + " arguments of " + entry.name());
  }
}
