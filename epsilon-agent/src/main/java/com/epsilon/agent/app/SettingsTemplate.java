package com.epsilon.agent.app;

import com.epsilon.agent.settings.Settings;

/**
 * 全てのキーを説明つきで書いた、設定ファイルのひな形を出す。設定ファイルと {@code --set} が書いた値はそのまま書き、既定値、
 * プリセット、ルールごとの既定の層の値はコメントにする。ルールを選ぶキーを書けば、そのルールの既定の層の値が入る。
 */
@Named(value = "settings template", doc = "Print a settings file template with every key described and the defaults commented out")
public record SettingsTemplate() implements BuiltinCommand {

  @Override
  public Outcome run(Context<Record> context) {
    Application<Record> application = context.application();
    String template =
        Settings.template(application.schema(), application.stackIfChosen(context.layers()), context.presets());
    context.out().print(template);
    return Outcome.ok(template.lines().count() + " lines");
  }
}
