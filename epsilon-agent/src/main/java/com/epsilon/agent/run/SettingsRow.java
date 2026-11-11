package com.epsilon.agent.run;

import com.epsilon.agent.settings.Change;
import com.epsilon.agent.settings.Difference;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Layer;
import com.epsilon.agent.settings.Node;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.Settings;
import com.epsilon.agent.settings.SettingsException;
import java.time.Instant;
import java.util.List;
import java.util.Objects;

/**
 * 設定の行（settings.jsonl の 1 行）。続いている run の設定の正本で、行は番号の順に足すだけにする。反復は始めるときに最新の行を固定し、
 * その行の値で最後まで回る。次の起動から効くキーだけは、JVM を起動したときの最新の行から読む。
 *
 * @param row 行の番号（1 から）
 * @param from 効き始める反復（行を足した後に始まる最初の反復）
 * @param time 行を足した時刻
 * @param jar 行を書いた jar の ID
 * @param reason 変えた理由
 * @param changes 前の行から変えたキー（1 行目は空）
 * @param settings 全てのキーの値（既定値も含む）
 */
public record SettingsRow(
    int row, int from, Instant time, String jar, String reason, List<ChangedKey> changes, Node.Table settings) {

  public SettingsRow {
    if (row < 1 || from < 1) {
      throw new IllegalArgumentException(
          "A settings row and its first iteration start at 1: row " + row + ", from " + from);
    }
    Objects.requireNonNull(time, "time");
    Objects.requireNonNull(jar, "jar");
    Objects.requireNonNull(reason, "reason");
    changes = List.copyOf(changes);
    Objects.requireNonNull(settings, "settings");
  }

  /** 1 行目に書く理由。 */
  public static final String CREATED = "created";

  /** この行の値をまとめた層。値の出どころは {@code settings row <番号>} と書く。 */
  public Layer layer() {
    return Layer.tree("settings row " + row, settings);
  }

  /**
   * この行の値を、run 全体の設定の型に結び付ける。
   *
   * @throws SettingsException この jar の設定の形と合わない（知らないキー、欠けたキー、型の違い）
   */
  public <R extends Record> Settings<R> bind(Class<R> schema) {
    return Settings.bind(schema, List.of(layer()));
  }

  /**
   * 前の行から変えたキー 1 つ。
   *
   * @param key キーのパス
   * @param before 前の値。前の行に無かったキーなら null
   * @param after 後の値。消したキーなら null
   * @param change このキーをいつ変えられ、いつから効くか
   */
  public record ChangedKey(
      String key,
      @Key(doc = "前の値", presence = Presence.ABSENT) Node before,
      @Key(doc = "後の値", presence = Presence.ABSENT) Node after,
      Change change) {

    public ChangedKey {
      Objects.requireNonNull(key, "key");
      Objects.requireNonNull(change, "change");
      if (before == null && after == null) {
        throw new IllegalArgumentException("A changed key needs a value before or after: " + key);
      }
    }

    public static ChangedKey of(Difference difference) {
      return new ChangedKey(difference.key(), difference.before(), difference.after(), difference.change());
    }
  }
}
