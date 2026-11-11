package com.epsilon.agent.run;

import com.epsilon.agent.settings.Node;
import java.util.Objects;

/**
 * run の種類が、公開したバージョンごとに作るジョブ。
 *
 * @param table 卓の名前（ジョブの名前はバージョンと卓から作る）
 * @param definition ジョブの定義（卓の値とバージョン。run の種類が自分の record に結び付ける）
 */
public record JobSpec(String table, Node.Table definition) {

  public JobSpec {
    Objects.requireNonNull(table, "table");
    Objects.requireNonNull(definition, "definition");
  }
}
