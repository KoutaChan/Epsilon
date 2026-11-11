package com.epsilon.agent.run;

import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;

/**
 * 反復を閉じるときに書く値。run の種類が段の報告から作り、{@code Runner} が書く。
 *
 * @param state 次の state（次の反復の番号は、閉じる反復の番号 + 1）
 * @param carry 反復の報告に書く持ち越し
 * @param reports reports の中のほかの jsonl に足す行（ファイルの名前ごと）。書くときは、反復の番号を行の先頭に足す
 * @param <S> state.json の record
 */
public record Closing<S extends Record & Progress>(S state, Record carry, Map<String, Record> reports) {

  public Closing {
    Objects.requireNonNull(state, "state");
    Objects.requireNonNull(carry, "carry");
    reports = Collections.unmodifiableMap(new LinkedHashMap<>(reports));
  }
}
