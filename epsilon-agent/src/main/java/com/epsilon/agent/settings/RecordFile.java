package com.epsilon.agent.settings;

import com.epsilon.agent.settings.Schema.RecordCodec;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * プログラムが書くファイル（state.json・job.json・status.json・設定の行・報告の行）を、設定と同じ厳しさで record に結び付けて読み書きする。
 *
 * <p>フィールドは全て必要で、知らないフィールドはエラーになる。値が null のフィールドは無いものとして読む。無くてよいのは
 * {@link Presence#ABSENT} のコンポーネントと、{@link Key} に既定値を書いたコンポーネントだけ。後者は、無ければ既定値を初期値として入れ、
 * {@link Loaded#added} に並べる（新しい jar が足したフィールド。呼び出し側が反復の区切りで 1 回だけファイルを書き直し、反復の記録に残す）。
 * 設定の木（設定の行の中身）は {@link Node.Table} のコンポーネントに、変えたキーの変更前後の値のような木のノードは {@link Node} の
 * コンポーネントに、そのまま持てる。
 */
public final class RecordFile {

  private RecordFile() {}

  /**
   * 読んだ record と、初期値を入れたフィールド。
   *
   * @param added 初期値を入れたフィールドのパス。全てのフィールドがあれば空
   */
  public record Loaded<R extends Record>(R value, List<String> added) {

    public Loaded {
      Objects.requireNonNull(value, "value");
      added = List.copyOf(added);
    }
  }

  /**
   * JSON の中身を読む。
   *
   * @param origin エラーメッセージに書くファイルの名前
   * @throws SettingsException 読めない、フィールドが欠けている、知らないフィールドがある、または値が型に合わない
   */
  public static <R extends Record> Loaded<R> read(Class<R> type, String json, String origin) {
    return read(type, Json.read(json, origin));
  }

  /** ファイルを読む。 */
  public static <R extends Record> Loaded<R> read(Class<R> type, Path file) throws IOException {
    return read(type, Files.readString(file, StandardCharsets.UTF_8), file.toString());
  }

  /** 読み終えた木（jsonl の行の中の木など）を record に結び付ける。 */
  public static <R extends Record> Loaded<R> read(Class<R> type, Node.Table tree) {
    RecordCodec codec = Schema.record(type);
    Binder binder = new Binder(Binder.Mode.RECORD, Set::of);
    Binder.Bound bound = binder.record(codec, tree, "", Change.INHERIT);
    if (!binder.problems().isEmpty()) {
      throw new SettingsException(binder.problems());
    }
    return new Loaded<>(type.cast(bound.value()), binder.added());
  }

  /** record を木にする。null のコンポーネントは書かない。 */
  public static Node.Table tree(Record record) {
    return (Node.Table) Schema.node(Schema.record(record.getClass()), record, "record");
  }

  /** record を 1 行の JSON にする（改行は付けない）。キーはコンポーネントの順に並ぶ。 */
  public static String json(Record record) {
    return Json.write(tree(record));
  }
}
