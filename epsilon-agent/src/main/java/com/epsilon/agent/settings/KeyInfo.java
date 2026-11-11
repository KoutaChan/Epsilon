package com.epsilon.agent.settings;

/**
 * 設定のキー 1 つの説明。使い方の表示とひな形がこれを読む。
 *
 * @param path キーのパス。名前をキーにしたテーブルでは、名前の所を {@code <name>} と書く
 * @param doc 説明
 * @param type 値の型の説明（英語）
 * @param value 既定値のリテラル。無ければ空
 * @param presence キーが無いときの扱い
 * @param range 数の範囲（{@code [1, 4]}、{@code >= 1}）。無ければ空
 * @param change 変えてよい時期。親に従うキーは親の時期になる。コマンドのセクションでは INHERIT のままのことがある
 * @param secret 秘密のキーか
 * @param condition 種類を選ぶセクションの中のキーなら、その種類を選ぶキーと値（{@code memory.scheme=EMA}）。ほかは空
 */
public record KeyInfo(
    String path,
    String doc,
    String type,
    String value,
    Presence presence,
    String range,
    Change change,
    boolean secret,
    String condition) {}
