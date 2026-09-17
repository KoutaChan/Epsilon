/**
 * 系列固有の項目・既定値を持つ設定レコードと、同梱設定を選ぶ起動生成処理を提供する。
 *
 * <p>共通の設定型とスナップショットの生成・検証は{@link com.epsilon.config.settings.SettingsLoader}が所有する。
 * 実行中は同じスナップショットから型付きレコードを取得し、設定ファイルや実行環境を再読込しない。
 */
package com.epsilon.nano.config.settings;
