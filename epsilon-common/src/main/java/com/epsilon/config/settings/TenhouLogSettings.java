package com.epsilon.config.settings;

/**
 * 天鳳互換JSONを外部ビューアへ渡すための設定。牌譜形式やモデル構造には依存しない。
 *
 * @param viewerUrlPrefix URL 符号化した牌譜JSONの直前へ付けるビューア URL
 */
@SettingsPrefix("epsilon.tenhou")
public record TenhouLogSettings(@Default("https://tenhou.net/6/#json=") String viewerUrlPrefix) {}
