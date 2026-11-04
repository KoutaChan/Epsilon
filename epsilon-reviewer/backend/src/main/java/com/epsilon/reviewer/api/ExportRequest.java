package com.epsilon.reviewer.api;

/** @param path 保存の画面で選んだ絶対パス */
public record ExportRequest(String resultId, String path) {}
