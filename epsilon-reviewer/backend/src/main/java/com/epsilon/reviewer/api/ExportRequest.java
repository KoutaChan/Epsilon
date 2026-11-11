package com.epsilon.reviewer.api;

/** @param path 保存のダイアログで選んだ絶対パス */
public record ExportRequest(String resultId, String path) {}
