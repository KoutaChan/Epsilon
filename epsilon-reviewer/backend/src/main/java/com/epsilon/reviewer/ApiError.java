package com.epsilon.reviewer;

/** 表示言語に依存しない識別子と、英語の診断文を分離したエラー。 */
public record ApiError(String code, String message) {}
