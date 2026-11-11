package com.epsilon.reviewer.stdio;

import com.google.gson.JsonObject;

/**
 * stdio でやり取りする 1 つのフレーム。
 *
 * @param header ヘッダー（UTF-8 の JSON）
 * @param body 本文。無ければ長さ 0
 */
public record Frame(JsonObject header, byte[] body) {}
