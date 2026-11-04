package com.epsilon.reviewer.stdio;

import com.google.gson.JsonObject;

/**
 * 1 枠。
 *
 * @param header UTF-8 の JSON の頭
 * @param body 本文（無ければ長さ 0）
 */
public record Frame(JsonObject header, byte[] body) {}
