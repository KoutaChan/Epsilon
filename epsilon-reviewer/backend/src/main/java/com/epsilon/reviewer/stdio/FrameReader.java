package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.json.Json;
import com.google.gson.JsonObject;
import java.io.BufferedInputStream;
import java.io.DataInputStream;
import java.io.IOException;
import java.io.InputStream;
import java.util.Optional;

/**
 * 枠を読む: [頭の長さ u32 BE][頭: UTF-8 の JSON][本文の長さ u32 BE][本文]。頭は 16 MiB、本文は 32 MiB まで
 * （protocol/stdio.ts）。上限を超えた枠は読まない（子プロセスを壊れたものとして扱う）。
 */
public final class FrameReader {
  /** 頭の上限（バイト）。history の答えは結果の数だけ伸びるので、5 万件まで入る大きさにする。 */
  public static final int HEADER_LIMIT = 16 * 1024 * 1024;

  /** 本文の上限（バイト）。 */
  public static final int BODY_LIMIT = 32 * 1024 * 1024;

  private final DataInputStream in;

  public FrameReader(InputStream in) {
    this.in = new DataInputStream(new BufferedInputStream(in, 64 * 1024));
  }

  /**
   * 次の枠。流れが枠の境で終われば empty。
   *
   * @throws IOException 枠が途中で切れた・上限を超えた・頭が JSON の物でない
   */
  public Optional<Frame> read() throws IOException {
    int first = in.read();
    if (first < 0) return Optional.empty();
    long headerLength =
        ((long) first << 24) | (in.readUnsignedByte() << 16) | in.readUnsignedShort();
    if (headerLength > HEADER_LIMIT)
      throw new IOException("A frame header of " + headerLength + " bytes exceeds 16 MiB.");
    byte[] header = new byte[(int) headerLength];
    in.readFully(header);
    long bodyLength = Integer.toUnsignedLong(in.readInt());
    if (bodyLength > BODY_LIMIT)
      throw new IOException("A frame body of " + bodyLength + " bytes exceeds 32 MiB.");
    byte[] body = new byte[(int) bodyLength];
    in.readFully(body);
    JsonObject json;
    try {
      json = Json.readObject(header, HEADER_LIMIT);
    } catch (ReviewerException failure) {
      throw new IOException("A frame header is not a JSON object: " + failure.getMessage());
    }
    return Optional.of(new Frame(json, body));
  }
}
