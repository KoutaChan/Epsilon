package com.epsilon.reviewer.stdio;

import com.epsilon.reviewer.Reviewer;
import com.epsilon.reviewer.StoredDocument;
import com.epsilon.reviewer.api.Access;
import com.epsilon.reviewer.api.Caller;
import com.epsilon.reviewer.api.Command;
import com.epsilon.reviewer.api.Dispatcher;
import com.epsilon.reviewer.api.RequestBody;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.error.ReviewerException;
import com.epsilon.reviewer.json.Json;
import com.epsilon.reviewer.json.StrictRecords;
import com.epsilon.reviewer.store.OwnerId;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonPrimitive;
import java.io.IOException;
import java.util.Optional;
import java.util.concurrent.Executors;
import java.util.concurrent.ThreadFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * stdio で要求を受けて応答する。要求を読むループは呼び出したスレッドで回し、要求は reviewer-stdio-&lt;n&gt; の 8 本のスレッドで
 * Dispatcher を呼んで応答を書く。応答の順は、要求の順と同じとは限らない。stdin が終わったら、処理中の要求を待ってから Reviewer を閉じる。
 *
 * <p>要求は {id, command, request} の形で、本文を持てるのは importBytes（本文が牌譜）だけ。応答は、成功なら
 * {id, ok: true, response?}、失敗なら {id, ok: false, error}。result の応答は response を持たず、保存した gzip を本文で返す。
 */
public final class StdioHost {
  private static final Logger LOG = LoggerFactory.getLogger(StdioHost.class);
  private static final Caller LOCAL = new Caller.Owner(OwnerId.LOCAL);

  /**
   * 要求を処理するスレッドの数。Nano・Major の方策は、評価を呼んだスレッドごとに作業領域（ページ固定のバッファを含む）を作り、方策を
   * 閉じるまで持つ。そこで、要求ごとにスレッドを作らず、決まった数のスレッドを使い回す。
   */
  static final int THREADS = 8;

  private static final ThreadFactory REQUEST_THREADS =
      Thread.ofPlatform().name("reviewer-stdio-", 1).factory();

  private final Reviewer reviewer;
  private final Dispatcher dispatcher;
  private final FrameReader in;
  private final FrameWriter out;

  public StdioHost(Reviewer reviewer, FrameReader in, FrameWriter out) {
    this.reviewer = reviewer;
    dispatcher = new Dispatcher(reviewer, null);
    this.in = in;
    this.out = out;
  }

  /**
   * stdin が終わるまで要求を受け、処理中の要求を待ってから Reviewer を閉じる。閉じるときは、待っている解析を取り消し、実行中の解析を止め、
   * モデルを閉じ、ロックを解放する。
   *
   * @throws IOException 壊れたフレームを受けた（上限を超えた・途中で切れた・ヘッダーに id が無い）
   */
  public void run() throws IOException {
    try (var requests = Executors.newFixedThreadPool(THREADS, REQUEST_THREADS)) {
      for (Optional<Frame> frame = in.read(); frame.isPresent(); frame = in.read()) {
        Frame request = frame.get();
        long id = id(request.header());
        requests.execute(() -> answer(id, request));
      }
    } finally {
      reviewer.close();
    }
  }

  private void answer(long id, Frame frame) {
    JsonObject reply = new JsonObject();
    reply.addProperty("id", id);
    byte[] body = null;
    try {
      Command command = command(frame.header());
      byte[] bytes = command.requestBody() == RequestBody.BINARY ? frame.body() : null;
      if (bytes == null && frame.body().length > 0)
        throw new ReviewerException(ErrorCode.INVALID_REQUEST, "Only importBytes takes a body.");
      Object response =
          dispatcher.dispatch(command, LOCAL, request(command, frame.header()), bytes);
      reply.addProperty("ok", true);
      switch (command.responseBody()) {
        case JSON -> reply.add("response", Json.GSON.toJsonTree(response));
        case DOCUMENT -> body = ((StoredDocument) response).gzip().get();
        case EMPTY -> {}
        case ATTACHMENT -> throw new IllegalStateException("Attachments are HTTP only.");
      }
    } catch (ReviewerException failure) {
      reply = failure(id, ErrorBody.of(failure));
      body = null;
    } catch (RuntimeException | Error failure) {
      LOG.error("Could not answer the request {}.", id, failure);
      reply = failure(id, ErrorBody.of(ErrorCode.SERVER_ERROR));
      body = null;
    }
    write(id, reply, body);
  }

  private void write(long id, JsonObject reply, byte[] body) {
    try {
      out.write(reply, body);
    } catch (IllegalArgumentException failure) {
      LOG.error("The reply to {} is too large for a frame.", id, failure);
      try {
        out.write(failure(id, ErrorBody.of(ErrorCode.SERVER_ERROR)), null);
      } catch (IOException ignored) {
        // stdout に書けなければ、次の読み込みか Electron の側で、やり取りが終わったことに気づく。
      }
    } catch (IOException failure) {
      LOG.error("Could not write the reply to {}.", id, failure);
    }
  }

  /** @throws ReviewerException invalid_request（知らないコマンド、または stdio では受けないコマンド） */
  private static Command command(JsonObject header) {
    JsonElement name = header.get("command");
    String text =
        name instanceof JsonPrimitive primitive && primitive.isString()
            ? primitive.getAsString()
            : "";
    return Command.byWireName(text)
        .filter(command -> command.stdio() && command.access().contains(Access.LOCAL))
        .orElseThrow(
            () ->
                new ReviewerException(
                    ErrorCode.INVALID_REQUEST, "Unknown command: " + text + "."));
  }

  private static Record request(Command command, JsonObject header) {
    if (!(header.get("request") instanceof JsonObject fields))
      throw new ReviewerException(ErrorCode.INVALID_REQUEST, "request must be an object.");
    return StrictRecords.read(fields, command.requestType());
  }

  /** @throws IOException id が正の整数でない。応答を返す先が無いので、やり取りの決まりに反したものとして止める */
  private static long id(JsonObject header) throws IOException {
    if (header.get("id") instanceof JsonPrimitive id && id.isNumber())
      try {
        long value = id.getAsBigDecimal().longValueExact();
        if (value > 0) return value;
      } catch (ArithmeticException ignored) {
        // この後で断る。
      }
    throw new IOException("A request frame has no positive integer id.");
  }

  private static JsonObject failure(long id, ErrorBody error) {
    JsonObject reply = new JsonObject();
    reply.addProperty("id", id);
    reply.addProperty("ok", false);
    reply.add("error", Json.GSON.toJsonTree(error));
    return reply;
  }
}
