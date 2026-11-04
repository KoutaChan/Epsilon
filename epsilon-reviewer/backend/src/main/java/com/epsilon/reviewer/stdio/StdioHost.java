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
 * stdio の操作。読みの輪は呼んだスレッドで回し、要求ごとに仮想スレッドで Dispatcher を呼んで答えを書く（答えの順は要求の順と限らない）。
 * stdin の終わりで、答えている途中の要求を待ってから Reviewer を閉じる。
 *
 * <p>要求 {id, command, request}（importBytes だけ本文が牌譜）。答え {id, ok: true, response?}（result は
 * response なしで本文が保存の gzip）か {id, ok: false, error}。
 */
public final class StdioHost {
  private static final Logger LOG = LoggerFactory.getLogger(StdioHost.class);
  private static final Caller LOCAL = new Caller.Owner(OwnerId.LOCAL);
  private static final ThreadFactory REQUEST_THREADS =
      Thread.ofVirtual().name("reviewer-stdio-", 1).factory();

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
   * stdin の終わりまで要求を受け、答えている途中の要求を待ってから Reviewer を閉じる（待ちの解析を取り消し、走る解析を止め、モデルを閉じ、
   * 鍵を放す）。
   *
   * @throws IOException 壊れた枠（上限を超えた・途中で切れた・id の無い頭）
   */
  public void run() throws IOException {
    try (var requests = Executors.newThreadPerTaskExecutor(REQUEST_THREADS)) {
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
        // 書けない stdout は次の読みか Electron の側で終わりに気づく。
      }
    } catch (IOException failure) {
      LOG.error("Could not write the reply to {}.", id, failure);
    }
  }

  /** @throws ReviewerException invalid_request（知らない操作・stdio で受けない操作） */
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

  /** @throws IOException id が正の整数でない（答えを返せないので枠の決まりの違反として止める） */
  private static long id(JsonObject header) throws IOException {
    if (header.get("id") instanceof JsonPrimitive id && id.isNumber())
      try {
        long value = id.getAsBigDecimal().longValueExact();
        if (value > 0) return value;
      } catch (ArithmeticException ignored) {
        // 下で断る。
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
