package com.epsilon.reviewer.http;

import com.epsilon.reviewer.StoredDocument;
import com.epsilon.reviewer.error.ErrorBody;
import com.epsilon.reviewer.error.ErrorCode;
import com.epsilon.reviewer.json.Json;
import com.sun.net.httpserver.Headers;
import com.sun.net.httpserver.HttpExchange;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.OutputStream;
import java.util.List;
import java.util.zip.GZIPInputStream;

/** リクエスト 1 つへの応答を書く（JSON・空・文書・添付・エラー）。送り始めた後はエラーを書かない（呼び出し元がログに残して接続を閉じる）。 */
final class HttpReplies {
  private static final String JSON_TYPE = "application/json; charset=utf-8";

  private final HttpExchange exchange;
  private boolean committed;

  HttpReplies(HttpExchange exchange) {
    this.exchange = exchange;
  }

  /** ステータス行を送り始めたなら true。 */
  boolean committed() {
    return committed;
  }

  void json(byte[] body) throws IOException {
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", JSON_TYPE);
    headers.set("Cache-Control", "no-store");
    send(200, body, false);
  }

  void empty() throws IOException {
    exchange.getResponseHeaders().set("Cache-Control", "no-store");
    committed = true;
    exchange.sendResponseHeaders(204, -1);
  }

  /**
   * 解析結果の文書を送る。文書は id ごとに変わらないので ETag は W/"&lt;id&gt;" とし、合えば中身を読まずに 304 を返す。
   * gzip を受け付ける相手には保存した gzip のまま送り、受け付けない相手には展開して送る。
   */
  void document(StoredDocument document) throws IOException {
    Headers request = exchange.getRequestHeaders();
    String tag = "W/\"" + document.id().value() + "\"";
    if (notModified(request, tag)) {
      documentHeaders(tag);
      notModified();
      return;
    }
    byte[] gzip = document.gzip().get();
    documentHeaders(tag);
    if (accepts(request, "gzip")) {
      exchange.getResponseHeaders().set("Content-Encoding", "gzip");
      send(200, gzip, false);
      return;
    }
    committed = true;
    exchange.sendResponseHeaders(200, 0);
    try (var json = new GZIPInputStream(new ByteArrayInputStream(gzip));
        OutputStream out = exchange.getResponseBody()) {
      json.transferTo(out);
    }
  }

  private void documentHeaders(String tag) {
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", JSON_TYPE);
    headers.set("Cache-Control", "private, no-cache");
    headers.set("ETag", tag);
    headers.set("Vary", "Accept-Encoding");
  }

  /** Web 用の書き出しをダウンロードさせる（保存した gzip のまま）。中身を読めなければ、添付のヘッダーを付けずにエラーを返す。 */
  void attachment(StoredDocument document) throws IOException {
    byte[] gzip = document.gzip().get();
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", "application/gzip");
    headers.set(
        "Content-Disposition",
        "attachment; filename=\"" + document.id().value() + ".epsilon-reviewer.json.gz\"");
    headers.set("Cache-Control", "no-store");
    send(200, gzip, false);
  }

  /**
   * エラーの本文 {"error":{code, message}} を送る。送り始めた後なら書かない。
   *
   * @return 書いたなら true
   */
  boolean error(ErrorCode code, String message) throws IOException {
    if (committed) return false;
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", JSON_TYPE);
    headers.set("Cache-Control", "no-store");
    send(code.status(), Json.write(new Envelope(ErrorBody.of(code, message))), false);
    return true;
  }

  /** @param head true なら本文を送らず、長さだけを書く（HEAD） */
  void send(int status, byte[] body, boolean head) throws IOException {
    committed = true;
    if (head) {
      exchange.getResponseHeaders().set("Content-Length", Integer.toString(body.length));
      exchange.sendResponseHeaders(status, -1);
      return;
    }
    exchange.sendResponseHeaders(status, body.length == 0 ? -1 : body.length);
    if (body.length > 0)
      try (OutputStream out = exchange.getResponseBody()) {
        out.write(body);
      }
  }

  void notModified() throws IOException {
    committed = true;
    exchange.sendResponseHeaders(304, -1);
  }

  /**
   * 相手がその圧縮形式（{@code coding}）を受け付けるかを、Accept-Encoding から決める。全ての Accept-Encoding ヘッダーの
   * 全ての項目を読み、その形式と * のそれぞれについて最後に出た q を取る（q が無ければ 1、形の正しくない q は 0）。その形式の
   * q があればそれを、無ければ * の q を使い、0 &lt; q ≤ 1 なら受け付ける。
   */
  static boolean accepts(Headers request, String coding) {
    Double named = null;
    Double wildcard = null;
    for (String header : request.getOrDefault("Accept-Encoding", List.of())) {
      for (String entry : header.split(",")) {
        String[] parameters = entry.trim().split(";");
        String name = parameters[0].trim();
        if (!name.equalsIgnoreCase(coding) && !name.equals("*")) continue;
        double quality = 1;
        for (int i = 1; i < parameters.length; i++) {
          String parameter = parameters[i].trim();
          if (parameter.startsWith("q="))
            try {
              quality = Double.parseDouble(parameter.substring(2));
            } catch (NumberFormatException failure) {
              quality = 0;
            }
        }
        if (name.equalsIgnoreCase(coding)) named = quality;
        else wildcard = quality;
      }
    }
    Double quality = named != null ? named : wildcard;
    return quality != null && quality > 0 && quality <= 1;
  }

  /** If-None-Match が {@code tag} と合うか（弱い比較で、* は何とでも合う）。 */
  static boolean notModified(Headers request, String tag) {
    String opaque = opaque(tag);
    for (String header : request.getOrDefault("If-None-Match", List.of()))
      for (String candidate : header.split(",")) {
        String value = candidate.trim();
        if (value.equals("*") || opaque(value).equals(opaque)) return true;
      }
    return false;
  }

  private static String opaque(String tag) {
    return tag.startsWith("W/") ? tag.substring(2) : tag;
  }

  /** HTTP のエラー応答の JSON（{"error": ...}）。 */
  private record Envelope(ErrorBody error) {}
}
