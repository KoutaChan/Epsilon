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

/** 1 つの応答の書き（JSON・空・文書・添付・失敗）。送り始めたら失敗を書かない（記録して接続を閉じる）。 */
final class HttpReplies {
  private static final String JSON_TYPE = "application/json; charset=utf-8";

  private final HttpExchange exchange;
  private boolean committed;

  HttpReplies(HttpExchange exchange) {
    this.exchange = exchange;
  }

  /** 状態の行を送ったか。 */
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
   * 解析結果の文書。文書は id ごとに変わらないので ETag は W/"&lt;id&gt;" で、合えば中身を読まずに 304。gzip を受ける相手には保存のまま、
   * 受けない相手には展開して送る。
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

  /** Web の書き出しのダウンロード（保存の gzip のまま）。読めなければ添付の頭を付けずに失敗を返す。 */
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
   * 失敗の本文 {"error":{code, message}}。送り始めていれば書かない。
   *
   * @return 書いたか
   */
  boolean error(ErrorCode code, String message) throws IOException {
    if (committed) return false;
    Headers headers = exchange.getResponseHeaders();
    headers.set("Content-Type", JSON_TYPE);
    headers.set("Cache-Control", "no-store");
    send(code.status(), Json.write(new Envelope(ErrorBody.of(code, message))), false);
    return true;
  }

  /** @param head 本文を送らず長さだけ書く（HEAD） */
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
   * Accept-Encoding が coding を受けるか。全ての頭と項目の coding;q=x を読み、coding と * の最後の q を取り
   * （形の違う q は 0）、coding の q があればそれ、無ければ * の q を使い、0 &lt; q ≤ 1 なら受ける。
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

  /** If-None-Match が tag と合うか（弱い比べ、* は全て）。 */
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

  /** HTTP の失敗の封筒。 */
  private record Envelope(ErrorBody error) {}
}
