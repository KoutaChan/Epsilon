package com.epsilon.reviewer.scenario;

import com.google.gson.stream.JsonReader;
import java.io.ByteArrayInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import java.util.zip.GZIPInputStream;

/**
 * 保存した文書から判断を探すためのインデックス。局と席ごとに、判断を文書に出てくる順に並べる。各判断には、判断が決まるイベントの位置と、
 * 候補の行動の番号を持たせる。もしもの座標（局・判断が決まる位置・席）は、その局でその席が何番目に行った判断かに読み替える。
 * 前のバージョンのエンジンが書いた文書では、イベントの位置が今の牌譜とずれることがある。そのため位置は文書の中でだけ使う。
 */
final class DocumentDecisions {
  /** [局][席] ごとに、文書に出てくる順に並べた判断。 */
  private final List<List<List<Stored>>> rounds;

  private DocumentDecisions(List<List<List<Stored>>> rounds) {
    this.rounds = rounds;
  }

  /**
   * gzip で圧縮した文書を、全体を展開せずに先頭から順に読む。
   *
   * @param seats 牌譜の席の数
   * @throws IOException 文書が gzip で圧縮した JSON でない
   * @throws IllegalArgumentException 判断の手順の書き方が違う（席が無いか範囲の外、候補が無い、局の {@code firstEvent} より前に
   *     {@code steps} が書かれている）
   */
  static DocumentDecisions read(byte[] document, int seats) throws IOException {
    List<List<List<Stored>>> rounds = new ArrayList<>();
    try (var json =
        new JsonReader(
            new InputStreamReader(
                new GZIPInputStream(new ByteArrayInputStream(document)),
                StandardCharsets.UTF_8))) {
      json.beginObject();
      while (json.hasNext()) {
        if (!json.nextName().equals("rounds")) {
          json.skipValue();
          continue;
        }
        json.beginArray();
        while (json.hasNext()) rounds.add(round(json, seats));
        json.endArray();
      }
      json.endObject();
    }
    return new DocumentDecisions(rounds);
  }

  /**
   * 判断が決まるイベントの位置から、判断を探す。
   *
   * @return その局でその席が何番目に行った判断かと、その候補。無ければ null
   */
  Stored find(int round, int seat, int resolve) {
    if (round < 0 || round >= rounds.size()) return null;
    List<List<Stored>> seats = rounds.get(round);
    if (seat < 0 || seat >= seats.size()) return null;
    for (Stored stored : seats.get(seat)) if (stored.resolve == resolve) return stored;
    return null;
  }

  private static List<List<Stored>> round(JsonReader json, int seats) throws IOException {
    List<List<Stored>> bySeat = new ArrayList<>(seats);
    for (int seat = 0; seat < seats; seat++) bySeat.add(new ArrayList<>());
    int firstEvent = -1;
    json.beginObject();
    while (json.hasNext()) {
      switch (json.nextName()) {
        case "firstEvent" -> firstEvent = json.nextInt();
        case "steps" -> steps(json, firstEvent, bySeat);
        default -> json.skipValue();
      }
    }
    json.endObject();
    return bySeat;
  }

  /**
   * 局の手順を読み、判断を席ごとに足す。イベントの位置は、局の最初のイベントから 1 つずつ増える。判断のきっかけ（{@code cause}）が
   * 書かれていなければ、直前のイベントの位置を使う。判断が決まる位置（{@code resolve}）が書かれていなければ、きっかけの次の位置を使う。
   */
  private static void steps(JsonReader json, int firstEvent, List<List<Stored>> bySeat)
      throws IOException {
    if (firstEvent < 0)
      throw new IllegalArgumentException("A round lists its steps before its first event.");
    int lastEvent = firstEvent - 1;
    json.beginArray();
    while (json.hasNext()) {
      boolean event = false;
      int seat = -1;
      int cause = -1;
      int resolve = -1;
      int[] candidates = null;
      json.beginObject();
      while (json.hasNext()) {
        switch (json.nextName()) {
          case "event" -> {
            event = true;
            json.skipValue();
          }
          case "seat" -> seat = json.nextInt();
          case "cause" -> cause = json.nextInt();
          case "resolve" -> resolve = json.nextInt();
          case "candidates" -> candidates = candidates(json);
          default -> json.skipValue();
        }
      }
      json.endObject();
      if (event) {
        lastEvent++;
        continue;
      }
      if (seat < 0 || seat >= bySeat.size() || candidates == null)
        throw new IllegalArgumentException("A decision step has no seat or candidates.");
      if (cause < 0) cause = lastEvent;
      List<Stored> decisions = bySeat.get(seat);
      decisions.add(new Stored(decisions.size(), resolve < 0 ? cause + 1 : resolve, candidates));
    }
    json.endArray();
  }

  /** 候補の各行から、最初の値（行動の番号）だけを読む。 */
  private static int[] candidates(JsonReader json) throws IOException {
    List<Integer> codes = new ArrayList<>();
    json.beginArray();
    while (json.hasNext()) {
      json.beginArray();
      codes.add(json.nextInt());
      while (json.hasNext()) json.skipValue();
      json.endArray();
    }
    json.endArray();
    return codes.stream().mapToInt(Integer::intValue).toArray();
  }

  /**
   * 文書に書かれた 1 つの判断。
   *
   * @param ordinal その局でその席が何番目に行った判断か（0 から）
   * @param resolve 判断が決まるイベントの位置
   * @param candidates 候補の行動の番号
   */
  record Stored(int ordinal, int resolve, int[] candidates) {}
}
