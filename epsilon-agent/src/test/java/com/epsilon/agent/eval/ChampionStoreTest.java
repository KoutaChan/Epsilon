package com.epsilon.agent.eval;

import static org.testng.Assert.assertEquals;
import static org.testng.Assert.assertFalse;
import static org.testng.Assert.assertNotEquals;
import static org.testng.Assert.assertNull;
import static org.testng.Assert.assertTrue;
import static org.testng.Assert.expectThrows;

import com.epsilon.agent.eval.ChampionStore.Champion;
import com.epsilon.agent.eval.ChampionStore.Entry;
import com.epsilon.agent.eval.StoreEvent.Initial;
import com.epsilon.agent.eval.StoreEvent.NotConfirmed;
import com.epsilon.agent.eval.StoreEvent.Paused;
import com.epsilon.agent.eval.StoreEvent.Promoted;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Instant;
import java.util.List;
import org.testng.annotations.Test;

/**
 * 王者の保存先を確かめる。参照の書き替えの途中で落ちたときの復旧、読むだけの開き方、run の外から置いた王者だけの SHA-256、
 * 同じイベントの再適用、つじつまの合わない記録、確かめの卓のディレクトリの分け方を見る。
 */
public class ChampionStoreTest {

  private static final String DUEL_2 = "v00002-champion";
  private static final String DUEL_3 = "v00003-champion";

  /** 保存先と、中身の違う 3 つのチェックポイント。 */
  private record Fixture(Path store, Champion a, Champion b, Champion c) {

    static Fixture create(String name) throws IOException {
      Path root = Files.createTempDirectory(name);
      return new Fixture(
          root.resolve("run").resolve("champion"),
          Champion.of("v00001", checkpoint(root, "v1", "a"), 1),
          Champion.of("v00002", checkpoint(root, "v2", "b"), 2),
          Champion.of("v00003", checkpoint(root, "v3", "c"), 3));
    }

    Path current() {
      return store.resolve(ChampionStore.CURRENT);
    }

    Path history() {
      return store.resolve(ChampionStore.HISTORY);
    }

    Path temporary() {
      return store.resolve("." + ChampionStore.CURRENT + ".7.tmp");
    }
  }

  private static Path checkpoint(Path root, String name, String weights) throws IOException {
    Path directory = root.resolve(name);
    Files.createDirectories(directory);
    Files.writeString(directory.resolve("weights.bin"), weights, StandardCharsets.UTF_8);
    Files.writeString(directory.resolve("model.json"), "{}", StandardCharsets.UTF_8);
    return directory;
  }

  @Test
  public void aCrashAfterTheTemporaryFileAndBeforeTheRenameKeepsThePreviousReference() throws Exception {
    Fixture f = Fixture.create("champion-crash-rename");
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    byte[] previous = Files.readAllBytes(f.current());
    store.append(new Promoted(f.b(), DUEL_2));
    byte[] next = Files.readAllBytes(f.current());
    byte[] history = Files.readAllBytes(f.history());

    // 記録に行を足し、一時ファイルを書いて fsync した後、rename の前に落ちた形に戻す
    Files.write(f.current(), previous);
    Files.write(f.temporary(), next);
    assertEquals(store.current().orElseThrow(), f.a());

    // 開き直すと一時ファイルを消し、参照を記録の最後の王者に書き直す
    ChampionStore reopened = ChampionStore.open(f.store());
    assertFalse(Files.exists(f.temporary()));
    assertEquals(Files.readAllBytes(f.current()), next);
    assertEquals(Files.readAllBytes(f.history()), history);
    assertEquals(reopened.current().orElseThrow(), f.b());
    assertEquals(reopened.past(), List.of(f.a()));
  }

  @Test
  public void aViewReadsTheRecordAndWritesNothing() throws Exception {
    Fixture f = Fixture.create("champion-view");
    assertTrue(ChampionStore.view(f.store()).current().isEmpty());
    assertFalse(Files.exists(f.store()));
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    byte[] previous = Files.readAllBytes(f.current());
    store.append(new Promoted(f.b(), DUEL_2));
    byte[] next = Files.readAllBytes(f.current());
    byte[] history = Files.readAllBytes(f.history());

    // 書く側が記録に行を足し、一時ファイルを書いて fsync した後、rename する前の形
    Files.write(f.current(), previous);
    Files.write(f.temporary(), next);
    ChampionStore view = ChampionStore.view(f.store());
    assertEquals(view.current().orElseThrow(), f.b());
    assertEquals(view.past(), List.of(f.a()));
    assertEquals(view.history().size(), 2);
    expectThrows(IllegalStateException.class, () -> view.append(new Promoted(f.c(), DUEL_3)));
    assertEquals(Files.readAllBytes(f.temporary()), next);
    assertEquals(Files.readAllBytes(f.current()), previous);
    assertEquals(Files.readAllBytes(f.history()), history);
  }

  @Test
  public void reapplyingTheDuelAfterACrashBeforeTheReferenceIsWrittenRepairsIt() throws Exception {
    Fixture f = Fixture.create("champion-crash-reapply");
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    byte[] previous = Files.readAllBytes(f.current());
    store.append(new Promoted(f.b(), DUEL_2));
    byte[] next = Files.readAllBytes(f.current());
    byte[] history = Files.readAllBytes(f.history());

    // 記録に足した後、参照を書く前に落ちた形にする。開き直さずに、ジョブの処理が同じイベントをもう一度適用する
    Files.write(f.current(), previous);
    assertEquals(store.current().orElseThrow(), f.a());
    assertFalse(store.append(new Promoted(f.b(), DUEL_2)));
    assertEquals(Files.readAllBytes(f.current()), next);
    assertEquals(Files.readAllBytes(f.history()), history);
    assertEquals(store.current().orElseThrow(), f.b());
  }

  @Test
  public void onlyAChampionPlacedFromOutsideTheRunCarriesAndChecksItsSha256() throws Exception {
    Fixture f = Fixture.create("champion-sha");
    Path outside = checkpoint(f.store().getParent().getParent(), "outside", "o");
    Champion placed = Champion.placed("outside", outside, 0);
    assertEquals(placed.sha256().length(), 64);
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(placed));
    store.append(new Promoted(f.b(), DUEL_2));

    // run の中の王者は SHA-256 を持たず、記録にも参照にも書かない
    assertNull(f.b().sha256());
    assertFalse(Files.readString(f.current(), StandardCharsets.UTF_8).contains("sha256"));
    assertTrue(Files.readString(f.history(), StandardCharsets.UTF_8).contains(placed.sha256()));
    expectThrows(IllegalStateException.class, f.b()::verify);

    // 中身が替わっても保存先は照合しない。置いた王者は、開く側が照合すると違いに気付く
    Files.writeString(f.c().checkpoint().resolve("weights.bin"), "changed", StandardCharsets.UTF_8);
    assertTrue(store.append(new Promoted(f.c(), DUEL_3)));
    assertEquals(ChampionStore.open(f.store()).past(), List.of(placed, f.b()));
    placed.verify();
    Files.writeString(outside.resolve("weights.bin"), "changed", StandardCharsets.UTF_8);
    expectThrows(IOException.class, placed::verify);

    // 書き方の誤った SHA-256 と、相対パスのチェックポイントは拒む
    String sha = placed.sha256();
    Path checkpoint = f.b().checkpoint();
    expectThrows(IllegalArgumentException.class, () -> new Champion("v00002", checkpoint, 2, sha.toUpperCase()));
    expectThrows(IllegalArgumentException.class, () -> new Champion("v00002", checkpoint, 2, sha.substring(1)));
    expectThrows(IllegalArgumentException.class, () -> new Champion("v00002", Path.of("v2"), 2, sha));
  }

  @Test
  public void applyingTheSameDuelTwiceIsTheSameAsOnce() throws Exception {
    Fixture f = Fixture.create("champion-twice");
    List<StoreEvent> events =
        List.of(
            new Initial(f.a()),
            new Promoted(f.b(), DUEL_2),
            new NotConfirmed(f.c(), DUEL_3, "v00003-outside"),
            new Paused("v00004-champion"));
    ChampionStore store = ChampionStore.open(f.store());
    for (StoreEvent event : events) {
      assertTrue(store.append(event));
      byte[] history = Files.readAllBytes(f.history());
      byte[] current = Files.readAllBytes(f.current());
      assertFalse(store.append(event));
      // 再起動した後にもう一度適用しても同じ
      assertFalse(ChampionStore.open(f.store()).append(event));
      assertEquals(Files.readAllBytes(f.history()), history);
      assertEquals(Files.readAllBytes(f.current()), current);
    }
    assertEquals(store.history().stream().map(Entry::event).toList(), events);
    assertEquals(store.current().orElseThrow(), f.b());
    assertEquals(store.past(), List.of(f.a()));

    // 同じ決闘の違うイベントは拒み、何も書かない
    byte[] history = Files.readAllBytes(f.history());
    byte[] current = Files.readAllBytes(f.current());
    expectThrows(IllegalStateException.class, () -> store.append(new Paused(DUEL_2)));
    expectThrows(IllegalStateException.class, () -> store.append(new Promoted(f.c(), DUEL_3)));
    expectThrows(IllegalStateException.class, () -> store.append(new Initial(f.c())));
    assertEquals(Files.readAllBytes(f.history()), history);
    assertEquals(Files.readAllBytes(f.current()), current);
  }

  @Test
  public void eventsThatDoNotFollowTheStoreAreRejected() throws Exception {
    Fixture f = Fixture.create("champion-order");
    ChampionStore store = ChampionStore.open(f.store());
    assertTrue(store.current().isEmpty());
    assertEquals(store.past(), List.of());
    expectThrows(IllegalStateException.class, () -> store.append(new Promoted(f.b(), DUEL_2)));
    expectThrows(IllegalStateException.class, () -> store.append(new Paused(DUEL_2)));
    assertFalse(Files.exists(f.store()));

    store.append(new Initial(f.a()));
    expectThrows(IllegalStateException.class, () -> store.append(new Promoted(f.a(), DUEL_2)));
    assertEquals(store.history().size(), 1);
    assertTrue(store.append(new Promoted(f.b(), DUEL_2)));
    assertTrue(store.append(new Promoted(f.c(), DUEL_3)));
    assertEquals(store.past(), List.of(f.a(), f.b()));
  }

  @Test
  public void pastReturnsTheNewestChampionsUpToTheLimit() throws Exception {
    Fixture f = Fixture.create("champion-newest");
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    store.append(new Promoted(f.b(), DUEL_2));
    store.append(new Promoted(f.c(), DUEL_3));
    assertEquals(store.past(5), List.of(f.a(), f.b()));
    assertEquals(store.past(1), List.of(f.b()));
    assertEquals(store.past(0), List.of());
    expectThrows(IllegalArgumentException.class, () -> store.past(-1));
  }

  /** 確かめの卓のジョブは、計算の型・打ち手のバージョンごとのディレクトリに置くので、どちらかが違えば値を混ぜない。 */
  @Test
  public void confirmTablesAreKeptApartByPrecisionAndVersion() throws Exception {
    Fixture f = Fixture.create("champion-confirms");
    ChampionStore store = ChampionStore.open(f.store());
    Path measured = store.confirms(Precision.FLOAT32, f.b());
    assertEquals(
        measured,
        f.store().toAbsolutePath().normalize().resolve(ChampionStore.CONFIRMS).resolve("FLOAT32").resolve("v00002"));
    assertEquals(store.confirms(Precision.FLOAT32, Champion.of("v00002", f.b().checkpoint(), 2)), measured);
    assertNotEquals(store.confirms(Precision.FLOAT32, f.c()), measured);
    assertNotEquals(store.confirms(Precision.BFLOAT16, f.b()), measured);
  }

  @Test
  public void aStoreWhoseRecordDoesNotHoldTogetherDoesNotOpen() throws Exception {
    Fixture f = Fixture.create("champion-broken");
    Instant at = Instant.parse("2026-11-09T00:00:00Z");
    String initial = RecordFile.json(new Entry(new Initial(f.a()), at));
    String promoted = RecordFile.json(new Entry(new Promoted(f.b(), DUEL_2), at));
    Files.createDirectories(f.store());

    Files.writeString(f.history(), promoted + "\n", StandardCharsets.UTF_8);
    expectThrows(IOException.class, () -> ChampionStore.open(f.store()));
    Files.writeString(f.history(), initial + "\n" + promoted + "\n" + initial + "\n", StandardCharsets.UTF_8);
    expectThrows(IOException.class, () -> ChampionStore.open(f.store()));
    String unknownField = initial.replace("\"champion\":{", "\"champion\":{\"extra\":1,");
    assertFalse(unknownField.equals(initial));
    Files.writeString(f.history(), unknownField + "\n", StandardCharsets.UTF_8);
    expectThrows(IOException.class, () -> ChampionStore.open(f.store()));

    // 記録が無いのに参照だけがあると開けない
    Files.writeString(f.history(), initial + "\n", StandardCharsets.UTF_8);
    ChampionStore.open(f.store());
    assertTrue(Files.exists(f.current()));
    Files.delete(f.history());
    expectThrows(IOException.class, () -> ChampionStore.open(f.store()));
  }

  @Test
  public void aHalfWrittenLastLineIsNotReadAndIsCutOnTheNextAppend() throws Exception {
    Fixture f = Fixture.create("champion-partial");
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    store.append(new Promoted(f.b(), DUEL_2));
    Files.writeString(f.history(), "{\"event\":{\"kind\":\"PAU", StandardCharsets.UTF_8, StandardOpenOption.APPEND);
    ChampionStore reopened = ChampionStore.open(f.store());
    assertEquals(reopened.history().size(), 2);
    assertTrue(reopened.append(new Promoted(f.c(), DUEL_3)));
    assertEquals(reopened.history().size(), 3);
    assertEquals(reopened.current().orElseThrow(), f.c());
    assertEquals(reopened.past(), List.of(f.a(), f.b()));
    assertFalse(Files.readString(f.history(), StandardCharsets.UTF_8).contains("PAU"));
  }

  @Test
  public void readingTheLastRecordedChampionWritesNothing() throws Exception {
    Fixture f = Fixture.create("champion-read-only");
    assertTrue(ChampionStore.lastRecorded(f.store()).isEmpty());
    assertFalse(Files.exists(f.store()));
    ChampionStore store = ChampionStore.open(f.store());
    store.append(new Initial(f.a()));
    assertEquals(ChampionStore.lastRecorded(f.store()).orElseThrow(), f.a());
    byte[] previous = Files.readAllBytes(f.current());
    store.append(new Promoted(f.b(), DUEL_2));
    byte[] next = Files.readAllBytes(f.current());
    byte[] history = Files.readAllBytes(f.history());

    // ほかのプロセスが記録に行を足し、一時ファイルを書いた後、rename の前に読む: 記録の最後の王者を返し、一時ファイルも参照もそのまま
    Files.write(f.current(), previous);
    Files.write(f.temporary(), next);
    assertEquals(ChampionStore.lastRecorded(f.store()).orElseThrow(), f.b());
    assertEquals(Files.readAllBytes(f.temporary()), next);
    assertEquals(Files.readAllBytes(f.current()), previous);
    assertEquals(Files.readAllBytes(f.history()), history);
  }
}
