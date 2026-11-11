package com.epsilon.agent.eval;

import com.epsilon.agent.checkpoint.CheckpointFiles;
import com.epsilon.agent.eval.StoreEvent.Initial;
import com.epsilon.agent.eval.StoreEvent.NotConfirmed;
import com.epsilon.agent.eval.StoreEvent.Paused;
import com.epsilon.agent.eval.StoreEvent.Promoted;
import com.epsilon.agent.eval.StoreEvent.RolledBack;
import com.epsilon.agent.run.DurableFiles;
import com.epsilon.agent.runtime.Precision;
import com.epsilon.agent.settings.Json;
import com.epsilon.agent.settings.Key;
import com.epsilon.agent.settings.Presence;
import com.epsilon.agent.settings.RecordFile;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.time.Instant;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;
import java.util.regex.Pattern;
import java.util.stream.Stream;

/**
 * 王者の保存先（run の champion ディレクトリ）。今の王者を指す参照（current.json）と、追記だけの記録（history.jsonl）を持つ。
 *
 * <p>正本は記録で、参照は記録の最後の王者を写したもの。王者を替えるイベントでは、まず記録に 1 行足して fsync する。次に参照を
 * 一時ファイル → fsync → rename → ディレクトリの fsync の順で書き換える。参照を書き換える前に落ちても、参照は前の王者のまま読める。
 * その参照は、次に開いたときか同じイベントをもう一度適用したときに、記録に合わせて書き直す。同じ決闘のイベントは 2 度適用しても
 * 1 度と同じ結果になる。
 *
 * <p>run の中の王者はバージョンで見分けるので、SHA-256 を持たない。run の外から置いた王者だけが、置いたときのチェックポイントの
 * SHA-256 を持ち、開く側がその値と照合する。書き込むのは、run の学習のロックを持つ 1 つのプロセスだけとする。
 *
 * <p>classic の campaign は、決闘に出した候補のチェックポイント（{@value #CANDIDATES}）と、王者との決闘と確かめの卓のジョブ（{@value #DUELS}・
 * {@value #CONFIRMS}）も、この下に置く。王者と過去の王者のチェックポイントは候補のディレクトリのままなので、候補は消さない。
 */
public final class ChampionStore {

  /** 今の王者を指す参照のファイル。 */
  public static final String CURRENT = "current.json";

  /** イベントの記録（1 行に 1 イベント、追記だけ）。 */
  public static final String HISTORY = "history.jsonl";

  /** 決闘に出した候補のチェックポイント（中は候補の名前ごとのディレクトリ）。 */
  public static final String CANDIDATES = "candidates";

  /** 王者の決闘のジョブ（run のジョブと同じ形）。 */
  public static final String DUELS = "duels";

  /** 確かめの卓のジョブ（中は計算の型・打ち手のバージョンごとの、run のジョブと同じ形のディレクトリ）。 */
  public static final String CONFIRMS = "confirms";

  private static final Pattern SHA256 = Pattern.compile("[0-9a-f]{64}");

  private final Path directory;

  /** 読むだけで開いた置き場か。 */
  private final boolean readOnly;

  private ChampionStore(Path directory, boolean readOnly) {
    this.directory = directory;
    this.readOnly = readOnly;
  }

  /**
   * 王者 1 人。
   *
   * @param label 人が読む名前（{@code v00112} など）
   * @param checkpoint チェックポイントのディレクトリ（絶対パス）
   * @param version バージョン（classic では macro の番号）
   * @param sha256 run の外から置いたときのチェックポイントの SHA-256（{@link CheckpointFiles#sha256}、小文字の 16 進で 64 文字）。
   *     run の中の王者には無い
   */
  public record Champion(
      String label,
      Path checkpoint,
      int version,
      @Key(doc = "run の外から置いたときのチェックポイントの SHA-256", presence = Presence.ABSENT) String sha256) {

    public Champion {
      if (label == null || label.isBlank()) {
        throw new IllegalArgumentException("A champion needs a label");
      }
      Objects.requireNonNull(checkpoint, "checkpoint");
      if (!checkpoint.isAbsolute()) {
        throw new IllegalArgumentException("The checkpoint of a champion must be an absolute path: " + checkpoint);
      }
      checkpoint = checkpoint.normalize();
      if (version < 0) {
        throw new IllegalArgumentException("Invalid version of champion " + label + ": " + version);
      }
      if (sha256 != null && !SHA256.matcher(sha256).matches()) {
        throw new IllegalArgumentException("Invalid SHA-256 of champion " + label + ": " + sha256);
      }
    }

    /** run の中のチェックポイントの王者を作る。 */
    public static Champion of(String label, Path checkpoint, int version) {
      return new Champion(label, checkpoint.toAbsolutePath().normalize(), version, null);
    }

    /** run の外から置くチェックポイントの王者を、今の中身の SHA-256 を添えて作る。 */
    public static Champion placed(String label, Path checkpoint, int version) throws IOException {
      Path absolute = checkpoint.toAbsolutePath().normalize();
      return new Champion(label, absolute, version, CheckpointFiles.sha256(absolute));
    }

    /**
     * run の外から置いた王者のチェックポイントの中身が、置いたときの SHA-256 のままかを確かめる。
     *
     * @throws IllegalStateException run の中の王者（SHA-256 が無い）のとき
     * @throws IOException 読めないとき、または SHA-256 が違うとき
     */
    public void verify() throws IOException {
      if (sha256 == null) {
        throw new IllegalStateException("Champion " + label + " is a version of the run and has no SHA-256 to verify");
      }
      String actual = CheckpointFiles.sha256(checkpoint);
      if (!actual.equals(sha256)) {
        throw new IOException(
            "The checkpoint of champion " + label + " at " + checkpoint + " has SHA-256 " + actual
                + ", but the store recorded " + sha256);
      }
    }
  }

  /**
   * 記録の 1 行。
   *
   * @param at 書き込んだ時刻
   */
  public record Entry(StoreEvent event, Instant at) {

    public Entry {
      Objects.requireNonNull(event, "event");
      Objects.requireNonNull(at, "at");
    }
  }

  /**
   * 参照の中身。
   *
   * @param duel 王者を決めた決闘の名前。初めの王者には無い
   * @param since 王者になった時刻（記録の行の時刻）
   */
  record Reference(
      Champion champion,
      @Key(doc = "王者を決めた決闘の名前", presence = Presence.ABSENT) String duel,
      Instant since) {}

  /**
   * 保存先を開く。参照が記録より遅れていれば（参照を書き換える前に落ちたとき）、記録に合わせて書き直す。ディレクトリが無ければ
   * 空の保存先として開き、ディレクトリは最初のイベントを書くときに作る。
   *
   * @param directory 保存先のディレクトリ（run の champion）
   * @throws IOException 記録が読めないとき、または記録の中身が矛盾するとき（初めの王者が先頭にない、王者がいないのに替えている、
   *     参照だけがある）
   */
  public static ChampionStore open(Path directory) throws IOException {
    ChampionStore store = new ChampionStore(directory.toAbsolutePath().normalize(), false);
    store.removeTemporaries();
    store.syncReference(store.reference(store.history()));
    return store;
  }

  /**
   * 記録の最後の王者を、保存先に何も書かずに読む（{@link #view} で開いた保存先の今の王者）。{@link #open} と違い、一時ファイルを消さず、
   * 参照も書き直さないので、学習のロックを持たないプロセスが、学習の JVM の書き込みと同時に呼べる。
   *
   * @param directory 保存先のディレクトリ（run の champion）
   * @return 記録に王者がいなければ空
   * @throws IOException 記録が読めないとき、または筋が通らないとき
   */
  public static Optional<Champion> lastRecorded(Path directory) throws IOException {
    return view(directory).current();
  }

  /**
   * 保存先を読むだけで開く。一時ファイルを消さず、参照も書かないので、学習のロックを持たないプロセスが、ほかのプロセスが書いている間でも
   * 使える。今の王者は、参照ではなく記録の最後の王者から求める（参照は記録より遅れていることがある）。イベントは適用できない。
   *
   * @param directory 保存先のディレクトリ（run の champion）
   */
  public static ChampionStore view(Path directory) {
    return new ChampionStore(directory.toAbsolutePath().normalize(), true);
  }

  /**
   * 今の王者を返す。参照を読む（読むだけで開いた保存先では、記録の最後の王者を求める）。
   *
   * @return 今の王者。保存先が空なら空
   * @throws IOException 参照か記録が読めないとき
   */
  public Optional<Champion> current() throws IOException {
    if (readOnly) {
      return reference(history()).map(Reference::champion);
    }
    Path file = directory.resolve(CURRENT);
    if (!Files.exists(file)) {
      return Optional.empty();
    }
    return Optional.of(read(file).champion());
  }

  /**
   * 過去の王者を、王者になった順に返す（今の王者は除く）。
   *
   * @throws IOException 記録が読めないとき
   */
  public List<Champion> past() throws IOException {
    return past(Integer.MAX_VALUE);
  }

  /**
   * 過去の王者のうち、新しいものから数えて上限までの王者（王者になった順、今の王者を除く）。
   *
   * @param limit 返す王者の数の上限
   * @throws IOException 記録が読めないとき
   */
  public List<Champion> past(int limit) throws IOException {
    if (limit < 0) {
      throw new IllegalArgumentException("Invalid limit of past champions: " + limit);
    }
    List<Champion> champions = new ArrayList<>();
    for (Entry entry : history()) {
      champion(entry.event()).ifPresent(champions::add);
    }
    if (!champions.isEmpty()) {
      champions.remove(champions.size() - 1);
    }
    return List.copyOf(champions.subList(Math.max(0, champions.size() - limit), champions.size()));
  }

  /**
   * @param label 候補の名前（{@code v00120}）
   * @return 候補のチェックポイントのディレクトリ
   */
  public Path candidate(String label) {
    return directory.resolve(CANDIDATES).resolve(label);
  }

  /** @return 王者の決闘のジョブのディレクトリ */
  public Path duels() {
    return directory.resolve(DUELS);
  }

  /**
   * 確かめの卓で打ち手を測ったジョブのディレクトリ。打ち手の計算の型が替わると同じ牌山でも半荘が変わりうるので、確かめの卓の値は
   * 同じ計算の型で測ったものどうしだけを対にする。打ち手はバージョンで分ける（保存先に置く打ち手は、どれもバージョンが違う）。
   *
   * @param precision 測る打ち手の計算の型
   * @param player 測る打ち手
   * @return 計算の型、打ち手のバージョン（{@code v00012}）で分けたディレクトリ
   */
  public Path confirms(Precision precision, Champion player) {
    return directory
        .resolve(CONFIRMS)
        .resolve(precision.name())
        .resolve(String.format(Locale.ROOT, "v%05d", player.version()));
  }

  /** 記録の全ての行（書き込んだ順）。 */
  public List<Entry> history() throws IOException {
    Path file = directory.resolve(HISTORY);
    List<String> lines = DurableFiles.lines(file);
    List<Entry> entries = new ArrayList<>(lines.size());
    for (int index = 0; index < lines.size(); index++) {
      String where = file + ":" + (index + 1);
      Entry entry;
      try {
        entry = RecordFile.read(Entry.class, Json.read(lines.get(index), where)).value();
      } catch (IllegalArgumentException broken) {
        throw new IOException("Cannot read " + where + ": " + broken.getMessage(), broken);
      }
      boolean initial = entry.event() instanceof Initial;
      if (index == 0 && !initial) {
        throw new IOException(where + " records a duel before the store has a champion");
      }
      if (index > 0 && initial) {
        throw new IOException(where + " holds an initial champion after line 1");
      }
      entries.add(entry);
    }
    return entries;
  }

  /**
   * イベントを適用する。同じ決闘のイベントが既に記録にあれば書き込まず、参照が記録より遅れていれば書き直すだけにする（落ちた後に
   * もう一度適用しても 1 度と同じ結果になる）。
   *
   * @return 記録に行を足したら true
   * @throws IllegalStateException 同じ決闘で違うイベントが記録してあるとき、空でない保存先に初めの王者を置こうとしたとき、空の
   *     保存先に決闘のイベントを適用しようとしたとき、今の王者を昇格させようとしたとき、読むだけで開いた保存先に適用しようとしたとき
   * @throws IOException 記録が読めないとき
   */
  public boolean append(StoreEvent event) throws IOException {
    if (readOnly) {
      throw new IllegalStateException("The champion store " + directory + " is open for reading only: " + event);
    }
    List<Entry> history = history();
    Optional<Entry> recorded = recorded(history, event);
    if (recorded.isPresent()) {
      if (!recorded.get().event().equals(event)) {
        throw new IllegalStateException(
            "The store already recorded " + recorded.get().event() + ", not " + event);
      }
      syncReference(reference(history));
      return false;
    }
    Optional<Reference> before = reference(history);
    switch (event) {
      case StoreEvent duel when history.isEmpty() && !(duel instanceof Initial) ->
          throw new IllegalStateException("The store has no champion yet: " + duel);
      case Promoted promoted when promoted.champion().equals(before.orElseThrow().champion()) ->
          throw new IllegalStateException("The candidate is already the champion: " + promoted);
      default -> {}
    }
    Optional<Champion> crowned = champion(event);
    Entry entry = new Entry(event, Instant.now());
    DurableFiles.append(directory.resolve(HISTORY), RecordFile.json(entry));
    if (crowned.isPresent()) {
      List<Entry> after = new ArrayList<>(history);
      after.add(entry);
      syncReference(reference(after));
    }
    return true;
  }

  /** 同じイベントを記録した行。初めの王者なら先頭の行、ほかは決闘の名前が同じ行。 */
  private static Optional<Entry> recorded(List<Entry> history, StoreEvent event) {
    Optional<String> duel = duel(event);
    if (duel.isEmpty()) {
      return history.stream().findFirst().filter(entry -> entry.event() instanceof Initial);
    }
    return history.stream().filter(entry -> duel.equals(duel(entry.event()))).findFirst();
  }

  private static Optional<String> duel(StoreEvent event) {
    return switch (event) {
      case Initial initial -> Optional.empty();
      case Promoted promoted -> Optional.of(promoted.duel());
      case NotConfirmed notConfirmed -> Optional.of(notConfirmed.duel());
      case Paused paused -> Optional.of(paused.duel());
      case RolledBack rolledBack -> Optional.of(rolledBack.duel());
    };
  }

  /** イベントで王者になった打ち手。王者を替えないイベントなら空。 */
  private static Optional<Champion> champion(StoreEvent event) {
    return switch (event) {
      case Initial initial -> Optional.of(initial.champion());
      case Promoted promoted -> Optional.of(promoted.champion());
      case NotConfirmed notConfirmed -> Optional.empty();
      case Paused paused -> Optional.empty();
      case RolledBack rolledBack -> Optional.empty();
    };
  }

  /** 記録の最後の王者を指す参照。 */
  private static Optional<Reference> reference(List<Entry> history) {
    Optional<Reference> reference = Optional.empty();
    for (Entry entry : history) {
      Optional<Champion> champion = champion(entry.event());
      if (champion.isPresent()) {
        reference = Optional.of(new Reference(champion.get(), duel(entry.event()).orElse(null), entry.at()));
      }
    }
    return reference;
  }

  /** 参照を記録に合わせる。既に同じなら書かない。 */
  private void syncReference(Optional<Reference> expected) throws IOException {
    Path file = directory.resolve(CURRENT);
    boolean exists = Files.exists(file);
    if (expected.isEmpty()) {
      if (exists) {
        throw new IOException(file + " names a champion, but " + directory.resolve(HISTORY) + " records none");
      }
      return;
    }
    if (exists && read(file).equals(expected.get())) {
      return;
    }
    DurableFiles.write(file, expected.get());
  }

  private static Reference read(Path file) throws IOException {
    try {
      return RecordFile.read(Reference.class, file).value();
    } catch (IllegalArgumentException broken) {
      throw new IOException("Cannot read " + file + ": " + broken.getMessage(), broken);
    }
  }

  /** 参照を書き換える途中で落ちたときに残った一時ファイルを消す。 */
  private void removeTemporaries() throws IOException {
    if (!Files.isDirectory(directory)) {
      return;
    }
    String prefix = "." + CURRENT + ".";
    List<Path> leftovers;
    try (Stream<Path> listed = Files.list(directory)) {
      leftovers =
          listed
              .filter(path -> {
                String name = path.getFileName().toString();
                return name.startsWith(prefix) && name.endsWith(".tmp");
              })
              .toList();
    }
    for (Path leftover : leftovers) {
      Files.delete(leftover);
    }
    if (!leftovers.isEmpty()) {
      DurableFiles.syncDirectory(directory);
    }
  }
}
