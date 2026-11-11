package com.epsilon.mahjong.engine;

import com.epsilon.mahjong.rules.Variant;
import com.epsilon.mahjong.score.WinContext;
import com.epsilon.mahjong.tile.TileSet;
import com.epsilon.mahjong.tile.Tiles;
import com.epsilon.mahjong.util.SeedMixer;
import java.util.Arrays;
import java.util.Objects;
import java.util.Random;

/**
 * 局の牌山。位置ごとに物理 id を持つ。エンジンだけが持ち、どのビューからも見えない。以下、L は牌の数、S は席の数。
 *
 * <p>位置: 配牌は [0, 13S)（親から 4 枚ずつ 3 巡配り、続けて 1 枚ずつ配る）、山は [13S, L − 14)、王牌は [L − 14, L)。表示牌 i は
 * L − 6 − 2i、裏ドラ表示牌 i は L − 5 − 2i に置く。嶺上牌か抜きの補充を 1 枚引くたびに、山の終わりが 1 つ手前へ来る。席別ツモ列を
 * 使わない局では、k 回目（1 始まり）の嶺上は k ≤ 4 なら L − k、k ≥ 5 なら L − 10 − k を読む。k ≥ 5 の位置は (k − 4) 回目の嶺上で
 * 山から外れた位置なので、山のツモと嶺上が同じ位置を読むことは無い。
 *
 * <p>牌山は局ごとに作る。牌の集合を物理 id の昇順に並べ、種 {@code SeedMixer.indexed(seed, SALT, epoch)} の Random で後ろから
 * Fisher–Yates のシャッフルを掛ける。epoch は局ごとに 1 進む。
 *
 * <p>席別ツモ列: 配牌の直後に、位置 13S + k の牌を席 (親 + k) mod S の列に割り当てる（S は席の数）。ツモでは、自分の列の次の牌のうち、
 * まだ山の中（読み位置から山の終わりの手前まで）にあるものを読み位置へ入れ替えてから読む。列が尽きたら位置の順に読む。嶺上牌と補充では、
 * 同じように選んだ牌を山の最後の位置へ入れ替えてから読み（列が尽きていればその位置の牌を読む）、山を 1 枚縮める。王牌の嶺上の位置は
 * 読まないので、鳴いても槓しても、列が残る間は各席が引く牌の並びは変わらない。割り当ては位置だけで決まり牌を見ないので、ツモの分布は
 * 位置の順に読むときと同じになる。
 *
 * <p>並べ直し: 判断の時点で、まだ読んでいない位置の牌を一様に並べ替える。種から物理 id ごとに乱数のキーを作り、まだ読んでいない牌を
 * キーの順に、まだ読んでいない位置へ位置の番号の順に置く。席別ツモ列のまだ使っていない項目は、元の牌があった位置に来た牌へ書き換える。
 * そのため列の並びは変わらず、中身だけが入れ替わる。同じ種なら、後の判断の時点での並べ直しは、前の並べ直しのキーの順から既に読んだ牌を
 * 除いた順になる。
 *
 * <p>牌譜の再生では記録の牌を同じ位置に置くので、位置の式は static メソッドで公開する。牌山そのものは公開しない。
 */
public final class Wall {

  /** 局ごとのシャッフルの種に混ぜるソルト。 */
  static final long SALT = 0xC6BC279692B5CC83L;

  /** 並べ直しのキーの種に混ぜるソルト。 */
  private static final long RESHUFFLE_SALT = 0x5B4E1D3A9F27C861L;

  /** 並べ直しのキーの下位ビットに物理 id を入れるためのマスク（物理 id は 136 未満）。 */
  private static final long ID_BITS = 0xFF;

  /** 牌譜の局で、牌が分からない位置。読むと IllegalStateException を投げる。 */
  public static final int UNKNOWN = -1;

  /** 配牌で 1 度に配る枚数。親から 4 枚ずつ 3 巡配り、続けて 1 枚ずつ配る。 */
  private static final int PACKET = 4;

  private static final int PACKETS = (Variant.HAND - 1) / PACKET;

  /** 王牌の末尾にある嶺上牌の数。その手前に、表示牌と裏ドラ表示牌の組が、表示牌の上限（{@link WinContext#MAX_INDICATORS}）の数だけ並ぶ。 */
  private static final int DEAD_RINSHAN = Variant.MAX_KANS;

  private final Variant variant;
  private final TileSet set;
  private final int seats;
  private final int size;

  /** 山の始まり（配牌の枚数）。 */
  private final int liveStart;

  /** 王牌の始まり。 */
  private final int deadStart;

  /** 1 席の列に入る牌の上限（山の枚数を席の数で割って切り上げた数）。 */
  private final int streamCapacity;

  /** 位置 → 物理 id。 */
  private final int[] tiles;

  /** [席 × 1 席の列の上限 ＋ i] に、その席の列の i 番目の物理 id を置く。 */
  private final int[] stream;

  private final int[] streamSize;
  private final int[] streamCursor;

  /** 物理 id → 位置。値が正しいのは席別ツモ列を使う局だけ。牌譜の牌山を読むときは、同じ物理 id が重なっていないかを調べるのに使う。 */
  private final int[] positionOf = new int[Tiles.PHYSICAL_IDS];

  /** 並べ直しの作業領域: まだ読んでいない位置。 */
  private final int[] unread;

  /** 並べ直しの作業領域: 上位ビットが乱数のキー、下位ビットが物理 id。 */
  private final long[] keys;

  /** 並べ直しの作業領域: 元の物理 id → その位置に新しく来た物理 id。まだ読んでいない牌でなければ {@link #UNKNOWN}。 */
  private final int[] image = new int[Tiles.PHYSICAL_IDS];

  /** 牌山を作る直前に毎回種を設定し直す。そのため、ここで与える種 0 も、コピーで乱数の状態を写さないことも、結果に影響しない。 */
  private final Random random = new Random(0);

  private long seed;

  /** 次の局の牌山の epoch。 */
  private int epoch;

  /** 半荘の各局のツモの順。 */
  private DrawOrder drawOrder = DrawOrder.WALL_ORDER;

  /** いまの局が席別ツモ列を使うか。使うのは {@link #shuffle} が列を作った局だけで、牌譜の局では使わない。 */
  private boolean streamed;

  /** 次に山から読む位置。 */
  private int next;

  /** 嶺上牌と抜きの補充でツモった枚数。 */
  private int rinshan;

  Wall(Variant variant) {
    this.variant = variant;
    set = variant.tiles();
    seats = variant.seats();
    size = set.size();
    liveStart = Variant.HAND * seats;
    deadStart = size - Variant.DEAD_WALL;
    streamCapacity = variant.maxDrawsPerSeat();
    tiles = new int[size];
    stream = new int[seats * streamCapacity];
    streamSize = new int[seats];
    streamCursor = new int[seats];
    unread = new int[size];
    keys = new long[size];
  }

  /**
   * 半荘を始める。
   *
   * @param seed 牌山の種
   * @param epoch 次の局の牌山の epoch
   * @param drawOrder 各局のツモの順
   */
  void start(long seed, int epoch, DrawOrder drawOrder) {
    this.seed = seed;
    this.epoch = epoch;
    this.drawOrder = Objects.requireNonNull(drawOrder, "drawOrder");
  }

  /** 次の局の牌山を作り、epoch を 1 進める。席別ツモ列を使う半荘なら、親から順に列を作る。 */
  void shuffle(int dealer) {
    Objects.checkIndex(dealer, seats);
    for (int position = 0; position < size; position++) {
      tiles[position] = set.physicalId(position);
    }
    random.setSeed(SeedMixer.indexed(seed, SALT, epoch));
    epoch++;
    for (int position = size - 1; position > 0; position--) {
      int other = random.nextInt(position + 1);
      int tile = tiles[position];
      tiles[position] = tiles[other];
      tiles[other] = tile;
    }
    begin();
    if (drawOrder == DrawOrder.SEAT_STREAMS) {
      buildStreams(dealer);
    }
  }

  /**
   * 牌譜の局の牌山にする。席別ツモ列は使わない。
   *
   * @param recorded 位置ごとの物理 id。牌が分からない位置は {@link #UNKNOWN}
   */
  void load(int[] recorded) {
    if (recorded.length != size) {
      throw new IllegalArgumentException(variant + " needs " + size + " wall positions, got " + recorded.length);
    }
    begin();
    Arrays.fill(positionOf, UNKNOWN);
    for (int position = 0; position < size; position++) {
      int tile = recorded[position];
      if (tile == UNKNOWN) {
        continue;
      }
      if (!set.contains(tile) || positionOf[tile] != UNKNOWN) {
        throw new IllegalArgumentException(
            "Wall position " + position + " repeats or does not belong to " + set + ": " + tile);
      }
      positionOf[tile] = position;
    }
    System.arraycopy(recorded, 0, tiles, 0, size);
  }

  /**
   * 配牌の位置。
   *
   * @param variant ルール
   * @param offset 親から何席目か
   * @param index 配牌の何枚目か（0 始まり）
   */
  public static int dealtPosition(Variant variant, int offset, int index) {
    int seats = variant.seats();
    Objects.checkIndex(offset, seats);
    int packet = Objects.checkIndex(index, Variant.HAND) / PACKET;
    return packet < PACKETS ? (packet * seats + offset) * PACKET + index % PACKET : PACKETS * PACKET * seats + offset;
  }

  /**
   * 席別ツモ列を使わない局で、山からツモる位置。
   *
   * @param variant ルール
   * @param k 山の何枚目のツモか（0 始まり）
   */
  public static int livePosition(Variant variant, int k) {
    return Variant.HAND * variant.seats() + Objects.checkIndex(k, variant.liveWall());
  }

  /**
   * 席別ツモ列を使わない局で、嶺上牌か抜きの補充を引く位置。
   *
   * @param variant ルール
   * @param k 何回目の補充か（1 始まり。嶺上から引ける枚数まで）
   */
  public static int rinshanPosition(Variant variant, int k) {
    if (k < 1 || k > variant.rinshanTiles()) {
      throw new IllegalArgumentException(
          variant + " has replacement draws 1 to " + variant.rinshanTiles() + ", not " + k);
    }
    int size = variant.tiles().size();
    return k <= DEAD_RINSHAN ? size - k : size - Variant.DEAD_WALL - (k - DEAD_RINSHAN);
  }

  /**
   * 表示牌の位置 L − 6 − 2i（i は 0 始まりの表示牌の番号）。嶺上牌の手前に、最初の表示牌から位置の小さいほうへ、（表示牌, 裏ドラ表示牌）
   * の組が並ぶ。
   *
   * @param variant ルール
   * @param index 表示牌の番号（0 始まり）
   */
  public static int indicatorPosition(Variant variant, int index) {
    return variant.tiles().size() - DEAD_RINSHAN - 2 * (Objects.checkIndex(index, WinContext.MAX_INDICATORS) + 1);
  }

  /** 表示牌に対応する裏ドラ表示牌の位置（表示牌の次の位置）。 */
  public static int uraPosition(Variant variant, int index) {
    return indicatorPosition(variant, index) + 1;
  }

  /** 配牌の物理 id。引数は {@link #dealtPosition} と同じ。 */
  int dealt(int offset, int index) {
    return tile(dealtPosition(variant, offset, index));
  }

  /** その席がツモる牌の物理 id を返す。山が尽きていれば IllegalStateException を投げる。 */
  int draw(int seat) {
    int end = liveEnd();
    if (next >= end) {
      throw new IllegalStateException("The live wall is exhausted");
    }
    moveStreamTile(seat, end, next);
    int tile = tile(next);
    next++;
    return tile;
  }

  /**
   * その席が嶺上牌か抜きの補充としてツモる牌を返し、山を 1 枚縮める。席別ツモ列の局ではその席の列の次の牌を、そうでなければ王牌の
   * 嶺上の位置の牌を読む。山が尽きているか、補充の牌が残っていなければ IllegalStateException を投げる。
   */
  int drawRinshan(int seat) {
    Objects.checkIndex(seat, seats);
    if (rinshan == variant.rinshanTiles()) {
      throw new IllegalStateException("All " + rinshan + " replacement tiles are drawn");
    }
    int end = liveEnd();
    if (next >= end) {
      throw new IllegalStateException("The live wall is exhausted");
    }
    int position;
    if (streamed) {
      position = end - 1;
      moveStreamTile(seat, end, position);
    } else {
      position = rinshanPosition(variant, rinshan + 1);
    }
    int tile = tile(position);
    rinshan++;
    return tile;
  }

  /**
   * まだ読んでいない位置の牌を、クラスの説明のとおり一様に並べ直す。牌譜の牌山では呼ばない（呼ぶ側が拒む）。メモリは確保しない。
   *
   * @param seed 並べ直しの種
   * @param chance 局で起きた偶然のイベントの記録。誰にも見えていない牌を、まだ読んでいない位置の牌として扱う
   */
  void reshuffleUnseen(long seed, ChanceLog chance) {
    int count = 0;
    for (int position = 0; position < size; position++) {
      int tile = tiles[position];
      if (chance.hidden(tile)) {
        unread[count] = position;
        keys[count] = SeedMixer.indexed(seed, RESHUFFLE_SALT, tile) & ~ID_BITS | tile;
        count++;
      }
    }
    Arrays.sort(keys, 0, count);
    Arrays.fill(image, UNKNOWN);
    for (int i = 0; i < count; i++) {
      int position = unread[i];
      int tile = (int) (keys[i] & ID_BITS);
      image[tiles[position]] = tile;
      tiles[position] = tile;
      positionOf[tile] = position;
    }
    if (!streamed) {
      return;
    }
    for (int seat = 0; seat < seats; seat++) {
      int base = seat * streamCapacity;
      for (int index = streamCursor[seat]; index < streamSize[seat]; index++) {
        int moved = image[stream[base + index]];
        if (moved != UNKNOWN) {
          stream[base + index] = moved;
        }
      }
    }
  }

  /** 表示牌の物理 id。引数は 0 始まりの表示牌の番号。 */
  int indicator(int index) {
    return tile(indicatorPosition(variant, index));
  }

  /** 表示牌に対応する裏ドラ表示牌の物理 id。 */
  int ura(int index) {
    return tile(uraPosition(variant, index));
  }

  /** 王牌を除いた山の残りの枚数。 */
  int liveLeft() {
    return liveEnd() - next;
  }

  /** 次の局の牌山の epoch。 */
  int epoch() {
    return epoch;
  }

  /** その位置の牌の物理 id。牌譜の局で牌が分からない位置なら IllegalStateException を投げる。 */
  int tile(int position) {
    int tile = tiles[Objects.checkIndex(position, size)];
    if (tile == UNKNOWN) {
      throw new IllegalStateException("Wall position " + position + " is not known");
    }
    return tile;
  }

  void copyFrom(Wall other) {
    if (other.variant != variant) {
      throw new IllegalArgumentException("Variant mismatch: expected " + variant + " but found " + other.variant);
    }
    System.arraycopy(other.tiles, 0, tiles, 0, tiles.length);
    System.arraycopy(other.stream, 0, stream, 0, stream.length);
    System.arraycopy(other.streamSize, 0, streamSize, 0, streamSize.length);
    System.arraycopy(other.streamCursor, 0, streamCursor, 0, streamCursor.length);
    System.arraycopy(other.positionOf, 0, positionOf, 0, positionOf.length);
    seed = other.seed;
    epoch = other.epoch;
    drawOrder = other.drawOrder;
    streamed = other.streamed;
    next = other.next;
    rinshan = other.rinshan;
  }

  /** 読み位置を配牌の直後に戻し、席別ツモ列を空にする。 */
  private void begin() {
    next = liveStart;
    rinshan = 0;
    streamed = false;
    Arrays.fill(streamSize, 0);
    Arrays.fill(streamCursor, 0);
  }

  private void buildStreams(int dealer) {
    for (int position = liveStart; position < deadStart; position++) {
      int seat = (dealer + position - liveStart) % seats;
      stream[seat * streamCapacity + streamSize[seat]] = tiles[position];
      streamSize[seat]++;
    }
    for (int position = 0; position < size; position++) {
      positionOf[tiles[position]] = position;
    }
    streamed = true;
  }

  /** 山の終わり。嶺上を引いた枚数だけ、王牌の始まりより手前に来る。 */
  private int liveEnd() {
    return deadStart - rinshan;
  }

  private boolean live(int position, int end) {
    return position >= next && position < end;
  }

  /** その席の列の次の牌のうち、まだ山の中（読み位置から山の終わりの手前まで）にあるものを、指定の位置へ入れ替える。列が尽きていれば何もしない。 */
  private void moveStreamTile(int seat, int end, int position) {
    int base = Objects.checkIndex(seat, seats) * streamCapacity;
    int cursor = streamCursor[seat];
    while (cursor < streamSize[seat] && !live(positionOf[stream[base + cursor]], end)) {
      cursor++;
    }
    streamCursor[seat] = cursor;
    if (cursor < streamSize[seat]) {
      moveTo(stream[base + cursor], position);
    }
  }

  private void moveTo(int tile, int position) {
    int from = positionOf[tile];
    if (from == position) {
      return;
    }
    int displaced = tiles[position];
    tiles[position] = tile;
    tiles[from] = displaced;
    positionOf[tile] = position;
    positionOf[displaced] = from;
  }
}
