package com.epsilon.reviewer.engine;

import com.epsilon.mahjong.io.mjai.MjaiCodec;
import com.epsilon.mahjong.record.SceneGames;
import com.epsilon.mahjong.rules.Variant;
import com.epsilon.reviewer.TestFiles;
import com.epsilon.reviewer.engine.load.LoadedRecord;
import com.epsilon.reviewer.engine.load.RecordLoader;
import com.epsilon.reviewer.engine.load.TestRecords;
import java.nio.charset.StandardCharsets;
import java.util.LinkedHashMap;
import java.util.Map;

/**
 * 解析ともしものテストで使う牌譜。エンジンで打った半荘（{@link TestFiles#played}）を、取り込みと同じ道で読み込む。同じ半荘を mjai に
 * 書いて読み直すと、物理牌は mjai を読むときの割り当て方で付け直される。三人打ちには、自己対局では起きにくい槍槓と抜いた北へのロンの
 * 局を、牌山を手で並べて足す。
 */
final class PlayedRecords {
  /** 使う半荘の数（乱数の種は 0 から）。 */
  private static final int GAMES = 4;

  /** 三人打ちで使う半荘の数（乱数の種は 0 から）。 */
  private static final int THREE_PLAYER_GAMES = 8;

  private PlayedRecords() {}

  /**
   * 乱数の種で打った半荘を読み込む。
   *
   * @param seed 半荘を打つ乱数の種
   */
  static LoadedRecord read(Variant variant, long seed) {
    return RecordLoader.load(TestFiles.played(variant, seed));
  }

  /** 四人打ちの半荘を、名前（played-種）から順に引ける表にして返す。 */
  static Map<String, LoadedRecord> fourPlayer() {
    return games(Variant.TENHOU_4P, GAMES);
  }

  /** 三人打ちの半荘を、名前（played-種）から順に引ける表にして返す。 */
  static Map<String, LoadedRecord> threePlayer() {
    return games(Variant.TENHOU_3P, THREE_PLAYER_GAMES);
  }

  /** 三人打ちの、牌山を手で並べた槍槓の局と抜いた北へのロンの局。読み込みの確かめの再生はしない。 */
  static Map<String, LoadedRecord> threePlayerScenes() {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    records.put("chankan", TestRecords.unchecked(SceneGames.chankan()));
    records.put("ron-on-nuki", TestRecords.unchecked(SceneGames.ronOnNuki(35_000, 35_000, 35_000)));
    return records;
  }

  /** 同じ半荘を mjai に書き出して読み直した牌譜。 */
  static LoadedRecord mjai(LoadedRecord record) {
    return RecordLoader.load(MjaiCodec.write(record.game()).getBytes(StandardCharsets.UTF_8));
  }

  private static Map<String, LoadedRecord> games(Variant variant, int count) {
    Map<String, LoadedRecord> records = new LinkedHashMap<>();
    for (long seed = 0; seed < count; seed++) records.put("played-" + seed, read(variant, seed));
    return records;
  }
}
