package com.epsilon.prism.input;

import com.epsilon.prism.input.InputSchema.TileField;
import com.epsilon.prism.kernel.CollectionCapabilities;
import com.epsilon.prism.kernel.CollectionPlanOptions.SegmentLayout;
import com.epsilon.prism.kernel.CollectionPlanOptions.SegmentSpan;
import java.util.ArrayList;
import java.util.List;
import java.util.function.ToIntFunction;

/** 遷移と牌の組に与える区分ごとのコードと、対応する重み表の配置。 */
public final class CandidateCodes {
  public static final int COUNT_WIDTH = CollectionCapabilities.COUNT_WIDTH;

  /** 候補区分の桁と、その形式で使う基数。 */
  public enum Digit {
    UKEIRE(2),
    OWN_RIVER(2),
    DORA(format -> format.tile().spec(TileField.DORA).vocabulary()),
    WAIT(2),
    DISCARDED(2),
    SAFETY(TileSafety.Safety.values().length + 1),
    RIICHI(2);

    private final ToIntFunction<InputFormat> radix;

    Digit(int radix) {
      this(format -> radix);
    }

    Digit(ToIntFunction<InputFormat> radix) {
      this.radix = radix;
    }
  }

  /** 区分に使う桁を上位から順に宣言する。席の区分は相対席ごとに繰り返す。 */
  public enum Segment {
    RIVER("ukeireRiver", Digit.UKEIRE, Digit.OWN_RIVER),
    DORA("ukeireDora", Digit.UKEIRE, Digit.DORA),
    SEAT("waitDiscardSafetyRiichi", Digit.WAIT, Digit.DISCARDED, Digit.SAFETY, Digit.RIICHI);

    final String parameter;
    final Digit[] digits;

    Segment(String parameter, Digit... digits) {
      this.parameter = parameter;
      this.digits = digits;
    }
  }

  /** 区分の書き先と重み表の対応。席に依らない区分では seat は −1。 */
  public record Slot(
      Segment segment,
      int seat,
      String parameter,
      int column,
      int firstRow,
      CodeSpace<Digit> space) {
    public int rows() {
      return space.size();
    }
  }

  private final List<Slot> slots;
  private final int codeWidth;
  private final int tableRows;

  CandidateCodes(InputFormat format) {
    var entries = new ArrayList<Slot>();
    int column = format.fixedTokens();
    int rows = 0;
    if (format.has(InputFeature.CANDIDATE_SEGMENTS)) {
      for (Segment segment : Segment.values()) {
        var space =
            new CodeSpace<>(Digit.class, segment.digits, digit -> digit.radix.applyAsInt(format));
        if (space.size() > (1 << Byte.SIZE))
          throw new IllegalArgumentException("Candidate code exceeds UINT8: " + segment);
        int repeats = segment == Segment.SEAT ? format.seats() : 1;
        for (int index = 0; index < repeats; index++) {
          int seat = segment == Segment.SEAT ? index : -1;
          String parameter = seat < 0 ? segment.parameter : segment.parameter + seat;
          entries.add(new Slot(segment, seat, parameter, column, rows, space));
          column += format.tiles();
          rows += space.size();
        }
      }
    }
    slots = List.copyOf(entries);
    codeWidth = column;
    tableRows = rows;
  }

  public int codeWidth() {
    return codeWidth;
  }

  public int tableRows() {
    return tableRows;
  }

  public List<Slot> slots() {
    return slots;
  }

  /** 公開順の入力から候補の区分を復元するために、入力の列・語彙の値・各桁の重みを推論カーネルに渡す。 */
  public int[] restoration(InputFormat format) {
    if (!format.eventsSettleOnPublication() || this != format.candidateCodes() || slots.isEmpty()) {
      throw new IllegalArgumentException(
          "candidate restoration requires public events and candidate segments");
    }
    var words = java.util.stream.IntStream.builder();
    var event = format.event();
    words.add(event.column(InputSchema.EventField.KIND));
    words.add(event.column(InputSchema.EventField.AUTHOR));
    words.add(event.column(InputSchema.EventField.TILE));
    words.add(event.column(InputSchema.EventField.RIICHI_DECLARATION));
    words.add(
        Ids.of(event.spec(InputSchema.EventField.KIND).values(), EventTable.PublicEvent.DISCARD));
    words.add(Ids.seat(0, format.seats()));
    words.add(format.tile().column(TileField.OWN_RIVER));
    words.add(format.seat().column(InputSchema.SeatField.RIICHI));
    words.add(
        Ids.of(
            format.seat().spec(InputSchema.SeatField.RIICHI).values(),
            com.epsilon.mahjong.score.RiichiState.NONE));
    var safety = format.tile().spec(TileField.SAFETY).values();
    words.add(Ids.of(safety, TileSafety.Safety.GENBUTSU));
    words.add(Ids.of(safety, TileSafety.Safety.PASSED_AFTER_RIICHI));
    words.add(Ids.of(safety, TileSafety.Safety.PASSED_THIS_CYCLE));
    words.add(format.tileKeyCode().first());
    var base = format.tileKeyCode().space();
    for (var digit :
        List.of(
            TileKeyCode.Digit.UKEIRE, TileKeyCode.Digit.WAIT_CLASS, TileKeyCode.Digit.DISCARDED)) {
      words.add(base.weight(digit));
      words.add(base.radix(digit));
    }
    for (Slot slot : slots) {
      var digits = slot.space();
      words.add(slot.seat);
      // 候補のドラの桁は表示牌の枚数だけを表す。牌特徴に含まれる抜きドラの加点を戻す。
      words.add(slot.segment == Segment.DORA ? format.tileTokens().nukiDoraToken() : -1);
      if (slot.segment == Segment.SEAT) {
        words.add(slot.seat == 0 ? -1 : format.tile().column(TileField.SAFETY, slot.seat - 1));
        words.add(digits.weight(Digit.WAIT));
        words.add(digits.weight(Digit.DISCARDED));
        words.add(digits.weight(Digit.SAFETY));
        words.add(digits.weight(Digit.RIICHI));
      } else {
        boolean river = slot.segment == Segment.RIVER;
        words.add(format.tile().column(river ? TileField.OWN_RIVER : TileField.DORA));
        words.add(digits.weight(Digit.UKEIRE));
        words.add(0);
        words.add(digits.weight(river ? Digit.OWN_RIVER : Digit.DORA));
        words.add(0);
      }
    }
    return words.build().toArray();
  }

  /** 各区分の列と行の範囲を推論カーネルに渡す。 */
  public SegmentLayout hip() {
    return new SegmentLayout(
        codeWidth,
        tableRows,
        slots.stream()
            .map(slot -> new SegmentSpan(slot.column(), slot.firstRow(), slot.rows()))
            .toList());
  }
}
