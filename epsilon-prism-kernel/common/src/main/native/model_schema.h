#pragma once
#include "declaration_io.h"
namespace prism::schema {
inline constexpr int32_t version =
#include "declaration_version.inc"
    ;
inline constexpr int32_t magic = 1145918032;
enum class Element : int32_t { U8 = 1, BF16 = 2, FP32 = 3, I32 = 4 };
inline Element readElement(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Element::U8;
    case 2:
      return Element::BF16;
    case 3:
      return Element::FP32;
    case 4:
      return Element::I32;
    default:
      throw std::invalid_argument("Unknown Element ID");
  }
}
inline void write(OutputWriter& out, Element value) { out.positive(static_cast<int32_t>(value)); }
enum class Axis : int32_t { ROW = 1, SEAT = 2, TILE = 3, EVENT = 4, TOKEN = 5, TRANSITION = 6, ACTION = 7 };
inline Axis readAxis(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Axis::ROW;
    case 2:
      return Axis::SEAT;
    case 3:
      return Axis::TILE;
    case 4:
      return Axis::EVENT;
    case 5:
      return Axis::TOKEN;
    case 6:
      return Axis::TRANSITION;
    case 7:
      return Axis::ACTION;
    default:
      throw std::invalid_argument("Unknown Axis ID");
  }
}
inline void write(OutputWriter& out, Axis value) { out.positive(static_cast<int32_t>(value)); }
enum class Queries : int32_t { ALL_TOKENS = 1, FIXED_TOKENS = 2 };
inline Queries readQueries(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Queries::ALL_TOKENS;
    case 2:
      return Queries::FIXED_TOKENS;
    default:
      throw std::invalid_argument("Unknown Queries ID");
  }
}
inline void write(OutputWriter& out, Queries value) { out.positive(static_cast<int32_t>(value)); }
enum class Mask : int32_t { ALL_KEYS = 1, PUBLIC_PREFIX = 2 };
inline Mask readMask(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Mask::ALL_KEYS;
    case 2:
      return Mask::PUBLIC_PREFIX;
    default:
      throw std::invalid_argument("Unknown Mask ID");
  }
}
inline void write(OutputWriter& out, Mask value) { out.positive(static_cast<int32_t>(value)); }
enum class Source : int32_t { EMBEDDING = 1, PREVIOUS_LAYER = 2 };
inline Source readSource(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Source::EMBEDDING;
    case 2:
      return Source::PREVIOUS_LAYER;
    default:
      throw std::invalid_argument("Unknown Source ID");
  }
}
inline void write(OutputWriter& out, Source value) { out.positive(static_cast<int32_t>(value)); }
enum class Normalization : int32_t { PRE_ATTENTION_AND_PRE_FFN = 1 };
inline Normalization readNormalization(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Normalization::PRE_ATTENTION_AND_PRE_FFN;
    default:
      throw std::invalid_argument("Unknown Normalization ID");
  }
}
inline void write(OutputWriter& out, Normalization value) { out.positive(static_cast<int32_t>(value)); }
enum class Order : int32_t { LEGAL_ACTION = 1 };
inline Order readOrder(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Order::LEGAL_ACTION;
    default:
      throw std::invalid_argument("Unknown Order ID");
  }
}
inline void write(OutputWriter& out, Order value) { out.positive(static_cast<int32_t>(value)); }
enum class FeedForward : int32_t { SHARED = 1, BY_TOKEN = 2 };
inline FeedForward readFeedForward(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return FeedForward::SHARED;
    case 2:
      return FeedForward::BY_TOKEN;
    default:
      throw std::invalid_argument("Unknown FeedForward ID");
  }
}
inline void write(OutputWriter& out, FeedForward value) { out.positive(static_cast<int32_t>(value)); }
enum class Stage : int32_t {
  COUNT_TILES = 1,
  SEGMENT_BIAS = 2,
  CANDIDATE_ATTENTION = 3,
  CANDIDATE_FFN = 4,
  EVENT_REREAD = 5,
  DISCARD_VALUE = 6,
  COMPOSE_VALUES = 7
};
inline Stage readStage(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Stage::COUNT_TILES;
    case 2:
      return Stage::SEGMENT_BIAS;
    case 3:
      return Stage::CANDIDATE_ATTENTION;
    case 4:
      return Stage::CANDIDATE_FFN;
    case 5:
      return Stage::EVENT_REREAD;
    case 6:
      return Stage::DISCARD_VALUE;
    case 7:
      return Stage::COMPOSE_VALUES;
    default:
      throw std::invalid_argument("Unknown Stage ID");
  }
}
inline void write(OutputWriter& out, Stage value) { out.positive(static_cast<int32_t>(value)); }
enum class Composition : int32_t { DISCARD_VALUE = 1, RIICHI_GATE = 2, RESPONSE_GATE = 3, TURN_GATE = 4, ZERO = 5 };
inline Composition readComposition(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Composition::DISCARD_VALUE;
    case 2:
      return Composition::RIICHI_GATE;
    case 3:
      return Composition::RESPONSE_GATE;
    case 4:
      return Composition::TURN_GATE;
    case 5:
      return Composition::ZERO;
    default:
      throw std::invalid_argument("Unknown Composition ID");
  }
}
inline void write(OutputWriter& out, Composition value) { out.positive(static_cast<int32_t>(value)); }
enum class Arithmetic : int32_t {
  EMBED_COLUMN_FMA = 1,
  RMS_XOR_1_2 = 2,
  BF16_TILE_K_ASCENDING = 3,
  PAIR_BIAS_01_23_45_6 = 4,
  PREFIX_EDGE_FP32_FMA = 5,
  PREFIX_FULL_SPLIT_BF16 = 6,
  RESIDUAL_ROUND_BF16 = 7,
  COUNT_TILE_MUL_ADD = 8,
  SEGMENT_DECLARATION_ORDER = 9,
  CANDIDATE_KEY_ORDER = 10,
  VALUE_ACTION_ORDER = 11,
  FOLDED_QKV_UNROUNDED_RMS = 12
};
inline Arithmetic readArithmetic(InputReader& in) {
  switch (in.positive()) {
    case 1:
      return Arithmetic::EMBED_COLUMN_FMA;
    case 2:
      return Arithmetic::RMS_XOR_1_2;
    case 3:
      return Arithmetic::BF16_TILE_K_ASCENDING;
    case 4:
      return Arithmetic::PAIR_BIAS_01_23_45_6;
    case 5:
      return Arithmetic::PREFIX_EDGE_FP32_FMA;
    case 6:
      return Arithmetic::PREFIX_FULL_SPLIT_BF16;
    case 7:
      return Arithmetic::RESIDUAL_ROUND_BF16;
    case 8:
      return Arithmetic::COUNT_TILE_MUL_ADD;
    case 9:
      return Arithmetic::SEGMENT_DECLARATION_ORDER;
    case 10:
      return Arithmetic::CANDIDATE_KEY_ORDER;
    case 11:
      return Arithmetic::VALUE_ACTION_ORDER;
    case 12:
      return Arithmetic::FOLDED_QKV_UNROUNDED_RMS;
    default:
      throw std::invalid_argument("Unknown Arithmetic ID");
  }
}
inline void write(OutputWriter& out, Arithmetic value) { out.positive(static_cast<int32_t>(value)); }
struct Section {
  const char* name;
  Axis axis;
  Element element;
};
inline constexpr Section sections[] = {
    {"ROUND_CATEGORY", Axis::ROW, Element::U8},
    {"ROUND_SCALAR", Axis::ROW, Element::BF16},
    {"SEAT_CATEGORY", Axis::SEAT, Element::U8},
    {"SEAT_SCALAR", Axis::SEAT, Element::BF16},
    {"TILE_CATEGORY", Axis::TILE, Element::U8},
    {"EVENT_CATEGORY", Axis::EVENT, Element::U8},
    {"EVENT_SCALAR", Axis::EVENT, Element::BF16},
    {"RELATION", Axis::TOKEN, Element::U8},
    {"TRANSITION_CATEGORY", Axis::TRANSITION, Element::U8},
    {"TRANSITION_SCALAR", Axis::TRANSITION, Element::BF16},
    {"TRANSITION_CODE", Axis::TRANSITION, Element::U8},
    {"DISCARD_EXACT", Axis::TRANSITION, Element::FP32},
    {"ACTION_COLUMN", Axis::ACTION, Element::I32},
    {"ACTION_EXACT", Axis::ACTION, Element::FP32},
    {"ROW_COLUMN", Axis::ROW, Element::I32},
};
struct Input {
  std::string name;
  Axis axis;
  Element element;
  int32_t width;
  std::vector<int32_t> vocabularies;
};
inline Input readInput(InputReader& in) {
  return Input{in.text(), readAxis(in), readElement(in), in.positive(), in.list([&]() { return in.positive(); })};
}
inline void write(OutputWriter& out, const Input& value) {
  out.text(value.name);
  write(out, value.axis);
  write(out, value.element);
  out.positive(value.width);
  out.count(value.vocabularies.size());
  for (const auto& item : value.vocabularies) {
    out.positive(item);
  }
}
struct Layer {
  Queries queries;
  Queries keys;
  Mask mask;
  Source source;
  Normalization normalization;
  FeedForward ffn;
  int32_t contextWidth;
  int32_t tileWidth;
};
inline Layer readLayer(InputReader& in) {
  return Layer{readQueries(in),       readQueries(in),     readMask(in),  readSource(in),
               readNormalization(in), readFeedForward(in), in.positive(), in.positive()};
}
inline void write(OutputWriter& out, const Layer& value) {
  write(out, value.queries);
  write(out, value.keys);
  write(out, value.mask);
  write(out, value.source);
  write(out, value.normalization);
  write(out, value.ffn);
  out.positive(value.contextWidth);
  out.positive(value.tileWidth);
}
struct Segment {
  std::string parameter;
  int32_t column;
  int32_t firstRow;
  int32_t rows;
};
inline Segment readSegment(InputReader& in) { return Segment{in.text(), in.size(), in.size(), in.positive()}; }
inline void write(OutputWriter& out, const Segment& value) {
  out.text(value.parameter);
  out.size(value.column);
  out.size(value.firstRow);
  out.positive(value.rows);
}
struct Role {
  std::string name;
  Composition composition;
};
inline Role readRole(InputReader& in) { return Role{in.text(), readComposition(in)}; }
inline void write(OutputWriter& out, const Role& value) {
  out.text(value.name);
  write(out, value.composition);
}
struct Weight {
  std::string name;
  Element element;
  std::vector<int32_t> axes;
};
inline Weight readWeight(InputReader& in) {
  return Weight{in.text(), readElement(in), in.list([&]() { return in.positive(); })};
}
inline void write(OutputWriter& out, const Weight& value) {
  out.text(value.name);
  write(out, value.element);
  out.count(value.axes.size());
  for (const auto& item : value.axes) {
    out.positive(item);
  }
}
struct Model {
  int32_t seats;
  int32_t tiles;
  int32_t maxEvents;
  int32_t hidden;
  int32_t heads;
  int32_t headWidth;
  float epsilon;
  float attentionScale;
  float temperature;
  int32_t gateWidth;
  int32_t candidateWidth;
  int32_t keyCodes;
  int32_t tileKeyFirstCode;
  int32_t tileRelations;
  Element activation;
  Element accumulation;
  Element score;
  Order representativeOrder;
  int32_t countWidth;
  int32_t codeWidth;
  int32_t tableRows;
  std::vector<Input> inputs;
  std::vector<Layer> layers;
  std::vector<Segment> segments;
  std::vector<Stage> stages;
  std::vector<Role> roles;
  std::vector<Arithmetic> arithmetic;
  std::vector<Weight> weights;
};
inline Model readModel(InputReader& in) {
  return Model{in.positive(),
               in.positive(),
               in.positive(),
               in.positive(),
               in.positive(),
               in.positive(),
               in.scale(),
               in.scale(),
               in.scale(),
               in.positive(),
               in.positive(),
               in.positive(),
               in.size(),
               in.positive(),
               readElement(in),
               readElement(in),
               readElement(in),
               readOrder(in),
               in.size(),
               in.positive(),
               in.size(),
               in.list([&]() { return readInput(in); }),
               in.list([&]() { return readLayer(in); }),
               in.list([&]() { return readSegment(in); }),
               in.list([&]() { return readStage(in); }),
               in.list([&]() { return readRole(in); }),
               in.list([&]() { return readArithmetic(in); }),
               in.list([&]() { return readWeight(in); })};
}
inline void write(OutputWriter& out, const Model& value) {
  out.positive(value.seats);
  out.positive(value.tiles);
  out.positive(value.maxEvents);
  out.positive(value.hidden);
  out.positive(value.heads);
  out.positive(value.headWidth);
  out.scale(value.epsilon);
  out.scale(value.attentionScale);
  out.scale(value.temperature);
  out.positive(value.gateWidth);
  out.positive(value.candidateWidth);
  out.positive(value.keyCodes);
  out.size(value.tileKeyFirstCode);
  out.positive(value.tileRelations);
  write(out, value.activation);
  write(out, value.accumulation);
  write(out, value.score);
  write(out, value.representativeOrder);
  out.size(value.countWidth);
  out.positive(value.codeWidth);
  out.size(value.tableRows);
  out.count(value.inputs.size());
  for (const auto& item : value.inputs) {
    write(out, item);
  }
  out.count(value.layers.size());
  for (const auto& item : value.layers) {
    write(out, item);
  }
  out.count(value.segments.size());
  for (const auto& item : value.segments) {
    write(out, item);
  }
  out.count(value.stages.size());
  for (const auto& item : value.stages) {
    write(out, item);
  }
  out.count(value.roles.size());
  for (const auto& item : value.roles) {
    write(out, item);
  }
  out.count(value.arithmetic.size());
  for (const auto& item : value.arithmetic) {
    write(out, item);
  }
  out.count(value.weights.size());
  for (const auto& item : value.weights) {
    write(out, item);
  }
}
}
