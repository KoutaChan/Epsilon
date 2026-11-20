#pragma once
#include "model_schema.h"
#include <algorithm>
#include <iterator>
#include <unordered_set>

namespace prism::schema {
inline int64_t weightBytes(const Model& model) {
  int64_t total = 0;
  for (const auto& w : model.weights) {
    require(w.element == Element::FP32, "Master weights must be FP32");
    int64_t n = 4;
    for (auto axis : w.axes) {
      require(axis > 0 && n <= INT64_MAX / axis, "Invalid weight axes");
      n *= axis;
    }
    require(n <= INT64_MAX - total, "Weight size overflow");
    total += n;
  }
  return total;
}
template <class Rows>
inline void uniqueNames(const Rows& rows) {
  std::unordered_set<std::string> names;
  for (const auto& item : rows) require(names.insert(item.name).second, "Duplicate model entry");
}
inline void validate(const Model& model) {
  int64_t column = 1 + static_cast<int64_t>(model.seats) + model.tiles;
  require(column + model.maxEvents <= INT32_MAX && static_cast<int64_t>(model.heads) * model.headWidth <= INT32_MAX,
          "Model dimensions overflow");
  require(!model.layers.empty() && !model.inputs.empty() && !model.stages.empty() && !model.weights.empty() &&
              !model.roles.empty() && !model.arithmetic.empty(),
          "Missing model components");
  uniqueNames(model.inputs);
  uniqueNames(model.weights);
  uniqueNames(model.roles);
  for (const auto& input : model.inputs) {
    auto section = std::find_if(std::begin(sections), std::end(sections),
                                [&](const auto& candidate) { return input.name == candidate.name; });
    require(section != std::end(sections) && input.axis == section->axis && input.element == section->element,
            "Invalid input axes or element");
    require(input.vocabularies.empty() ||
                (input.element == Element::U8 && input.vocabularies.size() == static_cast<size_t>(input.width)),
            "Invalid column vocabulary");
    for (auto n : input.vocabularies) require(n >= 1 && n <= 256, "Vocabulary outside UINT8");
  }
  for (size_t i = 0; i < model.layers.size(); ++i) {
    const auto& layer = model.layers[i];
    require(layer.source == (i == 0 ? Source::EMBEDDING : Source::PREVIOUS_LAYER), "Invalid layer input");
    require(layer.ffn != FeedForward::SHARED || layer.contextWidth == layer.tileWidth, "Shared FFN widths differ");
    require(layer.mask != Mask::PUBLIC_PREFIX || layer.queries == Queries::ALL_TOKENS,
            "Public prefix needs event queries");
  }
  int64_t row = 0;
  for (const auto& segment : model.segments) {
    require(segment.column == column && segment.firstRow == row && segment.rows <= 256, "Invalid candidate segment");
    column += model.tiles;
    row += segment.rows;
  }
  auto has = [&](Stage stage) {
    return std::find(model.stages.begin(), model.stages.end(), stage) != model.stages.end();
  };
  require(column == model.codeWidth && row == model.tableRows && has(Stage::SEGMENT_BIAS) == !model.segments.empty() &&
              has(Stage::COUNT_TILES) == (model.countWidth > 0),
          "Candidate axes do not match stages");
  weightBytes(model);
}
inline Model decode(const uint8_t* data, size_t length, int32_t abi) {
  InputReader in(data, length);
  require(abi == version && in.word() == magic && in.positive() == abi, "Declaration ABI mismatch");
  auto bytes = in.size();
  require(static_cast<size_t>(bytes) == in.remaining(), "Declaration byte length mismatch");
  auto model = readModel(in);
  require(in.remaining() == 0, "Unknown required declaration fields");
  validate(model);
  return model;
}
inline std::vector<uint8_t> encode(const Model& model) {
  validate(model);
  OutputWriter body;
  write(body, model);
  OutputWriter out;
  out.word(magic);
  out.positive(version);
  out.count(body.bytes.size());
  out.bytes.insert(out.bytes.end(), body.bytes.begin(), body.bytes.end());
  return out.bytes;
}
}
