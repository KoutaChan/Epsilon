#include <stdexcept>
#include <string>

#include "relation_encoding.h"
#include "training/training_launch.h"
#include "training_device.h"

#include "training/relation_codes.cuh"

namespace prism {

void checkRelationCodeShape(const RelationCodeShape& shape) {
  auto require = [](bool condition, const std::string& message) {
    if (!condition) {
      throw std::invalid_argument(message);
    }
  };
  const TokenLayout& layout = shape.layout;
  const RelationColumns& c = shape.columns;
  require(layout.rows > 0, "relation codes need rows, got " + std::to_string(layout.rows));
  require(layout.seats > 0 && 3 * layout.seats < kRelationEntries && layout.tiles > 0 && layout.tiles <= kMaxTiles,
          "seats " + std::to_string(layout.seats) + " and tiles " + std::to_string(layout.tiles) + " are out of range");
  require(shape.events >= 0, "event count must not be negative, got " + std::to_string(shape.events));
  require(shape.tokens >= layout.fixed() && shape.tokens <= kMaxRowTokens,
          "row tokens must be in " + std::to_string(layout.fixed()) + ".." + std::to_string(kMaxRowTokens) + ", got " +
              std::to_string(shape.tokens));
  require(shape.rowWidth > 0 && shape.rowFirstEvent >= 0 && shape.rowFirstEvent < shape.rowWidth &&
              shape.rowEvents >= 0 && shape.rowEvents < shape.rowWidth,
          "row column out of range");
  require(c.width > 0 && c.width <= kMaxAttributes, "relation attributes must be in 1.." +
                                                        std::to_string(kMaxAttributes) + " columns, got " +
                                                        std::to_string(c.width));
  for (int column : {c.tile, c.owner, c.event, c.riichi, c.lastDiscard, c.link}) {
    require(column >= 0 && column < c.width, "relation column out of range");
  }
  require(c.safety >= 0 && c.safeties >= 0 && c.safety + c.safeties <= c.width, "relation safety columns out of range");
  require(c.safeties <= kRelationOpponents, "relation codes hold the safety of at most " +
                                                std::to_string(kRelationOpponents) + " opponents, got " +
                                                std::to_string(c.safeties) + " safety columns");
}

void relationCodes(const RelationCodeShape& shape, const RelationCodeArrays& arrays, gpu::Stream stream) {
  checkRelationCodeShape(shape);
  if (reinterpret_cast<uintptr_t>(arrays.codes) % alignof(uint32_t) != 0) {
    throw std::invalid_argument("relation codes must be aligned to 4 bytes");
  }
  const size_t rows = static_cast<size_t>(shape.layout.rows);
  const size_t tokens = static_cast<size_t>(shape.tokens);
  RelationCodeArgs args{};
  args.shape = shape;
  args.relation = {arrays.relation, (rows * shape.layout.fixed() + shape.events) * shape.columns.width};
  args.rowColumn = {arrays.rowColumn, rows * shape.rowWidth};
  args.tileRelation = {arrays.tileRelation, static_cast<size_t>(shape.layout.tiles) * shape.layout.tiles};
  args.codes = {arrays.codes, rows * tokens * tokens};
  relationCodeKernel<<<dim3(shape.layout.rows), dim3(kThreads), 0, stream>>>(args);
  PRISM_GPU(gpu::lastError());
}

}  // namespace prism
