#include "model_declaration.h"
#include <fstream>
#include <iterator>

namespace {
// Java と native の両方が同じ宣言を書き、weight の名前だけで書き手を見分ける。
prism::schema::Model fixture(const std::string& weight) {
  using namespace prism::schema;
  return {3,
          27,
          48,
          96,
          3,
          16,
          1e-5f,
          .25f,
          .01f,
          64,
          128,
          96,
          1,
          4,
          Element::BF16,
          Element::FP32,
          Element::FP32,
          Order::LEGAL_ACTION,
          0,
          31,
          0,
          {{"ACTION_COLUMN", Axis::ACTION, Element::I32, 1, {}}},
          {{Queries::ALL_TOKENS, Queries::ALL_TOKENS, Mask::ALL_KEYS, Source::EMBEDDING,
            Normalization::PRE_ATTENTION_AND_PRE_FFN, FeedForward::SHARED, 128, 128}},
          {},
          {Stage::COMPOSE_VALUES},
          {{"PASS", Composition::ZERO}},
          {Arithmetic::VALUE_ACTION_ORDER},
          {{weight, Element::FP32, {1}}}};
}
}

int main(int argc, char** argv) {
  if (argc != 3) return 2;
  std::string mode(argv[1]);
  if (mode == "--write") {
    auto bytes = prism::schema::encode(fixture("native"));
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return out.good() ? 0 : 1;
  }
  std::ifstream file(mode == "--check-java" ? argv[2] : argv[1], std::ios::binary);
  std::vector<uint8_t> data{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
  try {
    auto model = prism::schema::decode(data.data(), data.size(), prism::schema::version);
    auto encoded = prism::schema::encode(model);
    if (encoded != data) return 1;
    if (mode == "--check-java") return encoded == prism::schema::encode(fixture("fixture")) ? 0 : 1;
    std::ofstream out(argv[2], std::ios::binary);
    out.write(reinterpret_cast<const char*>(encoded.data()), encoded.size());
    return out.good() ? 0 : 2;
  } catch (const std::exception&) {
    return 1;
  }
}
