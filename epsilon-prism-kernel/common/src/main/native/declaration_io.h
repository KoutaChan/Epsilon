#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>
namespace prism::schema {
inline void require(bool ok, const char* reason) {
  if (!ok) throw std::invalid_argument(reason);
}
inline void validText(const std::string& text) {
  require(!text.empty(), "Empty text");
  for (unsigned char c : text) require(c < 128, "Expected ASCII text");
}
class InputReader {
 public:
  InputReader(const uint8_t* data, size_t bytes) : data_(data), left_(bytes) {
    require(data || !bytes, "Null declaration");
  }
  size_t remaining() const { return left_; }
  uint32_t word() {
    require(left_ >= 4, "Truncated declaration");
    uint32_t v = 0;
    for (int i = 0; i < 4; ++i) v |= uint32_t(data_[i]) << (8 * i);
    data_ += 4;
    left_ -= 4;
    return v;
  }
  int32_t size() {
    auto n = word();
    require(n <= INT32_MAX, "Negative size");
    return static_cast<int32_t>(n);
  }
  int32_t positive() {
    auto n = size();
    require(n > 0, "Expected a positive dimension");
    return n;
  }
  float scale() {
    auto n = word();
    float f;
    std::memcpy(&f, &n, 4);
    require(f > 0 && std::isfinite(f), "Invalid scale");
    return f;
  }
  std::string text() {
    auto n = positive();
    require(static_cast<size_t>(n) <= left_, "Truncated string");
    std::string s(reinterpret_cast<const char*>(data_), n);
    validText(s);
    data_ += n;
    left_ -= n;
    return s;
  }
  template <class Read>
  auto list(Read read) {
    auto n = size();
    require(static_cast<size_t>(n) <= left_ / 4, "Truncated list");
    std::vector<decltype(read())> values;
    for (int32_t i = 0; i < n; ++i) values.push_back(read());
    return values;
  }

 private:
  const uint8_t* data_;
  size_t left_;
};
class OutputWriter {
 public:
  std::vector<uint8_t> bytes;
  void word(uint32_t n) {
    for (int i = 0; i < 4; ++i) bytes.push_back(static_cast<uint8_t>(n >> (8 * i)));
  }
  void size(int32_t n) {
    require(n >= 0, "Negative size");
    word(n);
  }
  void count(size_t n) {
    require(n <= INT32_MAX, "Size overflow");
    word(static_cast<uint32_t>(n));
  }
  void positive(int32_t n) {
    require(n > 0, "Expected a positive dimension");
    word(n);
  }
  void scale(float f) {
    require(f > 0 && std::isfinite(f), "Invalid scale");
    uint32_t n;
    std::memcpy(&n, &f, 4);
    word(n);
  }
  void text(const std::string& s) {
    validText(s);
    count(s.size());
    bytes.insert(bytes.end(), s.begin(), s.end());
  }
};
}
