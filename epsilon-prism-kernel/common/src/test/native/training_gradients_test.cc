#include "training/gradients.h"

#include <array>
#include <cassert>
#include <cmath>
#include <vector>

namespace {
constexpr int features = 16, heads = 2, half = heads * features, width = half * 2;
constexpr int keys = 3, transitions = 4, codes = 3;
constexpr float scale = 0.25f;
using Table = std::array<double, codes * width>;
using Vectors = std::array<float, transitions * half>;
using Weights = std::array<float, transitions * heads * keys>;

int code(int t, int k) { return (k + t + t * k) % codes; }

// FP64 の順伝播からスカラー損失を求める。必要なときは各キーの確率と上流勾配も保存する。
double loss(const Table& table, const Vectors& query, const Vectors& upstream, Weights* probabilities = nullptr,
            Weights* gradients = nullptr) {
  double total = 0;
  for (int t = 0; t < transitions; ++t) {
    for (int h = 0; h < heads; ++h) {
      std::array<double, keys> exp{}, value{};
      double denom = 0;
      for (int k = 0; k < keys; ++k) {
        int row = code(t, k);
        double logit = 0;
        for (int i = 0; i < features; ++i) {
          int c = h * features + i;
          logit += query[t * half + c] * table[row * width + c];
          value[k] += upstream[t * half + c] * table[row * width + half + c];
        }
        exp[k] = std::exp(scale * logit);
        denom += exp[k];
      }
      double mean = 0;
      for (int k = 0; k < keys; ++k) mean += exp[k] / denom * value[k];
      total += mean;
      if (probabilities) {
        for (int k = 0; k < keys; ++k) {
          int at = (t * heads + h) * keys + k;
          (*probabilities)[at] = static_cast<float>(exp[k] / denom);
          (*gradients)[at] = prism::training::scoreGradient((*probabilities)[at], value[k], mean);
        }
      }
    }
  }
  return total;
}

void tableGradients() {
  Table table;
  Vectors query, upstream;
  Weights probabilities, gradients;
  for (int i = 0; i < table.size(); ++i) table[i] = std::sin(i * 0.37);
  for (int i = 0; i < query.size(); ++i) {
    query[i] = std::sin(i * 0.13);
    upstream[i] = std::cos(i * 0.07);
  }
  loss(table, query, upstream, &probabilities, &gradients);
  constexpr double step = 1e-5;
  for (int row = 0; row < codes; ++row) {
    std::vector<int32_t> pairs;
    for (int t = 0; t < transitions; ++t)
      for (int k = 0; k < keys; ++k)
        if (code(t, k) == row) pairs.push_back(t * keys + k);
    for (int c = 0; c < width; ++c) {
      auto partial = [&](int first, int last) {
        return prism::training::tablePartial<width, features>(keys, c, scale, query.data(), upstream.data(),
                                                              gradients.data(), probabilities.data(), pairs.data(),
                                                              first, last);
      };
      int n = static_cast<int>(pairs.size());
      double actual = partial(0, n / 2) + partial(n / 2, n);
      auto plus = table, minus = table;
      plus[row * width + c] += step;
      minus[row * width + c] -= step;
      double expected = (loss(plus, query, upstream) - loss(minus, query, upstream)) / (2 * step);
      assert(std::abs(actual - expected) < 1e-6);
    }
  }
}

void countGradients() {
  constexpr double step = 1e-5;
  for (double x : {-4.0, -0.3, 0.0, 1.0, 5.0}) {
    constexpr float tile = -1.25f, upstream = 0.625f;
    auto loss = [upstream](double logit, double value) { return value * upstream / (1.0 + std::exp(-logit)); };
    float gate = prism::training::countGate(x);
    double dx = (loss(x + step, tile) - loss(x - step, tile)) / (2 * step);
    double dt = (loss(x, tile + step) - loss(x, tile - step)) / (2 * step);
    assert(std::abs(prism::training::gateGradient(upstream, tile, gate) - dx) < 1e-7);
    assert(std::abs(upstream * gate - dt) < 1e-7);
  }
}
}  // namespace

int main() {
  tableGradients();
  countGradients();
}
