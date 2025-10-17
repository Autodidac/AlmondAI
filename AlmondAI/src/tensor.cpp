#include "../AlmondAI/include/almondai/tensor.hpp"

#include <random>
#include <chrono>

namespace almondai {

Tensor Tensor::random(std::initializer_list<std::size_t> shape, double scale) {
    Tensor t(shape);
    const auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, scale);
    for (double& value : t.vector()) {
        value = dist(rng);
    }
    return t;
}

} // namespace almondai

