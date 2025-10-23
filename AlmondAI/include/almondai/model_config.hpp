#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace almondai {

struct ModelHyperParameters {
    std::size_t hidden_width = 256;
    std::size_t sequence_length = 512;
    double dropout = 0.1;
};

ModelHyperParameters resolve_model_hyperparameters();

} // namespace almondai

