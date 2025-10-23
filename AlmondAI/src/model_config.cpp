#include "../include/almondai/model_config.hpp"

#include <cstdlib>
#include <optional>

namespace almondai {

namespace {

std::optional<std::size_t> parse_size_env(const char* name) {
    if (const char* value = std::getenv(name)) {
        try {
            return static_cast<std::size_t>(std::stoull(value));
        } catch (...) {
        }
    }
    return std::nullopt;
}

std::optional<double> parse_double_env(const char* name) {
    if (const char* value = std::getenv(name)) {
        try {
            return std::stod(value);
        } catch (...) {
        }
    }
    return std::nullopt;
}

} // namespace

ModelHyperParameters resolve_model_hyperparameters() {
    ModelHyperParameters params;
    if (auto width = parse_size_env("ALMONDAI_WIDTH")) {
        params.hidden_width = *width;
    }
    if (auto seq_len = parse_size_env("ALMONDAI_SEQ_LEN")) {
        params.sequence_length = *seq_len;
    }
    if (auto dropout = parse_double_env("ALMONDAI_DROPOUT")) {
        params.dropout = *dropout;
    }
    return params;
}

} // namespace almondai

