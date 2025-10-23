#include "../include/almondai/model_config.hpp"

#include <cstdlib>
#include <memory>
#include <optional>
#include <string>

namespace almondai {

namespace {

std::optional<std::string> read_env(const char* name) {
#ifdef _WIN32
    size_t required = 0;
    char* buffer = nullptr;
    if (_dupenv_s(&buffer, &required, name) != 0) {
        return std::nullopt;
    }
    std::unique_ptr<char, decltype(&std::free)> holder(buffer, &std::free);
    if (!buffer) {
        return std::nullopt;
    }
    return std::string(buffer);
#else
    if (const char* value = std::getenv(name)) {
        return std::string(value);
    }
    return std::nullopt;
#endif
}

std::optional<std::size_t> parse_size_env(const char* name) {
    if (auto value = read_env(name)) {
        try {
            return static_cast<std::size_t>(std::stoull(*value));
        } catch (...) {
        }
    }
    return std::nullopt;
}

std::optional<double> parse_double_env(const char* name) {
    if (auto value = read_env(name)) {
        try {
            return std::stod(*value);
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

