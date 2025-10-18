#pragma once

#include "../json.hpp"
#include "../net/http.hpp"

#include <memory>
#include <string>
#include <vector>

namespace almondai::chat {

struct Message {
    std::string role;
    std::string text;
};

struct Backend {
    virtual ~Backend() = default;
    virtual std::string complete(const std::vector<Message>& messages) = 0;
};

using BackendPtr = std::unique_ptr<Backend>;

enum class Kind {
    Rasa,
    Botpress,
    DeepPavlov,
    H2O,
    OpenAICompat,
    HuggingFace,
    OpenRouter,
    TogetherAI,
    DeepInfra
};

BackendPtr make_backend(Kind kind, std::string a, std::string b = std::string(), std::string c = std::string());

Kind parse_kind(const std::string& name);
std::string kind_to_string(Kind kind);

} // namespace almondai::chat

