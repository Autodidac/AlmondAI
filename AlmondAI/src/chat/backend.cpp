
#include "../../include/almondai/chat/backend.hpp"


#include <algorithm>
#include <cctype>
#include <iterator>
#include <sstream>
#include <stdexcept>

namespace {

using almondai::Json;
using almondai::JsonArray;
using almondai::JsonObject;
using almondai::chat::Message;

const Message* last_with_role(const std::vector<Message>& messages, const std::string& role) {
    for (auto it = messages.rbegin(); it != messages.rend(); ++it) {
        if (it->role == role) {
            return &*it;
        }
    }
    return nullptr;
}

JsonArray serialize_chat_messages(const std::vector<Message>& messages) {
    JsonArray array;
    for (const auto& msg : messages) {
        JsonObject entry;
        entry["role"] = Json(msg.role);
        entry["content"] = Json(msg.text);
        array.emplace_back(Json(entry));
    }
    return array;
}

std::string strip(const std::string& text) {
    auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char c) { return std::isspace(c) != 0; });
    auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char c) { return std::isspace(c) != 0; }).base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
}

} // namespace

namespace almondai::chat {

namespace {

class RasaBackend final : public Backend {
public:
    RasaBackend(std::string endpoint, std::string sender)
        : m_endpoint(std::move(endpoint)), m_sender_id(std::move(sender)) {}

    std::string complete(const std::vector<Message>& messages) override {
        const Message* user = last_with_role(messages, "user");
        if (!user) {
            throw std::runtime_error("rasa backend requires at least one user message");
        }

        JsonObject payload;
        payload["sender"] = Json(m_sender_id);
        payload["message"] = Json(user->text);

        const std::string response = almondai::net::post_json(m_endpoint, Json(payload).dump(), {});
        Json parsed = Json::parse(response);
        if (parsed.is_array()) {
            const auto& events = parsed.as_array();
            for (const auto& event : events) {
                if (event.is_object()) {
                    const auto& obj = event.as_object();
                    if (auto it = obj.find("text"); it != obj.end() && it->second.is_string()) {
                        const std::string text = strip(it->second.as_string());
                        if (!text.empty()) {
                            return text;
                        }
                    }
                }
            }
        }
        if (parsed.is_object()) {
            const auto& obj = parsed.as_object();
            if (auto it = obj.find("text"); it != obj.end() && it->second.is_string()) {
                const std::string text = strip(it->second.as_string());
                if (!text.empty()) {
                    return text;
                }
            }
        }
        return {};
    }

private:
    std::string m_endpoint;
    std::string m_sender_id;
};

class BotpressBackend final : public Backend {
public:
    BotpressBackend(std::string endpoint, std::string conversation, std::string token)
        : m_endpoint(std::move(endpoint)),
          m_conversation_id(std::move(conversation)),
          m_token(std::move(token)) {}

    std::string complete(const std::vector<Message>& messages) override {
        const Message* user = last_with_role(messages, "user");
        if (!user) {
            throw std::runtime_error("botpress backend requires at least one user message");
        }

        JsonObject payload;
        payload["type"] = Json("text");
        JsonObject nested;
        nested["text"] = Json(user->text);
        payload["payload"] = Json(nested);
        payload["conversationId"] = Json(m_conversation_id);

        std::vector<std::pair<std::string, std::string>> headers;
        if (!m_token.empty()) {
            headers.emplace_back("Authorization", "Bearer " + m_token);
        }

        const std::string response = almondai::net::post_json(m_endpoint, Json(payload).dump(), headers);
        Json parsed = Json::parse(response);
        if (parsed.is_object()) {
            const auto& obj = parsed.as_object();
            if (auto it = obj.find("responses"); it != obj.end() && it->second.is_array()) {
                const auto& responses = it->second.as_array();
                if (!responses.empty() && responses.front().is_object()) {
                    const auto& entry = responses.front().as_object();
                    if (auto payload_it = entry.find("payload"); payload_it != entry.end() && payload_it->second.is_object()) {
                        const auto& payload_obj = payload_it->second.as_object();
                        if (auto text_it = payload_obj.find("text"); text_it != payload_obj.end() && text_it->second.is_string()) {
                            return strip(text_it->second.as_string());
                        }
                    }
                }
            }
        }
        return {};
    }

private:
    std::string m_endpoint;
    std::string m_conversation_id;
    std::string m_token;
};

class DeepPavlovBackend final : public Backend {
public:
    explicit DeepPavlovBackend(std::string endpoint)
        : m_endpoint(std::move(endpoint)) {}

    std::string complete(const std::vector<Message>& messages) override {
        const Message* user = last_with_role(messages, "user");
        if (!user) {
            throw std::runtime_error("deeppavlov backend requires at least one user message");
        }
        JsonObject payload;
        JsonArray questions;
        questions.emplace_back(Json(user->text));
        payload["questions"] = Json(questions);

        const std::string response = almondai::net::post_json(m_endpoint, Json(payload).dump(), {});
        Json parsed = Json::parse(response);
        if (parsed.is_array()) {
            const auto& arr = parsed.as_array();
            if (!arr.empty() && arr.front().is_string()) {
                return strip(arr.front().as_string());
            }
            if (!arr.empty() && arr.front().is_array()) {
                const auto& inner = arr.front().as_array();
                if (!inner.empty() && inner.front().is_string()) {
                    return strip(inner.front().as_string());
                }
            }
        }
        if (parsed.is_object()) {
            const auto& obj = parsed.as_object();
            if (auto it = obj.find("results"); it != obj.end() && it->second.is_array()) {
                const auto& results = it->second.as_array();
                if (!results.empty() && results.front().is_string()) {
                    return strip(results.front().as_string());
                }
            }
        }
        return {};
    }

private:
    std::string m_endpoint;
};

class OpenAIBackend final : public Backend {
public:
    OpenAIBackend(std::string endpoint, std::string model, std::string api_key)
        : m_endpoint(std::move(endpoint)),
          m_model(std::move(model)),
          m_api_key(std::move(api_key)) {}

    std::string complete(const std::vector<Message>& messages) override {
        if (messages.empty()) {
            throw std::runtime_error("openai backend requires at least one message");
        }

        JsonObject payload;
        payload["model"] = Json(m_model);
        payload["messages"] = Json(serialize_chat_messages(messages));

        const std::string body = Json(payload).dump();
        std::vector<std::pair<std::string, std::string>> headers;
        if (!m_api_key.empty()) {
            headers.emplace_back("Authorization", "Bearer " + m_api_key);
        }

        const std::string response = almondai::net::post_json(m_endpoint, body, headers);
        Json parsed = Json::parse(response);
        if (parsed.is_object()) {
            const auto& obj = parsed.as_object();
            if (auto choices_it = obj.find("choices"); choices_it != obj.end() && choices_it->second.is_array()) {
                const auto& choices = choices_it->second.as_array();
                if (!choices.empty() && choices.front().is_object()) {
                    const auto& choice = choices.front().as_object();
                    if (auto msg_it = choice.find("message"); msg_it != choice.end() && msg_it->second.is_object()) {
                        const auto& message = msg_it->second.as_object();
                        if (auto content_it = message.find("content"); content_it != message.end() && content_it->second.is_string()) {
                            return strip(content_it->second.as_string());
                        }
                    }
                    if (auto text_it = choice.find("text"); text_it != choice.end() && text_it->second.is_string()) {
                        return strip(text_it->second.as_string());
                    }
                }
            }
        }
        return {};
    }

private:
    std::string m_endpoint;
    std::string m_model;
    std::string m_api_key;
};

} // namespace

BackendPtr make_backend(Kind kind, std::string a, std::string b, std::string c) {
    switch (kind) {
    case Kind::Rasa:
        if (a.empty()) {
            throw std::runtime_error("rasa backend requires endpoint");
        }
        if (b.empty()) {
            b = "almondai";
        }
        return std::make_unique<RasaBackend>(std::move(a), std::move(b));
    case Kind::Botpress:
        if (a.empty() || b.empty()) {
            throw std::runtime_error("botpress backend requires endpoint and conversation id");
        }
        return std::make_unique<BotpressBackend>(std::move(a), std::move(b), std::move(c));
    case Kind::DeepPavlov:
        if (a.empty()) {
            throw std::runtime_error("deeppavlov backend requires endpoint");
        }
        return std::make_unique<DeepPavlovBackend>(std::move(a));
    case Kind::H2O:
        if (a.empty() || b.empty()) {
            throw std::runtime_error("h2ogpt backend requires endpoint and model");
        }
        return std::make_unique<OpenAIBackend>(std::move(a), std::move(b), std::move(c));
    case Kind::OpenAICompat:
        if (a.empty() || b.empty()) {
            throw std::runtime_error("openai backend requires endpoint and model");
        }
        return std::make_unique<OpenAIBackend>(std::move(a), std::move(b), std::move(c));
    }
    throw std::runtime_error("unknown chat backend kind");
}

Kind parse_kind(const std::string& name) {
    std::string lowered;
    lowered.reserve(name.size());
    std::transform(name.begin(), name.end(), std::back_inserter(lowered), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });

    if (lowered == "rasa") return Kind::Rasa;
    if (lowered == "botpress") return Kind::Botpress;
    if (lowered == "deeppavlov" || lowered == "deep_pavlov") return Kind::DeepPavlov;
    if (lowered == "h2o" || lowered == "h2ogpt") return Kind::H2O;
    if (lowered == "openai" || lowered == "librechat" || lowered == "openai_compat") return Kind::OpenAICompat;
    throw std::runtime_error("unknown chat backend kind: " + name);
}

std::string kind_to_string(Kind kind) {
    switch (kind) {
    case Kind::Rasa: return "rasa";
    case Kind::Botpress: return "botpress";
    case Kind::DeepPavlov: return "deeppavlov";
    case Kind::H2O: return "h2o";
    case Kind::OpenAICompat: return "openai";
    }
    return "unknown";
}

} // namespace almondai::chat

