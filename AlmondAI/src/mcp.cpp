#include "../include/almondai/mcp.hpp"
#include "../include/almondai/fallback.hpp"
#include "../include/almondai/chat/backend.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <utility>
#include <variant>

namespace {

using almondai::Json;
using almondai::JsonArray;
using almondai::JsonObject;

bool is_null(const Json& value) {
    return std::holds_alternative<std::nullptr_t>(value.value());
}

std::string trim_whitespace(const std::string& text) {
    auto begin = std::find_if_not(text.begin(), text.end(), [](unsigned char ch) { return std::isspace(ch) != 0; });
    auto end = std::find_if_not(text.rbegin(), text.rend(), [](unsigned char ch) { return std::isspace(ch) != 0; }).base();
    if (begin >= end) {
        return std::string();
    }
    return std::string(begin, end);
}

JsonObject call_gpt(Json params, almondai::chat::Backend* backend) {
    std::string prompt;
    Json constraints;
    if (params.is_object()) {
        const auto& obj = params.as_object();
        if (auto it = obj.find("prompt"); it != obj.end() && it->second.is_string()) {
            prompt = it->second.as_string();
        }
        if (auto it = obj.find("constraints"); it != obj.end()) {
            constraints = it->second;
        }
    }
    if (prompt.empty()) {
        return almondai::fallback_response(prompt);
    }

    std::string augmented = prompt;
    if (!is_null(constraints)) {
        augmented += "\n\nConstraints:\n" + constraints.dump();
    }

    JsonObject fallback = almondai::fallback_response(prompt);

    if (!backend) {
        fallback["error"] = Json("chat backend unavailable");
        return fallback;
    }

    try {
        std::vector<almondai::chat::Message> conversation;
        conversation.push_back({"system", "You are AlmondAI's teacher model. Provide thorough, safe answers suitable for fine-tuning."});
        conversation.push_back({"user", std::move(augmented)});
        std::string reply = trim_whitespace(backend->complete(conversation));
        if (reply.empty()) {
            fallback["error"] = Json("chat backend returned empty response");
            return fallback;
        }

        JsonObject provenance;
        provenance["source"] = Json("chat_backend");
        provenance["status"] = Json("remote");

        JsonObject payload;
        payload["output"] = Json(reply);
        payload["provenance"] = Json(provenance);
        return payload;
    } catch (const std::exception& ex) {
        fallback["error"] = Json(ex.what());
        return fallback;
    } catch (...) {
        fallback["error"] = Json("chat backend threw unknown exception");
        return fallback;
    }
}

} // namespace

namespace almondai {

std::optional<MCPBridge::Request> MCPBridge::read_request(std::istream& in) const {
    std::string line;
    if (!std::getline(in, line)) {
        return std::nullopt;
    }
    if (line.empty()) {
        return std::nullopt;
    }
    Json parsed = Json::parse(line);
    if (!parsed.is_object()) {
        return std::nullopt;
    }
    Request request;
    const auto& obj = parsed.as_object();
    if (auto it = obj.find("id"); it != obj.end() && it->second.is_string()) {
        request.id = it->second.as_string();
    } else {
        request.id = "";
    }
    if (auto it = obj.find("method"); it != obj.end() && it->second.is_string()) {
        request.method = it->second.as_string();
    }
    if (auto it = obj.find("params"); it != obj.end()) {
        request.params = it->second;
    }
    return request;
}

void MCPBridge::send_response(std::ostream& out, const std::string& id, const Json& result) const {
    JsonObject obj;
    obj["jsonrpc"] = Json("2.0");
    obj["id"] = Json(id);
    obj["result"] = result;
    out << Json(obj).dump() << '\n';
}

void MCPBridge::send_error(std::ostream& out, const std::string& id, const std::string& message) const {
    JsonObject err;
    err["code"] = Json(-1);
    err["message"] = Json(message);
    JsonObject obj;
    obj["jsonrpc"] = Json("2.0");
    obj["id"] = Json(id);
    obj["error"] = Json(err);
    out << Json(obj).dump() << '\n';
}

Json MCPBridge::call(const std::string& method, Json params) {
    JsonObject response;
    response["method"] = Json(method);
    response["params"] = params;
    if (method == "gpt.generate") {
        response["result"] = Json(call_gpt(std::move(params), m_chat_backend));
    }
    return Json(response);
}

} // namespace almondai
