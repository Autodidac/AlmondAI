#include "../AlmondAI/include/almondai/mcp.hpp"

#include <sstream>
#include <cstdlib>
#include <memory>
#include <array>
#include <algorithm>
#include <variant>
#include <utility>

#ifdef _WIN32
#include <stdio.h>
#define popen _popen
#define pclose _pclose
#else
#include <stdio.h>
#endif

namespace {

using almondai::Json;
using almondai::JsonArray;
using almondai::JsonObject;

std::string escape_single_quotes(const std::string& input) {
    std::string escaped;
    escaped.reserve(input.size() + 8);
    for (char c : input) {
        if (c == '\'') {
            escaped += "'\\''";
        } else {
            escaped.push_back(c);
        }
    }
    return escaped;
}

std::string run_command(const std::string& command) {
    std::array<char, 512> buffer{};
    std::string output;
    std::unique_ptr<FILE, decltype(&pclose)> pipe(popen(command.c_str(), "r"), pclose);
    if (!pipe) {
        return output;
    }
    while (fgets(buffer.data(), static_cast<int>(buffer.size()), pipe.get()) != nullptr) {
        output.append(buffer.data());
    }
    return output;
}

bool is_null(const Json& value) {
    return std::holds_alternative<std::nullptr_t>(value.value());
}

std::string extract_openai_text(const Json& response) {
    if (!response.is_object()) {
        return std::string{};
    }
    const auto& obj = response.as_object();
    if (auto it = obj.find("choices"); it != obj.end() && it->second.is_array()) {
        const auto& choices = it->second.as_array();
        if (!choices.empty() && choices.front().is_object()) {
            const auto& choice = choices.front().as_object();
            if (auto msg_it = choice.find("message"); msg_it != choice.end() && msg_it->second.is_object()) {
                const auto& message = msg_it->second.as_object();
                if (auto content_it = message.find("content"); content_it != message.end()) {
                    if (content_it->second.is_string()) {
                        return content_it->second.as_string();
                    }
                    if (content_it->second.is_array()) {
                        std::string text;
                        for (const auto& part : content_it->second.as_array()) {
                            if (part.is_object()) {
                                const auto& part_obj = part.as_object();
                                if (auto text_it = part_obj.find("text"); text_it != part_obj.end() && text_it->second.is_string()) {
                                    text += text_it->second.as_string();
                                }
                            }
                        }
                        if (!text.empty()) {
                            return text;
                        }
                    }
                }
            }
            if (auto text_it = choice.find("text"); text_it != choice.end() && text_it->second.is_string()) {
                return text_it->second.as_string();
            }
        }
    }
    if (auto it = obj.find("output_text"); it != obj.end() && it->second.is_array()) {
        std::string combined;
        for (const auto& item : it->second.as_array()) {
            if (item.is_string()) {
                combined += item.as_string();
            }
        }
        if (!combined.empty()) {
            return combined;
        }
    }
    if (auto it = obj.find("content"); it != obj.end() && it->second.is_string()) {
        return it->second.as_string();
    }
    return std::string{};
}

JsonObject fallback_response(const std::string& prompt) {
    JsonObject provenance;
    provenance["source"] = Json("gpt");
    provenance["status"] = Json("placeholder");
    JsonObject payload;
    payload["output"] = Json("Teacher model unavailable. Please provide teacher_output manually.");
    payload["provenance"] = Json(provenance);
    payload["prompt"] = Json(prompt);
    return payload;
}

std::string safe_getenv(const char* name) {
    char* buf = nullptr;
    size_t len = 0;
    if (_dupenv_s(&buf, &len, name) == 0 && buf) {
        std::string val(buf);
        free(buf);
        return val;
    }
    return {};
}

JsonObject call_gpt(Json params) {
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
        return fallback_response(prompt);
    }

    std::string api_key = safe_getenv("ALMONDAI_GPT_API_KEY");
    if (api_key.empty()) {
        return fallback_response(prompt);
    }

    std::string endpoint = safe_getenv("ALMONDAI_GPT_ENDPOINT");
    if (endpoint.empty()) {
        endpoint = "https://api.openai.com/v1/chat/completions";
    }

    std::string model = safe_getenv("ALMONDAI_GPT_MODEL");
    if (model.empty()) {
        model = "gpt-4o-mini";
    }

    std::string augmented_prompt = prompt;
    if (!is_null(constraints)) {
        augmented_prompt += "\n\nConstraints:\n" + constraints.dump();
    }

    JsonObject body;
    body["model"] = Json(model);
    body["temperature"] = Json(0.2);
    JsonArray messages;
    JsonObject system_msg;
    system_msg["role"] = Json("system");
    system_msg["content"] = Json("You are AlmondAI's teacher model. Provide thorough, safe answers suitable for fine-tuning.");
    messages.emplace_back(Json(system_msg));
    JsonObject user_msg;
    user_msg["role"] = Json("user");
    user_msg["content"] = Json(augmented_prompt);
    messages.emplace_back(Json(user_msg));
    body["messages"] = Json(messages);

    std::string body_str = Json(body).dump();
    std::ostringstream command;
    command << "curl -s -X POST '" << endpoint << "'"
            << " -H 'Content-Type: application/json'"
            << " -H 'Authorization: Bearer " << api_key << "'"
            << " -d '" << escape_single_quotes(body_str) << "'";

    std::string raw_response = run_command(command.str());
    if (raw_response.empty()) {
        return fallback_response(prompt);
    }

    try {
        Json parsed = Json::parse(raw_response);
        std::string output = extract_openai_text(parsed);
        if (output.empty()) {
            return fallback_response(prompt);
        }
        JsonObject provenance;
        provenance["source"] = Json("gpt");
        provenance["model"] = Json(model);
        JsonObject payload;
        payload["output"] = Json(output);
        payload["provenance"] = Json(provenance);
        return payload;
    } catch (...) {
        return fallback_response(prompt);
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
        response["result"] = Json(call_gpt(std::move(params)));
    }
    return Json(response);
}

} // namespace almondai

