#include "../AlmondAI/include/almondai/mcp.hpp"

#include <sstream>

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
        JsonObject payload;
        payload["output"] = Json("teacher-response-placeholder");
        payload["provenance"] = JsonObject{{"source", Json("gpt")}};
        response["result"] = Json(payload);
    }
    return Json(response);
}

} // namespace almondai

