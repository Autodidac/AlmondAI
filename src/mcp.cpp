#include "almondai/mcp.hpp"

#include <sstream>

namespace almondai {

void MCPServer::register_handler(const std::string &method, Handler handler) {
    m_handlers[method] = std::move(handler);
}

void MCPServer::serve(std::istream &input, std::ostream &output) {
    std::string line;
    while (std::getline(input, line)) {
        auto request = parse_request(line);
        MCPResponse response;
        response.id = request.id;
        auto it = m_handlers.find(request.method);
        if (it == m_handlers.end()) {
            response.error_json = "{\"code\":-32601,\"message\":\"Method not found\"}";
        } else {
            response.result_json = it->second(request.params_json);
        }
        output << build_response(response) << std::endl;
    }
}

MCPRequest MCPServer::parse_request(const std::string &line) const {
    MCPRequest request;
    auto find_value = [&](const std::string &key) {
        auto pos = line.find('"' + key + '"');
        if (pos == std::string::npos) {
            return std::string{};
        }
        auto colon = line.find(':', pos);
        auto quote_start = line.find('"', colon + 1);
        auto quote_end = line.find('"', quote_start + 1);
        if (quote_start == std::string::npos || quote_end == std::string::npos) {
            return std::string{};
        }
        return line.substr(quote_start + 1, quote_end - quote_start - 1);
    };
    request.id = find_value("id");
    request.method = find_value("method");
    auto params_pos = line.find("\"params\"");
    if (params_pos != std::string::npos) {
        auto start = line.find('{', params_pos);
        auto end = line.rfind('}');
        if (start != std::string::npos && end != std::string::npos && end > start) {
            request.params_json = line.substr(start, end - start + 1);
        }
    }
    return request;
}

std::string MCPServer::build_response(const MCPResponse &response) const {
    std::ostringstream oss;
    oss << "{\"jsonrpc\":\"2.0\",\"id\":\"" << response.id << "\"";
    if (!response.error_json.empty()) {
        oss << ",\"error\":" << response.error_json;
    } else {
        oss << ",\"result\":";
        if (response.result_json.empty()) {
            oss << "{}";
        } else {
            oss << response.result_json;
        }
    }
    oss << "}";
    return oss.str();
}

std::string build_diagnostics_response(const std::vector<Diagnostic> &diagnostics) {
    std::ostringstream oss;
    oss << "{\"diagnostics\":[";
    for (std::size_t i = 0; i < diagnostics.size(); ++i) {
        const auto &diag = diagnostics[i];
        oss << "{\"file\":\"" << diag.file << "\",\"line\":" << diag.line
            << ",\"col\":" << diag.column << ",\"code\":\"" << diag.code
            << "\",\"message\":\"" << diag.message << "\"}";
        if (i + 1 < diagnostics.size()) {
            oss << ',';
        }
    }
    oss << "]}";
    return oss.str();
}

} // namespace almondai
