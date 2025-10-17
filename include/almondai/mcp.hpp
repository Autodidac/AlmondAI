#ifndef ALMONDAI_MCP_HPP
#define ALMONDAI_MCP_HPP

#include <functional>
#include <istream>
#include <map>
#include <ostream>
#include <string>

#include "almondai/buildparse.hpp"

namespace almondai {

struct MCPRequest {
    std::string id;
    std::string method;
    std::string params_json;
};

struct MCPResponse {
    std::string id;
    std::string result_json;
    std::string error_json;
};

class MCPServer {
public:
    using Handler = std::function<std::string(const std::string &)>;

    void register_handler(const std::string &method, Handler handler);
    void serve(std::istream &input, std::ostream &output);

private:
    MCPRequest parse_request(const std::string &line) const;
    std::string build_response(const MCPResponse &response) const;

    std::map<std::string, Handler> m_handlers;
};

std::string build_diagnostics_response(const std::vector<Diagnostic> &diagnostics);

} // namespace almondai

#endif // ALMONDAI_MCP_HPP
