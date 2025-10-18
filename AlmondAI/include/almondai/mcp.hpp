#pragma once

#include "json.hpp"
#include "chat/backend.hpp"

#include <string>
#include <optional>
#include <istream>
#include <ostream>

namespace almondai {

class MCPBridge {
public:
    struct Request {
        std::string id;
        std::string method;
        Json params;
    };

    std::optional<Request> read_request(std::istream& in) const;
    void send_response(std::ostream& out, const std::string& id, const Json& result) const;
    void send_error(std::ostream& out, const std::string& id, const std::string& message) const;

    Json call(const std::string& method, Json params);

    void set_chat_backend(chat::Backend* backend) noexcept { m_chat_backend = backend; }
    chat::Backend* chat_backend() const noexcept { return m_chat_backend; }

private:
    chat::Backend* m_chat_backend = nullptr;
};

} // namespace almondai
