#pragma once

#include "train.hpp"
#include "mcp.hpp"
#include "buildparse.hpp"
#include "chat/backend.hpp"

#include <istream>
#include <ostream>
#include <string>

namespace almondai {

class Service {
public:
    Service(ContinuousLearner& learner, MCPBridge bridge);

    void run(std::istream& in, std::ostream& out);

    void set_chat_backend(chat::Backend* backend, std::string route_label = std::string());
    chat::Backend* chat_backend() const noexcept { return m_chat_backend; }

private:
    ContinuousLearner* m_learner;
    MCPBridge m_bridge;
    chat::Backend* m_chat_backend = nullptr;
    std::string m_chat_route;

    JsonObject handle_request(const MCPBridge::Request& request);
    void handle_trainer_fit(const MCPBridge::Request& request, std::ostream& out);
    void handle_train_self_loop(const MCPBridge::Request& request, std::ostream& out);
};

} // namespace almondai

