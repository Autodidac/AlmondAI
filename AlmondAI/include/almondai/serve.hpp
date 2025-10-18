#pragma once

#include "train.hpp"
#include "mcp.hpp"
#include "buildparse.hpp"

#include <istream>
#include <ostream>

namespace almondai {

class Service {
public:
    Service(ContinuousLearner& learner, MCPBridge bridge);

    void run(std::istream& in, std::ostream& out);

private:
    ContinuousLearner* m_learner;
    MCPBridge m_bridge;

    Json handle_request(const MCPBridge::Request& request);
    void handle_trainer_fit(const MCPBridge::Request& request, std::ostream& out);
};

} // namespace almondai

