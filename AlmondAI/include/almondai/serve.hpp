#pragma once

#include "../AlmondAI/include/almondai/train.hpp"
#include "../AlmondAI/include/almondai/mcp.hpp"
#include "../AlmondAI/include/almondai/buildparse.hpp"

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
};

} // namespace almondai

