#pragma once

#include "json.hpp"

#include <string>
#include <vector>
#include <unordered_set>
#include <mutex>

namespace almondai {

struct GovernorReport {
    bool allowed = true;
    std::vector<std::string> violations;
};

class PolicyGovernor {
public:
    PolicyGovernor();
    PolicyGovernor(const PolicyGovernor&) = delete;
    PolicyGovernor& operator=(const PolicyGovernor&) = delete;
    PolicyGovernor(PolicyGovernor&& other) noexcept;
    PolicyGovernor& operator=(PolicyGovernor&& other) noexcept;

    void set_schema(Json schema);
    void set_blocklist(std::vector<std::string> tokens);

    GovernorReport validate_output(const std::string& text, const Json& schema_override = Json()) const;

private:
    Json m_schema;
    std::unordered_set<std::string> m_blocklist;
    mutable std::mutex m_mutex;

    GovernorReport validate_schema(const Json& schema, const Json& value, const std::string& path) const;
};

} // namespace almondai

