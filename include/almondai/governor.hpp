#ifndef ALMONDAI_GOVERNOR_HPP
#define ALMONDAI_GOVERNOR_HPP

#include <optional>
#include <set>
#include <string>
#include <vector>

namespace almondai {

struct GovernedResult {
    bool allowed{true};
    std::string output;
    std::vector<std::string> violations;
};

class PolicyGovernor {
public:
    PolicyGovernor();

    GovernedResult inspect(const std::string &output,
                           const std::vector<std::string> &blocklist_tokens,
                           const std::optional<std::string> &schema_json) const;

private:
    bool violates_blocklist(const std::string &output, const std::vector<std::string> &blocklist) const;
    bool validate_schema(const std::string &schema_json, const std::string &output) const;
};

} // namespace almondai

#endif // ALMONDAI_GOVERNOR_HPP
