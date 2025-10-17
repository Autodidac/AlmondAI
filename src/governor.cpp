#include "almondai/governor.hpp"

#include <algorithm>
#include <cctype>

namespace almondai {

PolicyGovernor::PolicyGovernor() = default;

GovernedResult PolicyGovernor::inspect(const std::string &output,
                                       const std::vector<std::string> &blocklist_tokens,
                                       const std::optional<std::string> &schema_json) const {
    GovernedResult result;
    result.output = output;
    if (violates_blocklist(output, blocklist_tokens)) {
        result.allowed = false;
        result.violations.push_back("blocklist");
    }
    if (result.allowed && schema_json && !schema_json->empty()) {
        if (!validate_schema(*schema_json, output)) {
            result.allowed = false;
            result.violations.push_back("schema");
        }
    }
    return result;
}

bool PolicyGovernor::violates_blocklist(const std::string &output,
                                        const std::vector<std::string> &blocklist) const {
    for (const auto &token : blocklist) {
        if (!token.empty()) {
            auto it = std::search(output.begin(), output.end(), token.begin(), token.end(), [](char a, char b) {
                return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
            });
            if (it != output.end()) {
                return true;
            }
        }
    }
    return false;
}

bool PolicyGovernor::validate_schema(const std::string &schema_json, const std::string &output) const {
    if (schema_json.find("object") == std::string::npos) {
        return true;
    }
    if (output.empty() || output.front() != '{' || output.back() != '}') {
        return false;
    }
    std::vector<std::string> required_fields;
    std::string needle = "\"required\"";
    auto pos = schema_json.find(needle);
    if (pos != std::string::npos) {
        auto start = schema_json.find('[', pos);
        auto end = schema_json.find(']', start);
        if (start != std::string::npos && end != std::string::npos && end > start) {
            std::string slice = schema_json.substr(start + 1, end - start - 1);
            std::string current;
            for (char ch : slice) {
                if (ch == '"') {
                    if (!current.empty()) {
                        required_fields.push_back(current);
                        current.clear();
                    } else {
                        current.clear();
                    }
                } else if (!std::isspace(static_cast<unsigned char>(ch)) && ch != ',') {
                    current.push_back(ch);
                }
            }
        }
    }
    for (const auto &field : required_fields) {
        const std::string quoted = '"' + field + '"';
        if (output.find(quoted) == std::string::npos) {
            return false;
        }
    }
    return true;
}

} // namespace almondai
