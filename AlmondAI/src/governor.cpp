#include "../AlmondAI/include/almondai/governor.hpp"

#include <sstream>
#include <variant>
#include <mutex>

namespace almondai {

PolicyGovernor::PolicyGovernor() = default;

PolicyGovernor::PolicyGovernor(PolicyGovernor&& other) noexcept {
    std::scoped_lock lock(other.m_mutex);
    m_schema = other.m_schema;
    m_blocklist = std::move(other.m_blocklist);
}

PolicyGovernor& PolicyGovernor::operator=(PolicyGovernor&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_schema = other.m_schema;
        m_blocklist = std::move(other.m_blocklist);
    }
    return *this;
}

void PolicyGovernor::set_schema(Json schema) {
    std::scoped_lock lock(m_mutex);
    m_schema = std::move(schema);
}

void PolicyGovernor::set_blocklist(std::vector<std::string> tokens) {
    std::scoped_lock lock(m_mutex);
    m_blocklist.clear();
    for (auto& token : tokens) {
        m_blocklist.insert(std::move(token));
    }
}

GovernorReport PolicyGovernor::validate_output(const std::string& text, const Json& schema_override) const {
    GovernorReport report;
    Json schema;
    {
        std::scoped_lock lock(m_mutex);
        schema = schema_override.is_object() || schema_override.is_array() || schema_override.is_string()
                     ? schema_override
                     : m_schema;
        for (const auto& token : m_blocklist) {
            if (text.find(token) != std::string::npos) {
                report.allowed = false;
                report.violations.emplace_back("blocklist:" + token);
            }
        }
    }

    if (schema.is_object() && schema.as_object().count("type") != 0) {
        try {
            Json parsed = Json::parse(text);
            GovernorReport schema_report = validate_schema(schema, parsed, "$");
            if (!schema_report.allowed) {
                report.allowed = false;
                report.violations.insert(report.violations.end(), schema_report.violations.begin(), schema_report.violations.end());
            }
        } catch (const std::exception& ex) {
            report.allowed = false;
            report.violations.emplace_back(std::string("json-parse:") + ex.what());
        }
    }
    return report;
}

GovernorReport PolicyGovernor::validate_schema(const Json& schema, const Json& value, const std::string& path) const {
    GovernorReport report;
    const auto& obj = schema.as_object();
    const auto type_it = obj.find("type");
    if (type_it != obj.end() && type_it->second.is_string()) {
        const std::string& type = type_it->second.as_string();
        if (type == "object") {
            if (!value.is_object()) {
                report.allowed = false;
                report.violations.emplace_back(path + ":expected-object");
                return report;
            }
            const auto props_it = obj.find("properties");
            if (props_it != obj.end() && props_it->second.is_object()) {
                for (const auto& [prop, prop_schema] : props_it->second.as_object()) {
                    const auto& val_obj = value.as_object();
                    const auto value_it = val_obj.find(prop);
                    if (value_it == val_obj.end()) {
                        report.allowed = false;
                        report.violations.emplace_back(path + "." + prop + ":missing");
                    } else {
                        GovernorReport child = validate_schema(prop_schema, value_it->second, path + "." + prop);
                        if (!child.allowed) {
                            report.allowed = false;
                            report.violations.insert(report.violations.end(), child.violations.begin(), child.violations.end());
                        }
                    }
                }
            }
        } else if (type == "string") {
            if (!value.is_string()) {
                report.allowed = false;
                report.violations.emplace_back(path + ":expected-string");
            }
        } else if (type == "number") {
            if (!std::holds_alternative<double>(value.value())) {
                report.allowed = false;
                report.violations.emplace_back(path + ":expected-number");
            }
        } else if (type == "boolean") {
            if (!std::holds_alternative<bool>(value.value())) {
                report.allowed = false;
                report.violations.emplace_back(path + ":expected-boolean");
            }
        } else if (type == "array") {
            if (!value.is_array()) {
                report.allowed = false;
                report.violations.emplace_back(path + ":expected-array");
            }
        }
    }
    return report;
}

} // namespace almondai

