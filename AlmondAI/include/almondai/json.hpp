#pragma once

#include <string>
#include <string_view>
#include <map>
#include <vector>
#include <variant>
#include <optional>
#include <sstream>
#include <cctype>

namespace almondai {

class Json;

using JsonObject = std::map<std::string, Json>;
using JsonArray = std::vector<Json>;

class Json {
public:
    using Value = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;

    Json() : m_value(nullptr) {}
    Json(std::nullptr_t) : m_value(nullptr) {}
    Json(bool b) : m_value(b) {}
    Json(double d) : m_value(d) {}
    Json(int i) : m_value(static_cast<double>(i)) {}
    Json(std::string s) : m_value(std::move(s)) {}
    Json(const char* s) : m_value(std::string(s)) {}
    Json(JsonArray arr) : m_value(std::move(arr)) {}
    Json(JsonObject obj) : m_value(std::move(obj)) {}

    const Value& value() const noexcept { return m_value; }
    Value& value() noexcept { return m_value; }

    bool is_object() const noexcept { return std::holds_alternative<JsonObject>(m_value); }
    bool is_array() const noexcept { return std::holds_alternative<JsonArray>(m_value); }
    bool is_string() const noexcept { return std::holds_alternative<std::string>(m_value); }

    const JsonObject& as_object() const { return std::get<JsonObject>(m_value); }
    const JsonArray& as_array() const { return std::get<JsonArray>(m_value); }
    const std::string& as_string() const { return std::get<std::string>(m_value); }

    JsonObject& as_object() { return std::get<JsonObject>(m_value); }
    JsonArray& as_array() { return std::get<JsonArray>(m_value); }
    std::string& as_string() { return std::get<std::string>(m_value); }

    std::string dump() const {
        std::ostringstream oss;
        dump_internal(oss);
        return oss.str();
    }

    static Json parse(std::string_view text);

private:
    Value m_value;

    void dump_internal(std::ostringstream& oss) const;
    static void skip_ws(std::string_view text, std::size_t& pos);
    static Json parse_value(std::string_view text, std::size_t& pos);
    static Json parse_string(std::string_view text, std::size_t& pos);
    static Json parse_array(std::string_view text, std::size_t& pos);
    static Json parse_object(std::string_view text, std::size_t& pos);
};

} // namespace almondai

