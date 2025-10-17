#include "../AlmondAI/include/almondai/json.hpp"

#include <stdexcept>
#include <iomanip>
#include <type_traits>

namespace almondai {

void Json::dump_internal(std::ostringstream& oss) const {
    std::visit([
                   &oss](const auto& value) {
                       using T = std::decay_t<decltype(value)>;
                       if constexpr (std::is_same_v<T, std::nullptr_t>) {
                           oss << "null";
                       } else if constexpr (std::is_same_v<T, bool>) {
                           oss << (value ? "true" : "false");
                       } else if constexpr (std::is_same_v<T, double>) {
                           oss << value;
                       } else if constexpr (std::is_same_v<T, std::string>) {
                           oss << '"';
                           for (char c : value) {
                               switch (c) {
                               case '"': oss << "\\\""; break;
                               case '\\': oss << "\\\\"; break;
                               case '\n': oss << "\\n"; break;
                               case '\r': oss << "\\r"; break;
                               case '\t': oss << "\\t"; break;
                               default:
                                   if (static_cast<unsigned char>(c) < 0x20) {
                                       oss << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                                           << static_cast<int>(static_cast<unsigned char>(c)) << std::dec
                                           << std::setfill(' ');
                                   } else {
                                       oss << c;
                                   }
                               }
                           }
                           oss << '"';
                       } else if constexpr (std::is_same_v<T, JsonArray>) {
                           oss << '[';
                           bool first = true;
                           for (const auto& item : value) {
                               if (!first) {
                                   oss << ',';
                               }
                               first = false;
                               item.dump_internal(oss);
                           }
                           oss << ']';
                       } else if constexpr (std::is_same_v<T, JsonObject>) {
                           oss << '{';
                           bool first = true;
                           for (const auto& [key, val] : value) {
                               if (!first) {
                                   oss << ',';
                               }
                               first = false;
                               oss << '"' << key << '"' << ':';
                               val.dump_internal(oss);
                           }
                           oss << '}';
                       }
                   },
               m_value);
}

void Json::skip_ws(std::string_view text, std::size_t& pos) {
    while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos]))) {
        ++pos;
    }
}

Json Json::parse(std::string_view text) {
    std::size_t pos = 0;
    skip_ws(text, pos);
    Json value = parse_value(text, pos);
    skip_ws(text, pos);
    if (pos != text.size()) {
        throw std::runtime_error("unexpected trailing characters in JSON");
    }
    return value;
}

Json Json::parse_value(std::string_view text, std::size_t& pos) {
    skip_ws(text, pos);
    if (pos >= text.size()) {
        throw std::runtime_error("unexpected end of JSON");
    }
    const char c = text[pos];
    if (c == '"') {
        return parse_string(text, pos);
    }
    if (c == '[') {
        return parse_array(text, pos);
    }
    if (c == '{') {
        return parse_object(text, pos);
    }
    if (std::isdigit(static_cast<unsigned char>(c)) || c == '-') {
        std::size_t start = pos;
        ++pos;
        while (pos < text.size() && (std::isdigit(static_cast<unsigned char>(text[pos])) || text[pos] == '.' || text[pos] == 'e' || text[pos] == 'E' || text[pos] == '+' || text[pos] == '-')) {
            ++pos;
        }
        return Json(std::stod(std::string(text.substr(start, pos - start))));
    }
    if (text.substr(pos, 4) == "true") {
        pos += 4;
        return Json(true);
    }
    if (text.substr(pos, 5) == "false") {
        pos += 5;
        return Json(false);
    }
    if (text.substr(pos, 4) == "null") {
        pos += 4;
        return Json(nullptr);
    }
    throw std::runtime_error("invalid JSON token");
}

Json Json::parse_string(std::string_view text, std::size_t& pos) {
    if (text[pos] != '"') {
        throw std::runtime_error("expected string");
    }
    ++pos;
    std::string result;
    while (pos < text.size()) {
        char c = text[pos++];
        if (c == '"') {
            break;
        }
        if (c == '\\') {
            if (pos >= text.size()) {
                throw std::runtime_error("invalid escape");
            }
            char esc = text[pos++];
            switch (esc) {
            case '"': result.push_back('"'); break;
            case '\\': result.push_back('\\'); break;
            case '/': result.push_back('/'); break;
            case 'b': result.push_back('\b'); break;
            case 'f': result.push_back('\f'); break;
            case 'n': result.push_back('\n'); break;
            case 'r': result.push_back('\r'); break;
            case 't': result.push_back('\t'); break;
            case 'u': {
                if (pos + 4 > text.size()) {
                    throw std::runtime_error("invalid unicode escape");
                }
                // Minimal handling: read hex and keep as UTF-8 by encoding as bytes if <= 0x7F.
                unsigned code = 0;
                for (int i = 0; i < 4; ++i) {
                    char h = text[pos++];
                    code <<= 4;
                    if (h >= '0' && h <= '9') {
                        code |= static_cast<unsigned>(h - '0');
                    } else if (h >= 'a' && h <= 'f') {
                        code |= static_cast<unsigned>(h - 'a' + 10);
                    } else if (h >= 'A' && h <= 'F') {
                        code |= static_cast<unsigned>(h - 'A' + 10);
                    } else {
                        throw std::runtime_error("invalid unicode escape");
                    }
                }
                if (code <= 0x7F) {
                    result.push_back(static_cast<char>(code));
                } else {
                    // Basic UTF-8 encoding for BMP subset.
                    if (code <= 0x7FF) {
                        result.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                        result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    } else {
                        result.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                        result.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                        result.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                    }
                }
                break;
            }
            default:
                throw std::runtime_error("invalid escape");
            }
        } else {
            result.push_back(c);
        }
    }
    return Json(result);
}

Json Json::parse_array(std::string_view text, std::size_t& pos) {
    if (text[pos] != '[') {
        throw std::runtime_error("expected array");
    }
    ++pos;
    JsonArray arr;
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == ']') {
        ++pos;
        return Json(arr);
    }
    while (pos < text.size()) {
        arr.emplace_back(parse_value(text, pos));
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == ']') {
            ++pos;
            break;
        }
        throw std::runtime_error("expected comma or closing bracket");
    }
    return Json(arr);
}

Json Json::parse_object(std::string_view text, std::size_t& pos) {
    if (text[pos] != '{') {
        throw std::runtime_error("expected object");
    }
    ++pos;
    JsonObject obj;
    skip_ws(text, pos);
    if (pos < text.size() && text[pos] == '}') {
        ++pos;
        return Json(obj);
    }
    while (pos < text.size()) {
        skip_ws(text, pos);
        Json key = parse_string(text, pos);
        skip_ws(text, pos);
        if (pos >= text.size() || text[pos] != ':') {
            throw std::runtime_error("expected colon");
        }
        ++pos;
        obj.emplace(key.as_string(), parse_value(text, pos));
        skip_ws(text, pos);
        if (pos < text.size() && text[pos] == ',') {
            ++pos;
            continue;
        }
        if (pos < text.size() && text[pos] == '}') {
            ++pos;
            break;
        }
        throw std::runtime_error("expected comma or closing brace");
    }
    return Json(obj);
}

} // namespace almondai

