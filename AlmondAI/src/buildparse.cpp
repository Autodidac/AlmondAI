#include "almondai/buildparse.hpp"

#include <fstream>
#include <sstream>

namespace almondai {

Json parse_clang_diagnostics(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return Json(JsonArray{});
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    try {
        Json root = Json::parse(buffer.str());
        JsonArray results;
        if (root.is_array()) {
            for (const auto& entry : root.as_array()) {
                if (!entry.is_object()) {
                    continue;
                }
                JsonObject obj;
                auto& diag = entry.as_object();
                auto file_it = diag.find("file");
                auto line_it = diag.find("line");
                auto column_it = diag.find("column");
                auto code_it = diag.find("diagnostic");
                auto message_it = diag.find("message");
                if (file_it != diag.end()) {
                    obj["file"] = file_it->second;
                }
                if (line_it != diag.end()) {
                    obj["line"] = line_it->second;
                }
                if (column_it != diag.end()) {
                    obj["col"] = column_it->second;
                }
                if (code_it != diag.end()) {
                    obj["code"] = code_it->second;
                }
                if (message_it != diag.end()) {
                    obj["message"] = message_it->second;
                }
                results.emplace_back(obj);
            }
        }
        return Json(results);
    } catch (...) {
        return Json(JsonArray{});
    }
}

Json parse_msbuild_log(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return Json(JsonArray{});
    }
    JsonArray results;
    std::string line;
    while (std::getline(file, line)) {
        std::size_t colon = line.find(':');
        if (colon == std::string::npos) {
            continue;
        }
        std::string location = line.substr(0, colon);
        std::string rest = line.substr(colon + 1);
        std::size_t first_paren = location.find('(');
        std::size_t second_paren = location.find(')');
        if (first_paren == std::string::npos || second_paren == std::string::npos || second_paren <= first_paren) {
            continue;
        }
        std::string file_path = location.substr(0, first_paren);
        std::string coords = location.substr(first_paren + 1, second_paren - first_paren - 1);
        std::size_t comma = coords.find(',');
        std::string line_str = coords.substr(0, comma);
        std::string col_str = comma != std::string::npos ? coords.substr(comma + 1) : "0";
        JsonObject obj;
        obj["file"] = Json(file_path);
        obj["line"] = Json(std::stod(line_str));
        obj["col"] = Json(std::stod(col_str));

        std::size_t error_pos = rest.find("error");
        std::size_t warning_pos = rest.find("warning");
        std::size_t code_start = std::string::npos;
        if (error_pos != std::string::npos) {
            code_start = error_pos;
        } else if (warning_pos != std::string::npos) {
            code_start = warning_pos;
        }
        if (code_start != std::string::npos) {
            std::size_t code_end = rest.find(':', code_start);
            obj["code"] = Json(rest.substr(code_start, code_end - code_start));
            if (code_end != std::string::npos) {
                obj["message"] = Json(rest.substr(code_end + 1));
            }
        } else {
            obj["message"] = Json(rest);
        }
        results.emplace_back(obj);
    }
    return Json(results);
}

} // namespace almondai

