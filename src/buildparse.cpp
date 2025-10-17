#include "almondai/buildparse.hpp"

#include <fstream>
#include <regex>

namespace almondai {

namespace {
std::string extract_value(const std::string &content, const std::string &key, std::size_t start) {
    auto key_pos = content.find('"' + key + '"', start);
    if (key_pos == std::string::npos) {
        return {};
    }
    auto colon = content.find(':', key_pos);
    auto quote_start = content.find('"', colon + 1);
    auto quote_end = content.find('"', quote_start + 1);
    if (quote_start == std::string::npos || quote_end == std::string::npos) {
        auto value_start = content.find_first_of("0123456789", colon + 1);
        auto value_end = content.find_first_not_of("0123456789", value_start);
        if (value_start != std::string::npos) {
            return content.substr(value_start, value_end - value_start);
        }
        return {};
    }
    return content.substr(quote_start + 1, quote_end - quote_start - 1);
}

int extract_int(const std::string &value) {
    if (value.empty()) {
        return 0;
    }
    try {
        return std::stoi(value);
    } catch (...) {
        return 0;
    }
}
}

std::vector<Diagnostic> parse_clang_diagnostics_json(const std::string &path) {
    std::ifstream in(path);
    std::string content((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    std::vector<Diagnostic> diagnostics;
    std::size_t pos = 0;
    while (true) {
        auto diag_pos = content.find("\"DiagnosticName\"", pos);
        if (diag_pos == std::string::npos) {
            break;
        }
        Diagnostic diag;
        diag.code = extract_value(content, "DiagnosticName", diag_pos);
        diag.message = extract_value(content, "Message", diag_pos);
        diag.file = extract_value(content, "FilePath", diag_pos);
        diag.line = extract_int(extract_value(content, "LineNumber", diag_pos));
        diag.column = extract_int(extract_value(content, "ColumnNumber", diag_pos));
        diagnostics.push_back(diag);
        pos = diag_pos + 1;
    }
    return diagnostics;
}

std::vector<Diagnostic> parse_msbuild_log(const std::string &path) {
    std::ifstream in(path);
    std::vector<Diagnostic> diagnostics;
    std::string line;
    std::regex pattern(R"((.*)\((\d+),(\d+)\)\s*:\s*(error|warning)\s+([A-Za-z0-9]+)\s*:\s*(.*))");
    while (std::getline(in, line)) {
        std::smatch match;
        if (std::regex_search(line, match, pattern) && match.size() >= 7) {
            Diagnostic diag;
            diag.file = match[1];
            diag.line = std::stoi(match[2]);
            diag.column = std::stoi(match[3]);
            diag.code = match[5];
            diag.message = match[6];
            diagnostics.push_back(diag);
        }
    }
    return diagnostics;
}

} // namespace almondai
