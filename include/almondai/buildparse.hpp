#ifndef ALMONDAI_BUILDPARSE_HPP
#define ALMONDAI_BUILDPARSE_HPP

#include <string>
#include <vector>

namespace almondai {

struct Diagnostic {
    std::string file;
    int line{0};
    int column{0};
    std::string code;
    std::string message;
};

std::vector<Diagnostic> parse_clang_diagnostics_json(const std::string &path);
std::vector<Diagnostic> parse_msbuild_log(const std::string &path);

} // namespace almondai

#endif // ALMONDAI_BUILDPARSE_HPP
