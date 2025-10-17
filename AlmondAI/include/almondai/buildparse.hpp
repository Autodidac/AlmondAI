#pragma once

#include "../AlmondAI/include/almondai/json.hpp"

#include <string>

namespace almondai {

Json parse_clang_diagnostics(const std::string& path);
Json parse_msbuild_log(const std::string& path);

}

