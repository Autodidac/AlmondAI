#pragma once

#include <string>
#include <utility>
#include <vector>

namespace almondai::net {

std::string post_json(const std::string& url,
                      const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& headers,
                      long timeout_ms = -1);

} // namespace almondai::net

