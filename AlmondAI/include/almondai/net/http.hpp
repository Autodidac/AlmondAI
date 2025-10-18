#pragma once

#include <string>
#include <utility>
#include <vector>

namespace almondai::net {

std::string post_json(const std::string& url,
                      const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& headers,
                      long timeout_ms = 15000);

} // namespace almondai::net

