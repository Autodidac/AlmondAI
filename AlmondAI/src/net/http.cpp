#include "../../include/almondai/net/http.hpp"

#include <curl/curl.h>

#include <cstdlib>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class CurlGlobal {
public:
    CurlGlobal() {
        const CURLcode code = curl_global_init(CURL_GLOBAL_DEFAULT);
        if (code != CURLE_OK) {
            throw std::runtime_error("curl_global_init failed");
        }
    }

    ~CurlGlobal() {
        curl_global_cleanup();
    }
};

size_t write_callback(char* ptr, size_t size, size_t nmemb, void* userdata) {
    const size_t total = size * nmemb;
    auto* buffer = static_cast<std::string*>(userdata);
    buffer->append(ptr, total);
    return total;
}

std::string read_environment_variable(const char* name) {
#ifdef _WIN32
    size_t required = 0;
    char* buffer = nullptr;
    if (_dupenv_s(&buffer, &required, name) != 0 || !buffer) {
        return {};
    }
    std::string value(buffer);
    std::free(buffer);
    return value;
#else
    if (const char* raw = std::getenv(name)) {
        return std::string(raw);
    }
    return {};
#endif
}

long resolve_timeout(long timeout_ms) {
    if (timeout_ms > 0) {
        return timeout_ms;
    }

    long resolved = 60000; // 60 seconds gives local servers (e.g. LM Studio) time to answer.
    const std::string raw = read_environment_variable("ALMONDAI_HTTP_TIMEOUT_MS");
    if (!raw.empty()) {
        char* end = nullptr;
        const long candidate = std::strtol(raw.c_str(), &end, 10);
        if (end != raw.c_str() && candidate > 0) {
            resolved = candidate;
        }
    }
    return resolved;
}

} // namespace

namespace almondai::net {

std::string post_json(const std::string& url,
                      const std::string& body,
                      const std::vector<std::pair<std::string, std::string>>& headers,
                      long timeout_ms) {
    CurlGlobal global_guard;

    const long resolved_timeout = resolve_timeout(timeout_ms);

    CURL* handle = curl_easy_init();
    if (!handle) {
        throw std::runtime_error("curl_easy_init failed");
    }

    std::string response;
    curl_easy_setopt(handle, CURLOPT_URL, url.c_str());
    curl_easy_setopt(handle, CURLOPT_POST, 1L);
    curl_easy_setopt(handle, CURLOPT_POSTFIELDS, body.c_str());
    curl_easy_setopt(handle, CURLOPT_POSTFIELDSIZE, static_cast<long>(body.size()));
    curl_easy_setopt(handle, CURLOPT_WRITEFUNCTION, write_callback);
    curl_easy_setopt(handle, CURLOPT_WRITEDATA, &response);
    curl_easy_setopt(handle, CURLOPT_TIMEOUT_MS, resolved_timeout);
    curl_easy_setopt(handle, CURLOPT_CONNECTTIMEOUT_MS, resolved_timeout);

    struct curl_slist* header_list = nullptr;
    header_list = curl_slist_append(header_list, "Content-Type: application/json");
    for (const auto& header : headers) {
        const std::string line = header.first + ": " + header.second;
        header_list = curl_slist_append(header_list, line.c_str());
    }
    curl_easy_setopt(handle, CURLOPT_HTTPHEADER, header_list);

    const CURLcode code = curl_easy_perform(handle);
    long status = 0;
    if (code == CURLE_OK) {
        curl_easy_getinfo(handle, CURLINFO_RESPONSE_CODE, &status);
    }

    curl_slist_free_all(header_list);
    curl_easy_cleanup(handle);

    if (code != CURLE_OK) {
        std::ostringstream oss;
        oss << "[http] POST " << url << " failed " << curl_easy_strerror(code);
        throw std::runtime_error(oss.str());
    }

    if (status < 200 || status >= 300) {
        std::ostringstream oss;
        oss << "[http] POST " << url << " failed " << status;
        throw std::runtime_error(oss.str());
    }

    return response;
}

} // namespace almondai::net

