#include "../AlmondAI/include/almondai/ingest.hpp"

#include <regex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <mutex>

namespace almondai {

namespace {
std::string current_timestamp() {
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tm{};
#ifdef _WIN32
    gmtime_s(&tm, &t);
#else
    gmtime_r(&t, &tm);
#endif
    std::ostringstream oss;
    oss << std::put_time(&tm, "%Y-%m-%dT%H:%M:%SZ");
    return oss.str();
}
}

DataCurator::DataCurator() = default;

DataCurator::DataCurator(DataCurator&& other) noexcept {
    std::scoped_lock lock(other.m_mutex);
    m_seen_prompts = std::move(other.m_seen_prompts);
    m_preferences = std::move(other.m_preferences);
}

DataCurator& DataCurator::operator=(DataCurator&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_seen_prompts = std::move(other.m_seen_prompts);
        m_preferences = std::move(other.m_preferences);
    }
    return *this;
}

std::optional<CuratedSample> DataCurator::curate(const std::string& prompt,
                                                 const std::string& teacher_output,
                                                 Json constraints,
                                                 const std::string& prompt_hash) {
    if (!within_length_band(prompt) || !within_length_band(teacher_output)) {
        return std::nullopt;
    }
    if (!passes_complexity(teacher_output) || contains_secret(teacher_output) || contains_pii(teacher_output)) {
        return std::nullopt;
    }

    std::scoped_lock lock(m_mutex);
    if (!m_seen_prompts.insert(prompt_hash).second) {
        return std::nullopt;
    }

    CuratedSample sample;
    sample.prompt = prompt;
    sample.teacher_output = teacher_output;
    sample.constraints = std::move(constraints);
    sample.provenance = JsonObject{{"source", Json("gpt")},
                                   {"prompt_hash", Json(prompt_hash)},
                                   {"ts", Json(current_timestamp())}};
    return sample;
}

void DataCurator::record_student_response(const std::string& prompt,
                                          const std::string& student_output,
                                          const CuratedSample& teacher_sample) {
    (void)prompt;
    PreferencePair pair;
    pair.positive = teacher_sample;
    pair.negative = teacher_sample;
    pair.negative.teacher_output = student_output;

    if (student_output.size() >= teacher_sample.teacher_output.size()) {
        // Student output longer than teacher; treat teacher as better still.
        pair.negative.teacher_output = student_output;
    }

    std::scoped_lock lock(m_mutex);
    m_preferences.push_back(std::move(pair));
}

std::vector<PreferencePair> DataCurator::preferences() const {
    std::scoped_lock lock(m_mutex);
    return m_preferences;
}

bool DataCurator::contains_secret(const std::string& text) {
    return text.find("BEGIN PRIVATE KEY") != std::string::npos ||
           text.find("PASSWORD=") != std::string::npos;
}

bool DataCurator::contains_pii(const std::string& text) {
    static const std::regex email(R"(([A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}))");
    static const std::regex phone(R"((\+?\d[\d\-\s]{7,}))");
    return std::regex_search(text, email) || std::regex_search(text, phone);
}

bool DataCurator::within_length_band(const std::string& text) {
    return text.size() >= 10 && text.size() <= 4000;
}

bool DataCurator::passes_complexity(const std::string& text) {
    std::size_t sentences = 0;
    for (char c : text) {
        if (c == '.' || c == '!' || c == '?') {
            ++sentences;
        }
    }
    return sentences >= 1 && text.find(' ') != std::string::npos;
}

} // namespace almondai

