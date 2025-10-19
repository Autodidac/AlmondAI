#include "../include/almondai/ingest.hpp"

#include <regex>
#include <chrono>
#include <iomanip>
#include <sstream>
#include <ctime>
#include <mutex>
#include <algorithm>
#include <cctype>
#include <functional>
#include <string_view>

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

std::string collapse_whitespace(const std::string& text) {
    std::string result;
    result.reserve(text.size());
    bool in_space = false;
    for (unsigned char ch : text) {
        if (std::isspace(ch) != 0) {
            if (!result.empty() && !in_space) {
                result.push_back(' ');
            }
            in_space = true;
        } else {
            result.push_back(static_cast<char>(ch));
            in_space = false;
        }
    }
    while (!result.empty() && result.back() == ' ') {
        result.pop_back();
    }
    return result;
}

std::string canonicalise_apostrophes(std::string_view text) {
    std::string result;
    result.reserve(text.size());
    for (std::size_t i = 0; i < text.size();) {
        const unsigned char ch = static_cast<unsigned char>(text[i]);
        if (ch == 0xE2 && i + 2 < text.size()) {
            const unsigned char next1 = static_cast<unsigned char>(text[i + 1]);
            const unsigned char next2 = static_cast<unsigned char>(text[i + 2]);
            if (next1 == 0x80 && (next2 == 0x98 || next2 == 0x99)) {
                result.push_back('\'');
                i += 3;
                continue;
            }
        }
        result.push_back(static_cast<char>(ch));
        ++i;
    }
    return result;
}
}

DataCurator::DataCurator() = default;

DataCurator::DataCurator(DataCurator&& other) noexcept {
    std::scoped_lock lock(other.m_mutex);
    m_seen_samples = std::move(other.m_seen_samples);
    m_preferences = std::move(other.m_preferences);
}

DataCurator& DataCurator::operator=(DataCurator&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_seen_samples = std::move(other.m_seen_samples);
        m_preferences = std::move(other.m_preferences);
    }
    return *this;
}

std::optional<CuratedSample> DataCurator::curate(const std::string& prompt,
                                                 const std::string& teacher_output,
                                                 Json constraints,
                                                 const std::string& prompt_hash,
                                                 const std::string& teacher_source) {
    if (!within_length_band(prompt) || !within_length_band(teacher_output)) {
        return std::nullopt;
    }
    if (!passes_complexity(teacher_output) || contains_secret(teacher_output) || contains_pii(teacher_output)) {
        return std::nullopt;
    }

    const std::string source = canonical_source(teacher_source);
    const std::string sample_id = build_sample_id(prompt, teacher_output, source);
    {
        std::scoped_lock lock(m_mutex);
        if (!m_seen_samples.insert(sample_id).second) {
            return std::nullopt;
        }
    }

    CuratedSample sample;
    sample.prompt = prompt;
    sample.teacher_output = teacher_output;
    sample.constraints = std::move(constraints);
    JsonObject provenance;
    provenance["source"] = Json(source);
    provenance["prompt_hash"] = Json(prompt_hash);
    provenance["teacher_hash"] = Json(std::to_string(std::hash<std::string>{}(teacher_output)));
    provenance["sample_hash"] = Json(sample_id);
    provenance["ts"] = Json(current_timestamp());
    sample.provenance = Json(provenance);
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

void DataCurator::register_curated(CuratedSample& sample) {
    std::string source;
    if (sample.provenance.is_object()) {
        auto& prov = sample.provenance.as_object();
        if (auto it = prov.find("source"); it != prov.end() && it->second.is_string()) {
            source = it->second.as_string();
        }
        else if (auto it = prov.find("teacher_source"); it != prov.end() && it->second.is_string()) {
            source = it->second.as_string();
        }
    }

    const std::string canonical = canonical_source(source);
    const std::string sample_id = build_sample_id(sample.prompt, sample.teacher_output, canonical);

    {
        std::scoped_lock lock(m_mutex);
        m_seen_samples.insert(sample_id);
    }

    if (!sample.provenance.is_object()) {
        JsonObject prov;
        prov["source"] = Json(canonical);
        prov["sample_hash"] = Json(sample_id);
        sample.provenance = Json(prov);
        return;
    }

    auto& prov = sample.provenance.as_object();
    prov["source"] = Json(canonical);

    if (auto it = prov.find("sample_hash"); it == prov.end() || !it->second.is_string() || it->second.as_string().empty()) {
        prov["sample_hash"] = Json(sample_id);
    }
}

void DataCurator::mark_seen(const std::string& sample_id) {
    std::scoped_lock lock(m_mutex);
    m_seen_samples.insert(sample_id);
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

std::string DataCurator::canonical_source(const std::string& teacher_source) {
    std::string trimmed = teacher_source;
    trimmed.erase(trimmed.begin(), std::find_if(trimmed.begin(), trimmed.end(), [](unsigned char c) {
        return !std::isspace(c);
    }));
    trimmed.erase(std::find_if(trimmed.rbegin(), trimmed.rend(), [](unsigned char c) {
        return !std::isspace(c);
    }).base(), trimmed.end());
    if (trimmed.empty()) {
        return "gpt";
    }
    std::string lower;
    lower.reserve(trimmed.size());
    std::transform(trimmed.begin(), trimmed.end(), std::back_inserter(lower), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    return lower;
}

std::string DataCurator::normalize_for_hash(const std::string& text) {
    return canonicalise_apostrophes(collapse_whitespace(text));
}

std::string DataCurator::build_sample_id(const std::string& prompt,
                                         const std::string& teacher_output,
                                         const std::string& teacher_source) {
    std::hash<std::string> hasher;
    std::ostringstream oss;
    oss << teacher_source << "::" << std::hex << hasher(normalize_for_hash(prompt))
        << "::" << std::hex << hasher(normalize_for_hash(teacher_output));
    return oss.str();
}

} // namespace almondai

