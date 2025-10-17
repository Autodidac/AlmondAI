#pragma once

#include "almondai/json.hpp"

#include <string>
#include <vector>
#include <unordered_set>
#include <optional>
#include <mutex>

namespace almondai {

struct CuratedSample {
    std::string prompt;
    std::string teacher_output;
    Json constraints;
    Json provenance;
};

struct PreferencePair {
    CuratedSample positive;
    CuratedSample negative;
};

class DataCurator {
public:
    DataCurator();
    DataCurator(const DataCurator&) = delete;
    DataCurator& operator=(const DataCurator&) = delete;
    DataCurator(DataCurator&& other) noexcept;
    DataCurator& operator=(DataCurator&& other) noexcept;

    std::optional<CuratedSample> curate(const std::string& prompt,
                                        const std::string& teacher_output,
                                        Json constraints,
                                        const std::string& prompt_hash);

    void record_student_response(const std::string& prompt,
                                 const std::string& student_output,
                                 const CuratedSample& teacher_sample);

    std::vector<PreferencePair> preferences() const;

private:
    mutable std::mutex m_mutex;
    std::unordered_set<std::string> m_seen_prompts;
    std::vector<PreferencePair> m_preferences;

    static bool contains_secret(const std::string& text);
    static bool contains_pii(const std::string& text);
    static bool within_length_band(const std::string& text);
    static bool passes_complexity(const std::string& text);
};

} // namespace almondai

