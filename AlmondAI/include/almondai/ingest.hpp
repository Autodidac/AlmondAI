// include/almondai/ingest.hpp
#pragma once

#include "json.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <unordered_set>
#include <optional>
#include <mutex>
#include <cstdint>

namespace almondai {

    struct CuratedSample {
        std::string prompt;
        std::string teacher_output;
        Json constraints;
        Json provenance;
        std::vector<std::string> semantic_tags;
    };

    struct PreferencePair {
        CuratedSample positive; // teacher/reference
        CuratedSample negative; // student/candidate
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
            const std::string& prompt_hash,
            const std::string& teacher_source = std::string());

        void record_student_response(const std::string& prompt,
            const std::string& student_output,
            const CuratedSample& teacher_sample);

        std::vector<PreferencePair> preferences() const;

        void register_curated(CuratedSample& sample);
        void mark_seen(const std::string& sample_id);

    private:
        mutable std::mutex m_mutex;
        std::unordered_set<std::string> m_seen_samples;
        std::vector<PreferencePair> m_preferences;

        // Gates
        static bool contains_secret(const std::string& text);
        static bool contains_pii(const std::string& text);
        static bool contains_forbidden(const std::string& text);

        // Length/complexity
        static std::size_t rough_token_count(const std::string& s);
        static bool within_length_band(const std::string& text);
        static bool passes_complexity(const std::string& text);

        // Provenance + hashing
        static std::string canonical_source(const std::string& teacher_source);
        static std::string normalize_for_hash(const std::string& text);
        static std::uint64_t stable_hash64(std::string_view s);
        static std::string build_sample_id(const std::string& prompt,
            const std::string& teacher_output,
            const std::string& teacher_source);
    };

} // namespace almondai
