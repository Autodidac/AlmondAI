#include "almondai/ingest.hpp"

#include <chrono>
#include <regex>
#include <unordered_set>

namespace almondai {

DataCurator::DataCurator() = default;

std::vector<LabeledExample> DataCurator::curate(const std::vector<LabeledExample> &raw) {
    std::vector<LabeledExample> curated;
    std::unordered_set<std::string> seen_hashes;
    for (const auto &example : raw) {
        if (example.prompt.size() < 4 || example.prompt.size() > 4096) {
            continue;
        }
        if (contains_pii(example.prompt) || contains_pii(example.teacher_output)) {
            continue;
        }
        if (contains_license_terms(example.teacher_output)) {
            continue;
        }
        std::string hash = hash_prompt(example.prompt);
        if (!seen_hashes.insert(hash).second) {
            continue;
        }
        LabeledExample filtered = example;
        filtered.provenance.prompt_hash = hash;
        auto now = std::chrono::system_clock::now();
        auto epoch = std::chrono::duration_cast<std::chrono::seconds>(now.time_since_epoch()).count();
        filtered.provenance.timestamp = std::to_string(epoch);
        curated.push_back(filtered);
    }
    return curated;
}

std::vector<PreferencePair> DataCurator::build_preferences(const std::vector<LabeledExample> &labeled,
                                                           const std::vector<std::string> &student_outputs) {
    std::vector<PreferencePair> prefs;
    const std::size_t count = std::min(labeled.size(), student_outputs.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (student_outputs[i].empty()) {
            continue;
        }
        PreferencePair pair;
        pair.teacher_example = labeled[i];
        pair.student_output = student_outputs[i];
        pair.preferred = pair.teacher_example.teacher_output.size() >= pair.student_output.size() ? 1 : -1;
        prefs.push_back(pair);
    }
    return prefs;
}

bool DataCurator::contains_pii(const std::string &text) const {
    std::regex pii_pattern("(\\d{4,}|@|SSN|passport|credit)", std::regex_constants::icase);
    return std::regex_search(text, pii_pattern);
}

bool DataCurator::contains_license_terms(const std::string &text) const {
    std::regex license_pattern("(GPL|LGPL|copyleft|Creative Commons)", std::regex_constants::icase);
    return std::regex_search(text, license_pattern);
}

std::string DataCurator::hash_prompt(const std::string &prompt) const {
    std::size_t h = std::hash<std::string>{}(prompt);
    return std::to_string(h);
}

} // namespace almondai
