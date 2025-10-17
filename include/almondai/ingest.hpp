#ifndef ALMONDAI_INGEST_HPP
#define ALMONDAI_INGEST_HPP

#include <optional>
#include <string>
#include <vector>

namespace almondai {

struct Provenance {
    std::string source;
    std::string prompt_hash;
    std::string timestamp;
};

struct LabeledExample {
    std::string prompt;
    std::string teacher_output;
    std::string constraints;
    Provenance provenance;
};

struct PreferencePair {
    LabeledExample teacher_example;
    std::string student_output;
    int preferred; // 1 for teacher, -1 for student
};

class DataCurator {
public:
    DataCurator();

    std::vector<LabeledExample> curate(const std::vector<LabeledExample> &raw);
    std::vector<PreferencePair> build_preferences(const std::vector<LabeledExample> &labeled,
                                                  const std::vector<std::string> &student_outputs);

private:
    bool contains_pii(const std::string &text) const;
    bool contains_license_terms(const std::string &text) const;
    std::string hash_prompt(const std::string &prompt) const;
};

} // namespace almondai

#endif // ALMONDAI_INGEST_HPP
