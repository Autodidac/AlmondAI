#pragma once

#include "trainer.hpp"

#include <chrono>
#include <deque>
#include <filesystem>
#include <functional>
#include <limits>
#include <optional>
#include <regex>
#include <string>
#include <string_view>

namespace almondai {

class Autopilot {
public:
    using TeacherCallback = std::function<std::optional<std::string>(const TrainingExample&)>;

    Autopilot(Trainer& trainer, BpeTokenizer& tokenizer);

    void set_teacher(TeacherCallback teacher) { m_teacher = std::move(teacher); }

    void run();

private:
    Trainer& m_trainer;
    BpeTokenizer& m_tokenizer;
    TeacherCallback m_teacher;

    std::filesystem::path m_training_path;
    std::filesystem::path m_seed_path;
    std::filesystem::path m_eval_path;
    std::filesystem::path m_weights_path;

    std::deque<std::string> m_recent_outputs;
    std::size_t m_pending_since_train = 0;
    std::size_t m_last_eval_step = 0;
    double m_best_eval_perplexity = std::numeric_limits<double>::infinity();

    void warmup_if_needed();
    void run_warmup_epochs(const std::vector<TrainingExample>& seed_data);
    std::vector<TrainingExample> load_jsonl(const std::filesystem::path& path) const;
    void append_training_record(const TrainingExample& sample);
    bool gate_sample(const TrainingExample& sample) const;
    bool violates_forbidden_regex(const std::string& text) const;
    bool contains_pii(const std::string& text) const;
    double max_similarity_against_recent(const std::string& text) const;
    void remember_output(const std::string& text);
    std::uint64_t fnv1a_hash(const std::string& text) const;

    void enqueue_sample(const TrainingExample& sample);
    void maybe_train();
    void maybe_evaluate();
    void promote_if_improved(double perplexity);
    void rebuild_retrieval_index(const std::vector<TrainingExample>& dataset) const;
    void harvest_from_seed_files();
    void log(std::string_view message) const;
};

} // namespace almondai

