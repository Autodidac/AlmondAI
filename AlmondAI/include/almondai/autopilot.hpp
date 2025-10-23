#pragma once

#include "trainer.hpp"
#include "tokenizer_coordinator.hpp"
#include "governor.hpp"

#include <filesystem>
#include <functional>
#include <regex>
#include <string>
#include <string_view>
#include <vector>
#include <unordered_map>

#include <chrono>
#include <deque>
#include <limits>
#include <optional>

namespace almondai {

class ContinuousLearner;

class Autopilot {
public:
    using TeacherCallback = std::function<std::optional<std::string>(const TrainingExample&)>;

    struct GateDecision {
        bool accepted = false;
        std::size_t filtered_tokens = 0;
        double similarity = 0.0;
        bool contains_pii = false;
        bool regex_violation = false;
        double quality_score = 0.0;
        GovernorReport governor_report;
        std::vector<std::string> reasons;
    };

    using MutationCallback = std::function<void(const TrainingExample&, const GateDecision&)>;
    using RetrievalHook = std::function<void(const TrainingExample&, const GateDecision&)>;

    Autopilot(Trainer& trainer, TokenizerCoordinator& tokenizers);

    void set_teacher(TeacherCallback teacher) { m_teacher = std::move(teacher); }
    void set_policy_governor(const PolicyGovernor* governor) { m_policy_governor = governor; }
    void set_continuous_learner(ContinuousLearner* learner) { m_continuous_learner = learner; }
    void set_mutation_callback(MutationCallback callback) { m_mutation_callback = std::move(callback); }
    void set_retrieval_hook(RetrievalHook hook) { m_retrieval_hook = std::move(hook); }

    void run();

private:
    Trainer& m_trainer;
    TokenizerCoordinator& m_tokenizers;
    BpeTokenizer& m_tokenizer;
    TeacherCallback m_teacher;
    const PolicyGovernor* m_policy_governor = nullptr;
    ContinuousLearner* m_continuous_learner = nullptr;
    MutationCallback m_mutation_callback;
    RetrievalHook m_retrieval_hook;

    std::filesystem::path m_training_path;
    std::filesystem::path m_seed_path;
    std::filesystem::path m_eval_path;
    std::filesystem::path m_weights_path;
    std::filesystem::path m_mutation_ledger_path;
    std::filesystem::path m_telemetry_ledger_path;

    std::deque<std::string> m_recent_outputs;
    std::size_t m_pending_since_train = 0;
    std::size_t m_last_eval_step = 0;
    double m_best_eval_perplexity = std::numeric_limits<double>::infinity();
    double m_best_retrieval_hit_rate = -std::numeric_limits<double>::infinity();
    std::size_t m_lowest_policy_incidents = std::numeric_limits<std::size_t>::max();
    std::size_t m_policy_incidents_this_cycle = 0;
    bool m_policy_incidents_recorded = false;
    double m_quality_floor = 0.35;
    std::vector<std::string> m_curriculum_priority;

    void warmup_if_needed();
    void run_warmup_epochs(const std::vector<TrainingExample>& seed_data);
    std::vector<TrainingExample> load_jsonl(const std::filesystem::path& path) const;
    void append_training_record(const TrainingExample& sample);
    GateDecision gate_sample(const TrainingExample& sample) const;
    bool violates_forbidden_regex(const std::string& text) const;
    bool contains_pii(const std::string& text) const;
    double max_similarity_against_recent(const std::string& text) const;
    void remember_output(const std::string& text);
    std::uint64_t fnv1a_hash(const std::string& text) const;
    void record_mutation_decision(const TrainingExample& sample, const GateDecision& decision);
    void ingest_into_continuous_learner(const TrainingExample& sample, const GateDecision& decision);
    std::vector<TrainingExample> select_training_batch(std::size_t batch_size) const;
    void update_curriculum(const EvaluationReport& report);
    std::vector<std::string> sample_tags(const TrainingExample& sample) const;
    std::string derive_prompt_identifier(const TrainingExample& sample, const std::string& fallback) const;

    void enqueue_sample(const TrainingExample& sample);
    void maybe_train();
    void maybe_evaluate();
    void promote_if_improved(const EvaluationReport& report);
    void rebuild_retrieval_index(const std::vector<TrainingExample>& dataset) const;
    void harvest_from_seed_files();
    void log(std::string_view message) const;
    void record_promotion_rationale(const EvaluationReport& report) const;
};

} // namespace almondai

