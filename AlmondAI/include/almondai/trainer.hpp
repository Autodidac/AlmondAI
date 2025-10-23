#pragma once

#include "json.hpp"
#include "model.hpp"
#include "optim_adamw.hpp"
#include "scheduler.hpp"
#include "tokenizer_bpe.hpp"

#include <deque>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace almondai {

struct TrainingExample {
    Json constraints;
    std::string prompt;
    Json provenance;
    std::string teacher_output;
};

struct TrainingReport {
    std::size_t step = 0;
    std::size_t tokens = 0;
    double loss = 0.0;
    double perplexity = 0.0;
    bool checkpoint_saved = false;
};

struct EvaluationReport {
    std::size_t tokens = 0;
    double loss = 0.0;
    double perplexity = 0.0;
    std::unordered_map<std::string, double> tag_perplexity;
    std::unordered_map<std::string, std::size_t> tag_token_counts;
    double retrieval_hit_rate = 0.0;
    double retrieval_hit_rate_delta = 0.0;
    std::vector<double> retrieval_hit_rate_history;
    double current_adapter_norm = 0.0;
    std::vector<double> adapter_norm_history;
    std::size_t recent_policy_incident_count = 0;
    std::vector<std::size_t> policy_incident_history;
};

class Trainer {
public:
    struct Options {
        std::size_t batch_size = 8;
        double label_smoothing = 0.1;
        double gradient_clip = 1.0;
        std::size_t save_every = 200;
    };

    Trainer(StudentModel& model,
            BpeTokenizer& tokenizer,
            AdamWOptimizer optimizer,
            WarmupCosineScheduler scheduler);

    void set_options(Options options) { m_options = std::move(options); }
    const Options& options() const noexcept { return m_options; }

    TrainingReport train_on_batch(const std::vector<TrainingExample>& batch);
    EvaluationReport evaluate(const std::vector<TrainingExample>& dataset) const;

    void set_checkpoint_path(std::filesystem::path path);
    void set_eval_dataset(std::vector<TrainingExample> dataset);
    const std::vector<TrainingExample>& eval_dataset() const noexcept { return m_eval_dataset; }

    void append_training_example(const TrainingExample& example);
    const std::vector<TrainingExample>& training_data() const noexcept { return m_training_data; }

    void record_retrieval_hit_rate(double hit_rate) const;
    void record_adapter_norm(double norm) const;
    void record_policy_incidents(std::size_t incidents) const;

    std::size_t step() const noexcept { return m_step; }

    StudentModel& model() noexcept { return m_model; }
    const StudentModel& model() const noexcept { return m_model; }

    bool save_checkpoint() const;

private:
    StudentModel& m_model;
    BpeTokenizer& m_tokenizer;
    AdamWOptimizer m_optimizer;
    WarmupCosineScheduler m_scheduler;
    Options m_options;
    std::size_t m_step = 0;
    std::size_t m_tokens_trained = 0;
    std::filesystem::path m_checkpoint_path;
    std::vector<TrainingExample> m_eval_dataset;
    std::vector<TrainingExample> m_training_data;
    static constexpr std::size_t kTelemetryWindow = 12;
    mutable std::deque<double> m_retrieval_hit_rate_history;
    mutable std::deque<double> m_adapter_norm_history;
    mutable std::deque<std::size_t> m_policy_incident_history;
    std::deque<double> m_recent_losses;
    std::deque<std::size_t> m_recent_throughput;
    std::size_t m_last_scheduler_retune_step = 0;

    struct BatchTensor {
        std::vector<std::vector<int>> inputs;
        std::vector<std::vector<int>> targets;
        std::vector<std::vector<double>> masks;
        std::size_t token_count = 0;
    };

    BatchTensor prepare_batch(const std::vector<TrainingExample>& batch) const;
    std::vector<double> compute_logits_gradient(const std::vector<double>& logits,
                                                int target_id,
                                                double label_smoothing,
                                                double& loss_accumulator) const;
    void maybe_retune_scheduler(std::size_t tokens, double loss);
    void log_scheduler_event(const std::string& message) const;
};

} // namespace almondai

