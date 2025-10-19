#pragma once

#include "model.hpp"
#include "adapter.hpp"
#include "tokenizer_word.hpp"
#include "ingest.hpp"
#include "retrieval.hpp"
#include "eval.hpp"
#include "governor.hpp"
#include "json.hpp"

#include <optional>
#include <fstream>
#include <filesystem>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace almondai {

struct TrainingStats {
    std::size_t step = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    double adapter_norm = 0.0;
    double retrieval_hit_rate = 0.0;
    std::string teacher_source;
    std::vector<std::string> learning_tags;
    JsonArray learning_trace;
};

struct LoadStatus {
    std::string phase;
    std::string detail;
    std::size_t completed = 0;
    std::size_t total = 0;
};

using LoadStatusCallback = std::function<void(const LoadStatus&)>;

class ContinuousLearner {
public:
    ContinuousLearner(StudentModel student,
                      AdapterManager adapters,
                      WordTokenizer tokenizer,
                      PolicyGovernor governor,
                      LoadStatusCallback load_callback = LoadStatusCallback());

    std::optional<CuratedSample> ingest(const std::string& prompt,
                                         const std::string& teacher_output,
                                         Json constraints,
                                         const std::string& prompt_hash,
                                         const std::string& teacher_source = std::string());

    TrainingStats train_step(const CuratedSample& sample);
    TrainingStats evaluate_canary();

    void fit(const std::string& path,
             int epochs,
             int batch,
             std::function<void(int, double, double, double)> on_batch);

    void promote_adapter(const std::string& name);
    void rollback_adapter();

    StudentModel& student() { return m_student; }
    WordTokenizer& tokenizer() { return m_tokenizer; }
    AdapterManager& adapter_manager() { return m_adapters; }
    RetrievalIndex& retrieval() { return m_retrieval; }
    PolicyGovernor& governor() { return m_governor; }

    const CuratedSample* recall_sample(const std::string& document_id) const;
    std::vector<std::string> prompts_for_tags(const std::vector<std::string>& required_tags) const;
    void set_load_status_callback(LoadStatusCallback callback);

private:
    StudentModel m_student;
    AdapterManager m_adapters;
    WordTokenizer m_tokenizer;
    RetrievalIndex m_retrieval;
    Evaluator m_evaluator;
    PolicyGovernor m_governor;
    DataCurator m_curator;
    std::vector<CuratedSample> m_training_data;
    std::vector<CuratedSample> m_eval_data;
    std::unordered_map<std::string, std::size_t> m_document_to_index;
    std::ofstream m_log_file;
    std::size_t m_step = 0;
    LoadStatusCallback m_load_status_callback;

    void log_stats(const TrainingStats& stats);
    void load_persistent_data();
    void load_samples_from_file(const std::filesystem::path& path, std::size_t total_samples_hint);
    void persist_sample(const CuratedSample& sample);
    std::string derive_document_id(const CuratedSample& sample, std::size_t index) const;
    void report_load_status(std::string_view phase,
                            std::string_view detail,
                            std::size_t completed = 0,
                            std::size_t total = 0);
};

} // namespace almondai

