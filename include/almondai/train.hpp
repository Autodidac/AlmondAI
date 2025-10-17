#pragma once

#include "almondai/model.hpp"
#include "almondai/adapter.hpp"
#include "almondai/tokenizer_word.hpp"
#include "almondai/ingest.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/eval.hpp"
#include "almondai/governor.hpp"

#include <optional>
#include <fstream>

namespace almondai {

struct TrainingStats {
    std::size_t step = 0;
    double loss = 0.0;
    double accuracy = 0.0;
    double adapter_norm = 0.0;
    double retrieval_hit_rate = 0.0;
};

class ContinuousLearner {
public:
    ContinuousLearner(StudentModel student,
                      AdapterManager adapters,
                      WordTokenizer tokenizer,
                      PolicyGovernor governor);

    std::optional<CuratedSample> ingest(const std::string& prompt,
                                         const std::string& teacher_output,
                                         Json constraints,
                                         const std::string& prompt_hash);

    TrainingStats train_step(const CuratedSample& sample);
    TrainingStats evaluate_canary();

    void promote_adapter(const std::string& name);
    void rollback_adapter();

    StudentModel& student() { return m_student; }
    WordTokenizer& tokenizer() { return m_tokenizer; }
    AdapterManager& adapter_manager() { return m_adapters; }
    RetrievalIndex& retrieval() { return m_retrieval; }
    PolicyGovernor& governor() { return m_governor; }

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
    std::ofstream m_log_file;
    std::size_t m_step = 0;

    void log_stats(const TrainingStats& stats);
};

} // namespace almondai

