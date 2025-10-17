#ifndef ALMONDAI_TRAIN_HPP
#define ALMONDAI_TRAIN_HPP

#include <string>
#include <vector>

#include "almondai/adapter.hpp"
#include "almondai/ingest.hpp"
#include "almondai/eval.hpp"

namespace almondai {

class DecoderModel;
class TokenizerWord;
class RetrievalIndex;
class PolicyGovernor;

struct TrainingMetrics {
    double loss{0.0};
    double preference{0.0};
};

struct TrainingReport {
    std::string adapter_name;
    bool promoted{false};
    EvaluationMetrics evaluation;
};

class Trainer {
public:
    Trainer(DecoderModel &model,
            AdapterRegistry &registry,
            TokenizerWord &tokenizer,
            RetrievalIndex &retrieval,
            PolicyGovernor &governor);

    TrainingReport continuous_learning_cycle(const std::vector<LabeledExample> &raw,
                                             const std::vector<std::string> &student_outputs,
                                             const std::vector<std::string> &canary_prompts,
                                             const std::string &adapter_name);

private:
    AdapterState train_adapter(const std::vector<LabeledExample> &examples,
                               const std::vector<PreferencePair> &preferences,
                               const std::string &adapter_name,
                               TrainingMetrics &metrics);
    bool canary_test(const AdapterState &adapter, const std::vector<std::string> &prompts);
    void log_metrics(const std::string &path,
                     const TrainingMetrics &metrics,
                     const EvaluationMetrics &evaluation,
                     double canary_pass_rate) const;

    DecoderModel &m_model;
    AdapterRegistry &m_registry;
    TokenizerWord &m_tokenizer;
    RetrievalIndex &m_retrieval;
    PolicyGovernor &m_governor;
    DataCurator m_curator;
};

} // namespace almondai

#endif // ALMONDAI_TRAIN_HPP
