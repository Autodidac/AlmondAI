#ifndef ALMONDAI_EVAL_HPP
#define ALMONDAI_EVAL_HPP

#include <string>
#include <vector>

namespace almondai {

struct EvaluationMetrics {
    double reward{0.0};
    double alignment{0.0};
};

class DecoderModel;
class TokenizerWord;
class RetrievalIndex;
class PolicyGovernor;
class AdapterRegistry;

class Evaluator {
public:
    Evaluator(DecoderModel &model,
              TokenizerWord &tokenizer,
              RetrievalIndex &retrieval,
              PolicyGovernor &governor,
              AdapterRegistry &registry);

    EvaluationMetrics evaluate(const std::vector<std::string> &prompts,
                               const std::string &adapter_name);

private:
    DecoderModel &m_model;
    TokenizerWord &m_tokenizer;
    RetrievalIndex &m_retrieval;
    PolicyGovernor &m_governor;
    AdapterRegistry &m_registry;
};

} // namespace almondai

#endif // ALMONDAI_EVAL_HPP
