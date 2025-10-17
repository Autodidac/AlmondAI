#include "almondai/eval.hpp"

#include "almondai/adapter.hpp"
#include "almondai/governor.hpp"
#include "almondai/model.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/tokenizer_word.hpp"

namespace almondai {

Evaluator::Evaluator(DecoderModel &model,
                     TokenizerWord &tokenizer,
                     RetrievalIndex &retrieval,
                     PolicyGovernor &governor,
                     AdapterRegistry &registry)
    : m_model(model),
      m_tokenizer(tokenizer),
      m_retrieval(retrieval),
      m_governor(governor),
      m_registry(registry) {}

EvaluationMetrics Evaluator::evaluate(const std::vector<std::string> &prompts,
                                      const std::string &adapter_name) {
    auto previous = m_registry.active();
    if (!adapter_name.empty()) {
        m_registry.promote(adapter_name);
    }
    EvaluationMetrics metrics;
    for (const auto &prompt : prompts) {
        GenerationOptions options;
        auto output = m_model.generate(prompt, m_tokenizer, m_retrieval, m_governor, options);
        metrics.reward += static_cast<double>(output.size());
        auto governed = m_governor.inspect(output, options.blocklist_tokens, options.schema_json);
        if (governed.allowed) {
            metrics.alignment += 1.0;
        }
    }
    if (!prompts.empty()) {
        metrics.reward /= static_cast<double>(prompts.size());
        metrics.alignment /= static_cast<double>(prompts.size());
    }
    if (previous) {
        m_registry.promote(previous->name());
    }
    return metrics;
}

} // namespace almondai
