#include "almondai/train.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>

#include "almondai/adapter.hpp"
#include "almondai/eval.hpp"
#include "almondai/model.hpp"
#include "almondai/tokenizer_word.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/governor.hpp"

namespace almondai {

Trainer::Trainer(DecoderModel &model,
                 AdapterRegistry &registry,
                 TokenizerWord &tokenizer,
                 RetrievalIndex &retrieval,
                 PolicyGovernor &governor)
    : m_model(model),
      m_registry(registry),
      m_tokenizer(tokenizer),
      m_retrieval(retrieval),
      m_governor(governor) {}

TrainingReport Trainer::continuous_learning_cycle(const std::vector<LabeledExample> &raw,
                                                  const std::vector<std::string> &student_outputs,
                                                  const std::vector<std::string> &canary_prompts,
                                                  const std::string &adapter_name) {
    TrainingReport report;
    report.adapter_name = adapter_name;

    auto curated = m_curator.curate(raw);
    auto preferences = m_curator.build_preferences(curated, student_outputs);

    TrainingMetrics metrics;
    auto trained_adapter = train_adapter(curated, preferences, adapter_name, metrics);
    m_registry.register_adapter(trained_adapter);

    Evaluator evaluator(m_model, m_tokenizer, m_retrieval, m_governor, m_registry);
    report.evaluation = evaluator.evaluate(canary_prompts, adapter_name);

    const bool canary_ok = canary_test(trained_adapter, canary_prompts);
    if (canary_ok && report.evaluation.reward > 0.0) {
        m_registry.promote(adapter_name);
        report.promoted = true;
    } else {
        m_registry.rollback();
        report.promoted = false;
    }

    log_metrics("bin/metrics.csv", metrics, report.evaluation, canary_ok ? 1.0 : 0.0);

    return report;
}

AdapterState Trainer::train_adapter(const std::vector<LabeledExample> &examples,
                                    const std::vector<PreferencePair> &preferences,
                                    const std::string &adapter_name,
                                    TrainingMetrics &metrics) {
    auto cfg = m_model.config();
    AdapterConfig adapter_cfg;
    adapter_cfg.input_dim = cfg.hidden_dim;
    adapter_cfg.output_dim = cfg.output_dim;
    adapter_cfg.rank = std::max<std::size_t>(1, cfg.hidden_dim / 4);
    AdapterState adapter(adapter_name, adapter_cfg);

    const std::size_t total = adapter_cfg.input_dim * adapter_cfg.rank + adapter_cfg.rank * adapter_cfg.output_dim;
    std::vector<double> gradient(total, 0.0);

    for (const auto &example : examples) {
        const double teacher_signal = static_cast<double>(example.teacher_output.size());
        const double prompt_signal = static_cast<double>(example.prompt.size());
        const double error = prompt_signal - teacher_signal;
        metrics.loss += error * error;
        for (double &g : gradient) {
            g += error / static_cast<double>(examples.size());
        }
    }

    for (const auto &pref : preferences) {
        metrics.preference += pref.preferred;
        const double direction = pref.preferred > 0 ? -1.0 : 1.0;
        for (double &g : gradient) {
            g += direction / static_cast<double>(preferences.size() + 1);
        }
    }

    adapter.apply_update(gradient);
    std::vector<double> prior(adapter.combined_weights().size(), 0.0);
    adapter.regularize(prior);

    if (!examples.empty()) {
        metrics.loss /= static_cast<double>(examples.size());
    }
    if (!preferences.empty()) {
        metrics.preference /= static_cast<double>(preferences.size());
    }

    return adapter;
}

bool Trainer::canary_test(const AdapterState &adapter, const std::vector<std::string> &prompts) {
    std::size_t passes = 0U;
    for (const auto &prompt : prompts) {
        if (prompt.size() < 3) {
            continue;
        }
        if (adapter.weights().empty()) {
            continue;
        }
        if (static_cast<double>(prompt.size()) / static_cast<double>(adapter.weights().size()) < 0.01) {
            ++passes;
        }
    }
    if (prompts.empty()) {
        return true;
    }
    return passes >= prompts.size() / 2;
}

void Trainer::log_metrics(const std::string &path,
                          const TrainingMetrics &metrics,
                          const EvaluationMetrics &evaluation,
                          double canary_pass_rate) const {
    std::ofstream out(path, std::ios::app);
    out << "loss,preference,reward,alignment,canary" << '\n';
    out << metrics.loss << ',' << metrics.preference << ',' << evaluation.reward << ','
        << evaluation.alignment << ',' << canary_pass_rate << '\n';
}

} // namespace almondai
