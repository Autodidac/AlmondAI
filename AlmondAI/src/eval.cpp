#include "../include/almondai/eval.hpp"

#include <algorithm>
#include <numeric>

namespace almondai {

Evaluator::Evaluator(const WordTokenizer& tokenizer) : m_tokenizer(tokenizer) {}

EvalMetrics Evaluator::evaluate(StudentModel& model, const std::vector<CuratedSample>& dataset) const {
    EvalMetrics metrics;
    if (dataset.empty()) {
        return metrics;
    }
    double total_loss = 0.0;
    std::size_t correct = 0;
    for (const auto& sample : dataset) {
        auto input_tokens = m_tokenizer.encode(sample.prompt);
        auto logits = model.forward(input_tokens);
        auto teacher_tokens = m_tokenizer.encode(sample.teacher_output);
        int target = teacher_tokens.size() > 1 ? teacher_tokens[1] : 0;
        auto it = std::max_element(logits.begin(), logits.end());
        int prediction = static_cast<int>(std::distance(logits.begin(), it));
        if (prediction == target) {
            ++correct;
        }
        const double target_logit = logits[target % logits.size()];
        total_loss += (*it - target_logit) * (*it - target_logit);
    }
    metrics.loss = total_loss / static_cast<double>(dataset.size());
    metrics.accuracy = static_cast<double>(correct) / static_cast<double>(dataset.size());
    return metrics;
}

} // namespace almondai

