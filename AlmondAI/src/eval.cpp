#include "../include/almondai/eval.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

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
        auto forward = model.forward(input_tokens);
        const auto& logits = forward.logits;
        auto teacher_tokens = m_tokenizer.encode(sample.teacher_output);

        std::unordered_map<int, double> counts;
        for (int token : teacher_tokens) {
            if (token < 0) {
                continue;
            }
            const std::size_t index = static_cast<std::size_t>(token);
            if (index >= logits.size()) {
                continue;
            }
            counts[token] += 1.0;
        }
        if (counts.empty() && !logits.empty()) {
            counts[0] = 1.0;
        }

        const double total = std::accumulate(counts.begin(), counts.end(), 0.0,
                                             [](double sum, const auto& entry) {
                                                 return sum + entry.second;
                                             });
        std::vector<double> target_distribution(logits.size(), 0.0);
        for (const auto& [token, count] : counts) {
            const std::size_t index = static_cast<std::size_t>(token);
            target_distribution[index] = count / (total > 0.0 ? total : 1.0);
        }

        std::vector<double> probabilities(logits.size(), 0.0);
        double normaliser = 0.0;
        double max_logit = logits.empty() ? 0.0 : *std::max_element(logits.begin(), logits.end());
        for (std::size_t i = 0; i < logits.size(); ++i) {
            const double value = std::exp(logits[i] - max_logit);
            probabilities[i] = value;
            normaliser += value;
        }
        if (normaliser > 0.0) {
            for (double& probability : probabilities) {
                probability /= normaliser;
            }
        } else if (!probabilities.empty()) {
            const double uniform = 1.0 / static_cast<double>(probabilities.size());
            std::fill(probabilities.begin(), probabilities.end(), uniform);
        }

        constexpr double kEpsilon = 1e-12;
        double loss = 0.0;
        for (std::size_t i = 0; i < probabilities.size(); ++i) {
            if (target_distribution[i] > 0.0) {
                loss -= target_distribution[i] * std::log(std::max(probabilities[i], kEpsilon));
            }
        }
        total_loss += loss;

        std::unordered_set<int> target_tokens;
        for (const auto& [token, _] : counts) {
            (void)_; 
            target_tokens.insert(token);
        }
        auto it = std::max_element(probabilities.begin(), probabilities.end());
        if (it != probabilities.end()) {
            const int prediction = static_cast<int>(std::distance(probabilities.begin(), it));
            if (target_tokens.count(prediction)) {
                ++correct;
            }
        }
    }
    metrics.loss = total_loss / static_cast<double>(dataset.size());
    metrics.accuracy = static_cast<double>(correct) / static_cast<double>(dataset.size());
    return metrics;
}

} // namespace almondai

