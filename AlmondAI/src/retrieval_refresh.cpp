#include "../include/almondai/retrieval_refresh.hpp"

#include <algorithm>
#include <unordered_set>

namespace almondai {

namespace {

std::unordered_set<int> tokens_for(const BpeTokenizer& tokenizer, const std::string& text) {
    std::unordered_set<int> set;
    auto ids = tokenizer.encode(text);
    for (int id : ids) {
        if (id <= BpeTokenizer::PAD_ID || id == BpeTokenizer::EOS_ID) {
            continue;
        }
        set.insert(id);
    }
    return set;
}

} // namespace

void refresh_retrieval_index(const std::vector<TrainingExample>& dataset, BpeTokenizer& tokenizer) {
    if (dataset.empty()) {
        return;
    }
    std::size_t hits = 0;
    for (std::size_t i = 0; i < dataset.size(); ++i) {
        const auto prompt_tokens = tokens_for(tokenizer, dataset[i].prompt);
        double best = 0.0;
        for (std::size_t j = 0; j < dataset.size(); ++j) {
            if (i == j) {
                continue;
            }
            const auto other = tokens_for(tokenizer, dataset[j].prompt);
            if (other.empty()) {
                continue;
            }
            std::size_t intersection = 0;
            for (int token : prompt_tokens) {
                if (other.find(token) != other.end()) {
                    ++intersection;
                }
            }
            std::size_t union_size = prompt_tokens.size() + other.size() - intersection;
            if (union_size == 0) {
                continue;
            }
            double score = static_cast<double>(intersection) / static_cast<double>(union_size);
            best = std::max(best, score);
        }
        if (best > 0.0) {
            ++hits;
        }
    }
    double hit_rate = static_cast<double>(hits) / static_cast<double>(dataset.size());
    (void)hit_rate;
}

} // namespace almondai

