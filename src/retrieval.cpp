#include "almondai/retrieval.hpp"

#include <algorithm>
#include <cmath>

#include "almondai/tokenizer_word.hpp"
#include "almondai/tensor.hpp"

namespace almondai {

namespace {
std::vector<std::string> tokens_from_ids(const std::vector<int> &ids, const TokenizerWord &tokenizer) {
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (int id : ids) {
        tokens.push_back(tokenizer.token_by_id(id));
    }
    return tokens;
}
}

RetrievalIndex::RetrievalIndex() = default;

void RetrievalIndex::add_document(const std::string &id,
                                  const std::string &text,
                                  const TokenizerWord &tokenizer) {
    auto encoded = tokenizer.encode(text, false);
    auto tokens = tokens_from_ids(encoded, tokenizer);
    std::map<std::string, double> tf;
    for (const auto &token : tokens) {
        if (token.empty()) {
            continue;
        }
        tf[token] += 1.0;
    }
    double norm = 0.0;
    for (auto &entry : tf) {
        norm += entry.second * entry.second;
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) {
        norm = 1.0;
    }
    for (auto &entry : tf) {
        entry.second /= norm;
        m_document_frequency[entry.first] += 1.0;
    }
    m_tf_idf[id] = tf;
    m_documents[id] = text;
    ++m_total_docs;
}

std::vector<RetrievalResult> RetrievalIndex::query(const std::string &text,
                                                   const TokenizerWord &tokenizer,
                                                   std::size_t top_k) {
    auto encoded = tokenizer.encode(text, false);
    auto tokens = tokens_from_ids(encoded, tokenizer);
    std::map<std::string, double> query_tf;
    for (const auto &token : tokens) {
        if (token.empty()) {
            continue;
        }
        query_tf[token] += 1.0;
    }
    double norm = 0.0;
    for (auto &entry : query_tf) {
        double idf = 0.0;
        auto df_it = m_document_frequency.find(entry.first);
        if (df_it != m_document_frequency.end() && df_it->second > 0.0) {
            idf = std::log((m_total_docs + 1.0) / (df_it->second + 1.0));
        }
        entry.second *= idf;
        norm += entry.second * entry.second;
    }
    norm = std::sqrt(norm);
    if (norm == 0.0) {
        norm = 1.0;
    }
    for (auto &entry : query_tf) {
        entry.second /= norm;
    }

    std::vector<RetrievalResult> results;
    for (auto &doc : m_tf_idf) {
        double score = 0.0;
        for (const auto &token_score : query_tf) {
            auto it = doc.second.find(token_score.first);
            if (it != doc.second.end()) {
                score += token_score.second * it->second;
            }
        }
        if (score > 0.0) {
            results.push_back({m_documents[doc.first], score});
        }
    }
    std::sort(results.begin(), results.end(), [](const RetrievalResult &a, const RetrievalResult &b) {
        return a.score > b.score;
    });
    if (results.size() > top_k) {
        results.resize(top_k);
    }
    return results;
}

void RetrievalIndex::update_embedding(const std::vector<double> &hidden_state) {
    m_last_hidden_state = hidden_state;
}

} // namespace almondai
