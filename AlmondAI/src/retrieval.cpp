#include "almondai/retrieval.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <mutex>

namespace almondai {

RetrievalIndex::RetrievalIndex(const WordTokenizer& tokenizer) : m_tokenizer(tokenizer) {}

RetrievalIndex::RetrievalIndex(RetrievalIndex&& other) noexcept
    : m_tokenizer(other.m_tokenizer) {
    std::scoped_lock lock(other.m_mutex);
    m_term_counts = std::move(other.m_term_counts);
    m_document_frequency = std::move(other.m_document_frequency);
    m_cached_tokens = std::move(other.m_cached_tokens);
    m_query_count = other.m_query_count;
    m_hit_count = other.m_hit_count;
}

RetrievalIndex& RetrievalIndex::operator=(RetrievalIndex&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_term_counts = std::move(other.m_term_counts);
        m_document_frequency = std::move(other.m_document_frequency);
        m_cached_tokens = std::move(other.m_cached_tokens);
        m_query_count = other.m_query_count;
        m_hit_count = other.m_hit_count;
    }
    return *this;
}

void RetrievalIndex::ingest_document(const std::string& id, const std::string& text) {
    auto tokens = m_tokenizer.encode(text);
    std::unordered_map<int, int> counts;
    for (int token : tokens) {
        ++counts[token];
    }

    std::scoped_lock lock(m_mutex);
    m_cached_tokens[id] = tokens;
    m_term_counts[id] = counts;
    for (const auto& [token, count] : counts) {
        (void)count;
        ++m_document_frequency[token];
    }
}

std::vector<RetrievalResult> RetrievalIndex::query(const std::string& text, std::size_t top_k) const {
    auto query_tokens = m_tokenizer.encode(text);
    std::unordered_map<int, int> query_counts;
    for (int token : query_tokens) {
        ++query_counts[token];
    }

    std::vector<RetrievalResult> results;
    std::scoped_lock lock(m_mutex);
    ++m_query_count;
    const double doc_count = static_cast<double>(m_term_counts.size());
    for (const auto& [doc_id, counts] : m_term_counts) {
        double score = 0.0;
        for (const auto& [token, qcount] : query_counts) {
            const auto df_it = m_document_frequency.find(token);
            if (df_it == m_document_frequency.end()) {
                continue;
            }
            const double idf = std::log((doc_count + 1.0) / (df_it->second + 1.0)) + 1.0;
            const auto tf_it = counts.find(token);
            if (tf_it != counts.end()) {
                score += (static_cast<double>(tf_it->second) * idf) * static_cast<double>(qcount);
            }
        }
        if (score > 0.0) {
            results.push_back(RetrievalResult{doc_id, score, m_cached_tokens.at(doc_id)});
        }
    }

    std::sort(results.begin(), results.end(), [](const RetrievalResult& a, const RetrievalResult& b) {
        return a.score > b.score;
    });
    if (results.size() > top_k) {
        results.resize(top_k);
    }
    if (!results.empty()) {
        ++m_hit_count;
    }
    return results;
}

double RetrievalIndex::hit_rate() const {
    std::scoped_lock lock(m_mutex);
    if (m_query_count == 0) {
        return 0.0;
    }
    return static_cast<double>(m_hit_count) / static_cast<double>(m_query_count);
}

} // namespace almondai

