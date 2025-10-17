#pragma once

#include "almondai/tokenizer_word.hpp"

#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>

namespace almondai {

struct RetrievalResult {
    std::string document_id;
    double score = 0.0;
    std::vector<int> tokens;
};

class RetrievalIndex {
public:
    explicit RetrievalIndex(const WordTokenizer& tokenizer);
    RetrievalIndex(const RetrievalIndex&) = delete;
    RetrievalIndex& operator=(const RetrievalIndex&) = delete;
    RetrievalIndex(RetrievalIndex&& other) noexcept;
    RetrievalIndex& operator=(RetrievalIndex&& other) noexcept;

    void ingest_document(const std::string& id, const std::string& text);
    std::vector<RetrievalResult> query(const std::string& text, std::size_t top_k = 3) const;

    double hit_rate() const;

private:
    const WordTokenizer& m_tokenizer;
    mutable std::mutex m_mutex;
    std::unordered_map<std::string, std::unordered_map<int, int>> m_term_counts;
    std::unordered_map<int, int> m_document_frequency;
    std::unordered_map<std::string, std::vector<int>> m_cached_tokens;
    mutable std::size_t m_query_count = 0;
    mutable std::size_t m_hit_count = 0;
};

} // namespace almondai

