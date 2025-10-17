#ifndef ALMONDAI_RETRIEVAL_HPP
#define ALMONDAI_RETRIEVAL_HPP

#include <map>
#include <string>
#include <vector>
#include <unordered_map>

namespace almondai {

class TokenizerWord;

struct RetrievalResult {
    std::string document;
    double score{0.0};
};

class RetrievalIndex {
public:
    RetrievalIndex();

    void add_document(const std::string &id, const std::string &text, const TokenizerWord &tokenizer);
    std::vector<RetrievalResult> query(const std::string &text, const TokenizerWord &tokenizer, std::size_t top_k = 3);
    void update_embedding(const std::vector<double> &hidden_state);

private:
    std::unordered_map<std::string, std::map<std::string, double>> m_tf_idf;
    std::unordered_map<std::string, std::string> m_documents;
    std::map<std::string, double> m_document_frequency;
    std::size_t m_total_docs{0};
    std::vector<double> m_last_hidden_state;
};

} // namespace almondai

#endif // ALMONDAI_RETRIEVAL_HPP
