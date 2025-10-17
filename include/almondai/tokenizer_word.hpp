#ifndef ALMONDAI_TOKENIZER_WORD_HPP
#define ALMONDAI_TOKENIZER_WORD_HPP

#include <string>
#include <unordered_map>
#include <vector>
#include <set>

namespace almondai {

struct TokenizerConfig {
    bool case_fold{true};
    bool normalize_nfkc{false};
};

class TokenizerWord {
public:
    explicit TokenizerWord(TokenizerConfig cfg = {});

    void build(const std::vector<std::string> &corpus);
    std::vector<int> encode(const std::string &text, bool add_special_tokens = true) const;
    std::string decode(const std::vector<int> &tokens) const;

    void save_vocab(const std::string &path) const;
    void load_vocab(const std::string &path);

    std::size_t vocab_size() const noexcept;
    std::string token_by_id(int id) const;

private:
    std::vector<std::string> tokenize_words(const std::string &text) const;
    std::string normalize(const std::string &text) const;

    TokenizerConfig m_config;
    std::unordered_map<std::string, int> m_token_to_id;
    std::vector<std::string> m_id_to_token;
};

} // namespace almondai

#endif // ALMONDAI_TOKENIZER_WORD_HPP
