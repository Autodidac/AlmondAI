#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <locale>
#include <mutex>

namespace almondai {

struct TokenizerConfig {
    bool lowercase = false;
    bool normalize_nfkc = false;
};

class WordTokenizer {
public:
    WordTokenizer();
    WordTokenizer(const WordTokenizer&) = delete;
    WordTokenizer& operator=(const WordTokenizer&) = delete;
    WordTokenizer(WordTokenizer&& other) noexcept;
    WordTokenizer& operator=(WordTokenizer&& other) noexcept;

    void set_config(TokenizerConfig config);

    void build_vocab(const std::vector<std::string>& documents);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    void save_vocab(const std::string& path) const;
    void load_vocab(const std::string& path);

    const std::unordered_map<std::string, int>& vocab() const noexcept { return m_token_to_id; }

private:
    TokenizerConfig m_config;
    std::unordered_map<std::string, int> m_token_to_id;
    std::vector<std::string> m_id_to_token;
    mutable std::mutex m_mutex;

    std::string normalize(const std::string& token) const;
    std::vector<std::string> tokenize(const std::string& text) const;
    static bool is_delimiter(char32_t c);
    void ensure_special_tokens();
};

} // namespace almondai

