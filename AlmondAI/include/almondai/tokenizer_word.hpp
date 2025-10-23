#pragma once

#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <fstream>
#include <sstream>
#include <locale>
#include <mutex>
#include <string_view>

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

    // Ingests a prompt/reply training pair directly into the vocabulary.
    // Returns the number of new tokens that were added.
    std::size_t ingest_training_pair(std::string_view prompt, std::string_view teacher_output);

    std::vector<int> encode(const std::string& text) const;
    std::string decode(const std::vector<int>& tokens) const;

    int token_id(const std::string& token) const;

    void save_vocab(const std::string& path) const;
    void load_vocab(const std::string& path);

    const std::unordered_map<std::string, int>& vocab() const noexcept { return m_token_to_id; }
    std::size_t vocab_size() const;

private:
    TokenizerConfig m_config;
    std::unordered_map<std::string, int> m_token_to_id;
    std::vector<std::string> m_id_to_token;
    mutable std::mutex m_mutex;

    std::string normalize(const std::string& token) const;
    void consume_text(std::string_view text, std::unordered_set<std::string>& newly_added);
    static std::string codepoint_to_utf8(char32_t codepoint);
    void ensure_special_tokens();
};

} // namespace almondai

