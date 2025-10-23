#pragma once

#include <filesystem>
#include <optional>
#include <regex>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace almondai {

class BpeTokenizer {
public:
    static constexpr int PAD_ID = 0;
    static constexpr int EOS_ID = 1;
    static constexpr int UNK_ID = 2;

    BpeTokenizer();

    bool load(const std::filesystem::path& vocab_path,
              const std::filesystem::path& merges_path = {});

    [[nodiscard]] std::vector<int> encode(std::string_view text) const;
    [[nodiscard]] std::string decode(const std::vector<int>& tokens) const;

    [[nodiscard]] std::size_t vocab_size() const noexcept { return m_id_to_token.size(); }
    [[nodiscard]] bool ready() const noexcept { return m_ready; }

    int token_to_id(std::string_view token) const;
    std::string id_to_token(int id) const;

private:
    std::filesystem::path m_vocab_path;
    bool m_ready = false;
    std::vector<std::string> m_id_to_token;
    std::unordered_map<std::string, int> m_token_to_id;
    std::unordered_map<std::string, int> m_required_token_ids;

    static bool is_whitespace(std::string_view token);
    static bool is_punctuation(std::string_view token);
    static std::vector<std::string> segment_text(std::string_view text);
    static std::vector<std::string> wordpiece_tokens(std::string_view token,
                                                     const std::unordered_map<std::string, int>& vocab,
                                                     std::vector<std::string>& new_tokens);
    void ensure_token(const std::string& token);
    std::vector<std::string> tokenize_segment(std::string_view segment) const;
};

} // namespace almondai

