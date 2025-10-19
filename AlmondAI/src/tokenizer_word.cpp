#include "../include/almondai/tokenizer_word.hpp"

#include <locale>
#include <algorithm>
#include <sstream>
#include <mutex>
#include <cwctype> // Add this at the top if not already present
#include <array>
#include <unordered_set>

namespace almondai {

namespace {
constexpr const char* kSpecialPad = "<pad>";
constexpr const char* kSpecialBos = "<bos>";
constexpr const char* kSpecialEos = "<eos>";
constexpr const char* kSpecialUnk = "<unk>";
}

WordTokenizer::WordTokenizer() {
    ensure_special_tokens();
}

WordTokenizer::WordTokenizer(WordTokenizer&& other) noexcept
    : m_config(other.m_config),
      m_token_to_id(std::move(other.m_token_to_id)),
      m_id_to_token(std::move(other.m_id_to_token)) {
    ensure_special_tokens();
}

WordTokenizer& WordTokenizer::operator=(WordTokenizer&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_config = other.m_config;
        m_token_to_id = std::move(other.m_token_to_id);
        m_id_to_token = std::move(other.m_id_to_token);
        ensure_special_tokens();
    }
    return *this;
}

void WordTokenizer::set_config(TokenizerConfig config) {
    std::scoped_lock lock(m_mutex);
    m_config = config;
}

void WordTokenizer::ensure_special_tokens() {
    static constexpr std::array<const char*, 4> kSpecialTokens = {
        kSpecialPad,
        kSpecialBos,
        kSpecialEos,
        kSpecialUnk,
    };

    std::vector<std::string> rebuilt;
    rebuilt.reserve(m_id_to_token.size() + kSpecialTokens.size());
    std::unordered_set<std::string> seen;
    seen.reserve(m_id_to_token.size() + kSpecialTokens.size());

    auto add_token = [&](const std::string& token) {
        if (seen.insert(token).second) {
            rebuilt.push_back(token);
        }
    };

    for (const auto* token : kSpecialTokens) {
        add_token(token);
    }

    for (const auto& token : m_id_to_token) {
        add_token(token);
    }

    m_id_to_token = std::move(rebuilt);
    m_token_to_id.clear();
    for (std::size_t i = 0; i < m_id_to_token.size(); ++i) {
        m_token_to_id[m_id_to_token[i]] = static_cast<int>(i);
    }
}

std::string WordTokenizer::normalize(const std::string& token) const {
    if (token.empty()) {
        return token;
    }
    std::string result = token;
    if (m_config.lowercase) {
        std::locale loc;
        for (char& c : result) {
            c = static_cast<char>(std::tolower(c, loc));
        }
    }
    if (m_config.normalize_nfkc) {
        // Placeholder: full NFKC requires a full Unicode implementation.
        // We return the string unchanged to keep determinism while surfacing the option.
    }
    return result;
}

bool WordTokenizer::is_delimiter(char32_t c) {
    if (c <= 0x7F) {
        switch (static_cast<char>(c)) {
        case ' ': case '\t': case '\n': case '\r': case '\f': case '\v':
        case '.': case ',': case ';': case ':': case '!': case '?': case '\'': case '"':
        case '(': case ')': case '[': case ']': case '{': case '}': case '<': case '>':
        case '-': case '_': case '/': case '\\': case '|': case '@': case '#': case '$':
        case '%': case '^': case '&': case '*': case '+': case '=':
            return true;
        default:
            break;
        }
    }
    return ::iswspace(static_cast<wchar_t>(c)) != 0;
}

std::vector<std::string> WordTokenizer::tokenize(const std::string& text) const {
    std::u32string buffer;
    buffer.reserve(text.size());
    const char* ptr = text.data();
    const char* end = text.data() + text.size();
    while (ptr < end) {
        char32_t code = 0;
        unsigned char lead = static_cast<unsigned char>(*ptr);
        std::size_t length = 0;
        if (lead < 0x80) {
            code = lead;
            length = 1;
        } else if ((lead >> 5) == 0x6 && ptr + 1 < end) {
            code = ((lead & 0x1F) << 6) | (static_cast<unsigned char>(ptr[1]) & 0x3F);
            length = 2;
        } else if ((lead >> 4) == 0xE && ptr + 2 < end) {
            code = ((lead & 0x0F) << 12) | ((static_cast<unsigned char>(ptr[1]) & 0x3F) << 6)
                   | (static_cast<unsigned char>(ptr[2]) & 0x3F);
            length = 3;
        } else if ((lead >> 3) == 0x1E && ptr + 3 < end) {
            code = ((lead & 0x07) << 18) | ((static_cast<unsigned char>(ptr[1]) & 0x3F) << 12)
                   | ((static_cast<unsigned char>(ptr[2]) & 0x3F) << 6)
                   | (static_cast<unsigned char>(ptr[3]) & 0x3F);
            length = 4;
        } else {
            // Invalid sequence, skip byte.
            ++ptr;
            continue;
        }
        buffer.push_back(code);
        ptr += length;
    }

    std::vector<std::string> tokens;
    std::string current;
    for (char32_t code : buffer) {
        if (is_delimiter(code)) {
            if (!current.empty()) {
                tokens.push_back(normalize(current));
                current.clear();
            }
        } else {
            if (code <= 0x7F) {
                current.push_back(static_cast<char>(code));
            } else {
                if (code <= 0x7FF) {
                    current.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
                    current.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                } else if (code <= 0xFFFF) {
                    current.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
                    current.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                    current.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                } else {
                    current.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
                    current.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
                    current.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
                    current.push_back(static_cast<char>(0x80 | (code & 0x3F)));
                }
            }
        }
    }
    if (!current.empty()) {
        tokens.push_back(normalize(current));
    }
    return tokens;
}

void WordTokenizer::build_vocab(const std::vector<std::string>& documents) {
    std::scoped_lock lock(m_mutex);
    ensure_special_tokens();
    std::unordered_set<std::string> seen;
    for (const auto& doc : documents) {
        for (const auto& token : tokenize(doc)) {
            if (token.empty()) {
                continue;
            }
            if (m_token_to_id.find(token) == m_token_to_id.end()) {
                if (seen.insert(token).second) {
                    int id = static_cast<int>(m_id_to_token.size());
                    m_token_to_id[token] = id;
                    m_id_to_token.push_back(token);
                }
            }
        }
    }
}

std::vector<int> WordTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() / 4);
    const auto it = m_token_to_id.find(kSpecialBos);
    if (it != m_token_to_id.end()) {
        tokens.push_back(it->second);
    }
    for (const auto& token : tokenize(text)) {
        auto token_it = m_token_to_id.find(token);
        if (token_it == m_token_to_id.end()) {
            tokens.push_back(m_token_to_id.at(kSpecialUnk));
        } else {
            tokens.push_back(token_it->second);
        }
    }
    tokens.push_back(m_token_to_id.at(kSpecialEos));
    return tokens;
}

std::string WordTokenizer::decode(const std::vector<int>& tokens) const {
    std::ostringstream oss;
    bool first = true;
    for (int token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= m_id_to_token.size()) {
            continue;
        }
        const std::string& word = m_id_to_token[token];
        if (word == kSpecialBos || word == kSpecialEos || word == kSpecialPad) {
            continue;
        }
        if (!first) {
            oss << ' ';
        }
        first = false;
        oss << word;
    }
    return oss.str();
}

int WordTokenizer::token_id(const std::string& token) const {
    std::scoped_lock lock(m_mutex);
    auto it = m_token_to_id.find(token);
    if (it == m_token_to_id.end()) {
        return -1;
    }
    return it->second;
}

void WordTokenizer::save_vocab(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    for (std::size_t i = 0; i < m_id_to_token.size(); ++i) {
        file << m_id_to_token[i] << '\n';
    }
}

void WordTokenizer::load_vocab(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return;
    }
    std::scoped_lock lock(m_mutex);
    m_token_to_id.clear();
    m_id_to_token.clear();
    std::string line;
    int index = 0;
    while (std::getline(file, line)) {
        if (line.empty()) {
            continue;
        }
        m_token_to_id[line] = index++;
        m_id_to_token.push_back(line);
    }
    ensure_special_tokens();
}

} // namespace almondai

