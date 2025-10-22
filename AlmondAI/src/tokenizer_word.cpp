#include "../include/almondai/tokenizer_word.hpp"

#include <locale>
#include <cctype>
#include <sstream>
#include <mutex>
#include <unordered_set>
#include <string_view>
#include <iomanip>
#include <array>

namespace almondai {

namespace {
constexpr const char* kSpecialPad = "<pad>";
constexpr const char* kSpecialBos = "<bos>";
constexpr const char* kSpecialEos = "<eos>";
constexpr const char* kSpecialUnk = "<unk>";
void append_utf8(char32_t code, std::string& out) {
    if (code <= 0x7F) {
        out.push_back(static_cast<char>(code));
    } else if (code <= 0x7FF) {
        out.push_back(static_cast<char>(0xC0 | ((code >> 6) & 0x1F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else if (code <= 0xFFFF) {
        out.push_back(static_cast<char>(0xE0 | ((code >> 12) & 0x0F)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    } else {
        out.push_back(static_cast<char>(0xF0 | ((code >> 18) & 0x07)));
        out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
        out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
    }
}

std::string canonicalise_apostrophes(std::string_view token) {
    static constexpr std::string_view kCurlyApostrophe{"\xE2\x80\x99", 3};
    static constexpr std::string_view kCurlyOpenApostrophe{"\xE2\x80\x98", 3};

    std::string result;
    result.reserve(token.size());
    for (std::size_t i = 0; i < token.size();) {
        const unsigned char ch = static_cast<unsigned char>(token[i]);
        if (ch == 0xE2 && i + 2 < token.size()) {
            const unsigned char next1 = static_cast<unsigned char>(token[i + 1]);
            const unsigned char next2 = static_cast<unsigned char>(token[i + 2]);
            if (next1 == 0x80 && (next2 == 0x98 || next2 == 0x99)) {
                result.push_back('\'');
                i += 3;
                continue;
            }
        }
        result.push_back(static_cast<char>(ch));
        ++i;
    }
    return result;
}
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
    std::string result = canonicalise_apostrophes(token);
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

void WordTokenizer::consume_text(std::string_view text, std::unordered_set<std::string>& newly_added) {
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
            ++ptr;
            continue;
        }

        std::string token = codepoint_to_utf8(code);
        token = normalize(token);
        if (!token.empty() && m_token_to_id.find(token) == m_token_to_id.end()) {
            if (newly_added.insert(token).second) {
                int id = static_cast<int>(m_id_to_token.size());
                m_token_to_id[token] = id;
                m_id_to_token.push_back(token);
            }
        }
        ptr += length;
    }
}

std::string WordTokenizer::codepoint_to_utf8(char32_t codepoint) {
    std::string token;
    append_utf8(codepoint, token);
    return token;
}

void WordTokenizer::build_vocab(const std::vector<std::string>& documents) {
    std::scoped_lock lock(m_mutex);
    ensure_special_tokens();
    std::unordered_set<std::string> newly_added;
    for (const auto& doc : documents) {
        consume_text(doc, newly_added);
    }
}

std::size_t WordTokenizer::ingest_training_pair(std::string_view prompt, std::string_view teacher_output) {
    std::scoped_lock lock(m_mutex);
    ensure_special_tokens();
    std::unordered_set<std::string> newly_added;
    consume_text(prompt, newly_added);
    consume_text(teacher_output, newly_added);
    return newly_added.size();
}

std::vector<int> WordTokenizer::encode(const std::string& text) const {
    std::vector<int> tokens;
    tokens.reserve(text.size() / 4);
    const auto it = m_token_to_id.find(kSpecialBos);
    if (it != m_token_to_id.end()) {
        tokens.push_back(it->second);
    }
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
            ++ptr;
            continue;
        }
        std::string token = normalize(codepoint_to_utf8(code));
        auto token_it = m_token_to_id.find(token);
        if (token_it == m_token_to_id.end()) {
            tokens.push_back(m_token_to_id.at(kSpecialUnk));
        } else {
            tokens.push_back(token_it->second);
        }
        ptr += length;
    }
    tokens.push_back(m_token_to_id.at(kSpecialEos));
    return tokens;
}

std::string WordTokenizer::decode(const std::vector<int>& tokens) const {
    std::string result;
    result.reserve(tokens.size());
    for (int token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= m_id_to_token.size()) {
            continue;
        }
        const std::string& word = m_id_to_token[token];
        if (word == kSpecialBos || word == kSpecialEos || word == kSpecialPad) {
            continue;
        }
        // Tokens already carry spacing information because they represent
        // individual Unicode code points. Appending each token verbatim
        // faithfully reconstructs the input stream.
        result.append(word);
    }
    return result;
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
    for (const auto& token : m_id_to_token) {
        file << std::quoted(token) << '\n';
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
        std::istringstream iss(line);
        std::string token;
        if (line.front() == '"') {
            if (!(iss >> std::quoted(token))) {
                continue;
            }
        } else {
            token = line;
        }
        m_token_to_id[token] = index++;
        m_id_to_token.push_back(token);
    }
    ensure_special_tokens();
}

} // namespace almondai

