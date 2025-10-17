#include "almondai/tokenizer_word.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <sstream>

namespace almondai {

namespace {
const char *kPad = "<pad>";
const char *kBos = "<bos>";
const char *kEos = "<eos>";
const char *kUnk = "<unk>";
}

TokenizerWord::TokenizerWord(TokenizerConfig cfg) : m_config(cfg) {
    m_id_to_token = {kPad, kBos, kEos, kUnk};
    for (std::size_t i = 0; i < m_id_to_token.size(); ++i) {
        m_token_to_id[m_id_to_token[i]] = static_cast<int>(i);
    }
}

void TokenizerWord::build(const std::vector<std::string> &corpus) {
    std::set<std::string> seen;
    for (const auto &entry : corpus) {
        auto words = tokenize_words(entry);
        for (auto &word : words) {
            if (m_token_to_id.count(word) == 0 && seen.insert(word).second) {
                m_token_to_id[word] = static_cast<int>(m_id_to_token.size());
                m_id_to_token.push_back(word);
            }
        }
    }
}

std::vector<int> TokenizerWord::encode(const std::string &text, bool add_special_tokens) const {
    std::vector<int> result;
    if (add_special_tokens) {
        result.push_back(m_token_to_id.at(kBos));
    }
    auto words = tokenize_words(text);
    for (const auto &word : words) {
        auto it = m_token_to_id.find(word);
        if (it != m_token_to_id.end()) {
            result.push_back(it->second);
        } else {
            result.push_back(m_token_to_id.at(kUnk));
        }
    }
    if (add_special_tokens) {
        result.push_back(m_token_to_id.at(kEos));
    }
    return result;
}

std::string TokenizerWord::decode(const std::vector<int> &tokens) const {
    std::ostringstream oss;
    bool first = true;
    for (int token : tokens) {
        if (token < 0 || static_cast<std::size_t>(token) >= m_id_to_token.size()) {
            continue;
        }
        const std::string &word = m_id_to_token[token];
        if (word == kBos || word == kEos || word == kPad) {
            continue;
        }
        if (!first) {
            oss << ' ';
        }
        oss << word;
        first = false;
    }
    return oss.str();
}

void TokenizerWord::save_vocab(const std::string &path) const {
    std::ofstream out(path);
    for (std::size_t i = 0; i < m_id_to_token.size(); ++i) {
        out << i << '\t' << m_id_to_token[i] << '\n';
    }
}

void TokenizerWord::load_vocab(const std::string &path) {
    std::ifstream in(path);
    m_token_to_id.clear();
    m_id_to_token.clear();
    std::string line;
    while (std::getline(in, line)) {
        std::istringstream iss(line);
        int id;
        std::string token;
        if (iss >> id >> token) {
            if (static_cast<std::size_t>(id) >= m_id_to_token.size()) {
                m_id_to_token.resize(id + 1);
            }
            m_id_to_token[id] = token;
            m_token_to_id[token] = id;
        }
    }
}

std::size_t TokenizerWord::vocab_size() const noexcept { return m_id_to_token.size(); }

std::string TokenizerWord::token_by_id(int id) const {
    if (id < 0 || static_cast<std::size_t>(id) >= m_id_to_token.size()) {
        return std::string{};
    }
    return m_id_to_token[id];
}

std::vector<std::string> TokenizerWord::tokenize_words(const std::string &text) const {
    std::vector<std::string> result;
    std::string current;
    auto flush = [&]() {
        if (!current.empty()) {
            result.push_back(normalize(current));
            current.clear();
        }
    };

    for (unsigned char ch : text) {
        if (std::isspace(ch) || std::ispunct(ch)) {
            flush();
        } else {
            current.push_back(static_cast<char>(ch));
        }
    }
    flush();
    return result;
}

std::string TokenizerWord::normalize(const std::string &text) const {
    std::string result = text;
    if (m_config.case_fold) {
        std::transform(result.begin(), result.end(), result.begin(), [](unsigned char c) {
            return static_cast<char>(std::tolower(c));
        });
    }
    // Placeholder for NFKC normalization. Implementing full Unicode NFKC without
    // third-party libraries is outside the scope of this reference implementation.
    // The flag is preserved so that future extensions can add a compliant pass.
    (void)m_config.normalize_nfkc;
    return result;
}

} // namespace almondai
