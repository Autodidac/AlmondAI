#include "../include/almondai/tokenizer_bpe.hpp"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iterator>
#include <optional>
#include <set>
#include <system_error>

namespace almondai {

namespace {

bool is_alpha_num(char ch) {
    return static_cast<bool>(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_');
}

bool is_space(char ch) {
    return static_cast<bool>(std::isspace(static_cast<unsigned char>(ch)));
}

const std::vector<std::string> kCompoundTokens{
    "::", "->", "==", "!=", "<=", ">=", "()", "{}", "[]", "<>"
};

const std::vector<std::string> kRequiredTokens{
    "<pad>", "<eos>", "<unk>", " ", "\n", "\t",
    "::", "->", "()", "{}", "[]", "#include", "<", ">", ";",
    "template", "constexpr", "noexcept"
};

std::vector<std::string> utf8_graphemes(std::string_view text) {
    std::vector<std::string> result;
    for (std::size_t i = 0; i < text.size();) {
        unsigned char c = static_cast<unsigned char>(text[i]);
        std::size_t length = 1;
        if ((c & 0x80u) == 0) {
            length = 1;
        } else if ((c & 0xE0u) == 0xC0u) {
            length = 2;
        } else if ((c & 0xF0u) == 0xE0u) {
            length = 3;
        } else if ((c & 0xF8u) == 0xF0u) {
            length = 4;
        } else {
            length = 1;
        }
        if (i + length > text.size()) {
            length = text.size() - i;
        }
        result.emplace_back(text.substr(i, length));
        i += length;
    }
    return result;
}

} // namespace

BpeTokenizer::BpeTokenizer() {
    m_id_to_token.reserve(8192);
}

std::size_t BpeTokenizer::ingest_training_pair(std::string_view prompt, std::string_view teacher_output) {
    std::scoped_lock lock(m_mutex);
    const std::size_t before = m_id_to_token.size();
    ensure_tokens_for(prompt);
    ensure_tokens_for(teacher_output);
    return m_id_to_token.size() - before;
}

std::size_t BpeTokenizer::vocab_size() const {
    std::scoped_lock lock(m_mutex);
    return m_id_to_token.size();
}

bool BpeTokenizer::load(const std::filesystem::path& vocab_path,
                        const std::filesystem::path& merges_path) {
    std::scoped_lock lock(m_mutex);
    m_vocab_path = vocab_path;
    m_ready = false;
    m_id_to_token.clear();
    m_token_to_id.clear();
    m_recorded_merges.clear();

    std::ifstream vocab_file(vocab_path);
    if (vocab_file) {
        std::string line;
        while (std::getline(vocab_file, line)) {
            if (line.empty()) {
                continue;
            }
            auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
            line.erase(line.begin(), std::find_if(line.begin(), line.end(), not_space));
            line.erase(std::find_if(line.rbegin(), line.rend(), not_space).base(), line.end());
            if (line.empty()) {
                continue;
            }
            if (m_token_to_id.find(line) != m_token_to_id.end()) {
                continue;
            }
            const int id = static_cast<int>(m_id_to_token.size());
            m_token_to_id.emplace(line, id);
            m_id_to_token.push_back(line);
        }
    }

    if (m_id_to_token.size() < 3) {
        m_id_to_token.resize(3);
        m_id_to_token[0] = "<pad>";
        m_id_to_token[1] = "<eos>";
        m_id_to_token[2] = "<unk>";
        m_token_to_id["<pad>"] = PAD_ID;
        m_token_to_id["<eos>"] = EOS_ID;
        m_token_to_id["<unk>"] = UNK_ID;
    }

    ensure_token("<pad>", false);
    ensure_token("<eos>", false);
    ensure_token("<unk>", false);

    for (const auto& token : kRequiredTokens) {
        ensure_token(token, false);
    }

    if (!merges_path.empty()) {
        std::ifstream merges_file(merges_path);
        if (merges_file) {
            std::string merge_line;
            while (std::getline(merges_file, merge_line)) {
                if (merge_line.empty()) {
                    continue;
                }
                auto not_space = [](unsigned char ch) { return !std::isspace(ch); };
                merge_line.erase(merge_line.begin(), std::find_if(merge_line.begin(), merge_line.end(), not_space));
                merge_line.erase(std::find_if(merge_line.rbegin(), merge_line.rend(), not_space).base(), merge_line.end());
                if (merge_line.empty()) {
                    continue;
                }
                ensure_token(merge_line, false);
                if (std::find(m_recorded_merges.begin(), m_recorded_merges.end(), merge_line) == m_recorded_merges.end()) {
                    m_recorded_merges.push_back(merge_line);
                }
            }
        }
    }

    m_ready = true;
    return true;
}

std::vector<int> BpeTokenizer::encode(std::string_view text) const {
    std::scoped_lock lock(m_mutex);
    if (!m_ready) {
        return {};
    }
    std::vector<int> result;
    const auto segments = segment_text(text);
    for (const auto& segment : segments) {
        if (segment.empty()) {
            continue;
        }
        auto pieces = tokenize_segment(segment, false);
        if (pieces.empty()) {
            pieces.push_back("<unk>");
        }
        for (const auto& piece : pieces) {
            auto it = m_token_to_id.find(piece);
            if (it != m_token_to_id.end()) {
                result.push_back(it->second);
            } else {
                result.push_back(UNK_ID);
            }
        }
    }
    return result;
}

std::string BpeTokenizer::decode(const std::vector<int>& tokens) const {
    std::scoped_lock lock(m_mutex);
    if (!m_ready) {
        return {};
    }
    std::string result;
    bool last_was_punct = false;
    for (int token : tokens) {
        if (token == PAD_ID) {
            continue;
        }
        if (token < 0 || static_cast<std::size_t>(token) >= m_id_to_token.size()) {
            continue;
        }
        const std::string& piece = m_id_to_token[token];
        if (piece == "<eos>" || piece == "<pad>") {
            continue;
        }
        if (piece == "<unk>") {
            if (!result.empty() && !std::isspace(static_cast<unsigned char>(result.back()))) {
                result.push_back(' ');
            }
            result += "<unk>";
            last_was_punct = false;
            continue;
        }
        if (piece.rfind("##", 0) == 0) {
            result += piece.substr(2);
            last_was_punct = false;
            continue;
        }
        if (is_whitespace(piece)) {
            result += piece;
            last_was_punct = false;
            continue;
        }
        const bool punct = is_punctuation(piece);
        if (!result.empty() && !punct && !last_was_punct && !is_space(result.back())) {
            result.push_back(' ');
        }
        result += piece;
        last_was_punct = punct;
    }
    return result;
}

int BpeTokenizer::token_to_id(std::string_view token) const {
    std::scoped_lock lock(m_mutex);
    auto it = m_token_to_id.find(std::string(token));
    if (it == m_token_to_id.end()) {
        return UNK_ID;
    }
    return it->second;
}

std::string BpeTokenizer::id_to_token(int id) const {
    std::scoped_lock lock(m_mutex);
    if (id < 0 || static_cast<std::size_t>(id) >= m_id_to_token.size()) {
        return "<unk>";
    }
    return m_id_to_token[id];
}

bool BpeTokenizer::is_whitespace(std::string_view token) {
    return std::all_of(token.begin(), token.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    });
}

bool BpeTokenizer::is_punctuation(std::string_view token) {
    return !token.empty() &&
           std::all_of(token.begin(), token.end(), [](unsigned char ch) {
               return std::ispunct(ch) != 0 && ch != '#';
           });
}

std::vector<std::string> BpeTokenizer::segment_text(std::string_view text) {
    std::vector<std::string> segments;
    std::string current;
    enum class Mode { None, Whitespace, Word, Punct } mode = Mode::None;
    for (std::size_t i = 0; i < text.size();) {
        bool matched_compound = false;
        for (const auto& compound : kCompoundTokens) {
            if (!compound.empty() && text.size() >= i + compound.size() &&
                text.substr(i, compound.size()) == compound) {
                if (!current.empty()) {
                    segments.push_back(current);
                    current.clear();
                }
                segments.emplace_back(compound);
                i += compound.size();
                mode = Mode::None;
                matched_compound = true;
                break;
            }
        }
        if (matched_compound) {
            continue;
        }

        unsigned char ch = static_cast<unsigned char>(text[i]);
        if (is_space(ch)) {
            if (mode != Mode::Whitespace) {
                if (!current.empty()) {
                    segments.push_back(current);
                    current.clear();
                }
                mode = Mode::Whitespace;
            }
            current.push_back(static_cast<char>(ch));
            ++i;
            continue;
        }

        if (ch == '#' && i + 1 < text.size() && std::isalpha(static_cast<unsigned char>(text[i + 1])) != 0) {
            if (!current.empty()) {
                segments.push_back(current);
                current.clear();
            }
            std::size_t start = i;
            current.push_back('#');
            ++i;
            while (i < text.size() && is_alpha_num(text[i])) {
                current.push_back(text[i]);
                ++i;
            }
            segments.push_back(current);
            current.clear();
            mode = Mode::None;
            continue;
        }

        if (is_alpha_num(ch)) {
            if (mode != Mode::Word) {
                if (!current.empty()) {
                    segments.push_back(current);
                    current.clear();
                }
                mode = Mode::Word;
            }
            current.push_back(static_cast<char>(ch));
            ++i;
            continue;
        }

        if (mode != Mode::Punct) {
            if (!current.empty()) {
                segments.push_back(current);
                current.clear();
            }
            mode = Mode::Punct;
        }
        current.push_back(static_cast<char>(ch));
        ++i;
    }
    if (!current.empty()) {
        segments.push_back(current);
    }
    return segments;
}

std::vector<std::string> BpeTokenizer::wordpiece_tokens(
    std::string_view token,
    const std::unordered_map<std::string, int>& vocab,
    std::vector<std::string>& new_tokens) {
    std::vector<std::string> pieces;
    if (token.empty()) {
        return pieces;
    }
    const auto graphemes = utf8_graphemes(token);
    std::size_t start = 0;
    while (start < graphemes.size()) {
        std::size_t end = graphemes.size();
        std::optional<std::string> match;
        while (end > start) {
            std::string candidate;
            for (std::size_t i = start; i < end; ++i) {
                candidate += graphemes[i];
            }
            if (start > 0) {
                candidate.insert(0, "##");
            }
            if (vocab.find(candidate) != vocab.end()) {
                match = std::move(candidate);
                break;
            }
            --end;
        }
        if (!match) {
            // Fallback to single grapheme pieces to guarantee progress
            for (std::size_t i = start; i < graphemes.size(); ++i) {
                std::string piece = graphemes[i];
                if (i > start) {
                    piece.insert(0, "##");
                }
                new_tokens.push_back(piece);
                pieces.push_back(piece);
            }
            break;
        }
        pieces.push_back(*match);
        start = end;
    }
    return pieces;
}

void BpeTokenizer::ensure_tokens_for(std::string_view text) {
    const auto segments = segment_text(text);
    for (const auto& segment : segments) {
        tokenize_segment(segment, true);
    }
}

bool BpeTokenizer::ensure_token(const std::string& token, bool record) {
    if (m_token_to_id.find(token) != m_token_to_id.end()) {
        return false;
    }
    const int id = static_cast<int>(m_id_to_token.size());
    m_token_to_id.emplace(token, id);
    m_id_to_token.push_back(token);
    if (record) {
        if (std::find(m_recorded_merges.begin(), m_recorded_merges.end(), token) == m_recorded_merges.end()) {
            m_recorded_merges.push_back(token);
        }
    }
    return true;
}

std::vector<std::string> BpeTokenizer::tokenize_segment(std::string_view segment, bool ensure_new_tokens) const {
    if (segment.empty()) {
        return {};
    }
    if (is_whitespace(segment)) {
        if (ensure_new_tokens) {
            const_cast<BpeTokenizer*>(this)->ensure_token(std::string(segment), false);
        }
        return {std::string(segment)};
    }

    std::vector<std::string> new_tokens;
    auto pieces = wordpiece_tokens(segment, m_token_to_id, new_tokens);
    if (ensure_new_tokens) {
        for (const auto& token : new_tokens) {
            const_cast<BpeTokenizer*>(this)->ensure_token(token, true);
        }
        for (const auto& piece : pieces) {
            const_cast<BpeTokenizer*>(this)->ensure_token(piece, false);
        }
    }
    if (!pieces.empty()) {
        return pieces;
    }
    if (ensure_new_tokens) {
        const_cast<BpeTokenizer*>(this)->ensure_token("<unk>", false);
    }
    return {"<unk>"};
}

void BpeTokenizer::save_vocab(const std::filesystem::path& path) const {
    if (path.empty()) {
        return;
    }
    std::scoped_lock lock(m_mutex);
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(path);
    if (!out) {
        return;
    }
    for (const auto& token : m_id_to_token) {
        out << token << '\n';
    }
}

void BpeTokenizer::save_merges(const std::filesystem::path& path) const {
    if (path.empty()) {
        return;
    }
    std::scoped_lock lock(m_mutex);
    if (path.has_parent_path()) {
        std::error_code ec;
        std::filesystem::create_directories(path.parent_path(), ec);
    }
    std::ofstream out(path);
    if (!out) {
        return;
    }
    for (const auto& merge : m_recorded_merges) {
        out << merge << '\n';
    }
}

} // namespace almondai

