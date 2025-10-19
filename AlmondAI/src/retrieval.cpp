#include "../include/almondai/retrieval.hpp"
#include "../include/almondai/json.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <mutex>
#include <filesystem>
#include <fstream>
#include <set>
#include <cctype>
#include <optional>
#include <type_traits>

namespace almondai {

RetrievalIndex::RetrievalIndex(const WordTokenizer& tokenizer) : m_tokenizer(tokenizer) {}

RetrievalIndex::RetrievalIndex(RetrievalIndex&& other) noexcept
    : m_tokenizer(other.m_tokenizer) {
    std::scoped_lock lock(other.m_mutex);
    m_term_counts = std::move(other.m_term_counts);
    m_document_frequency = std::move(other.m_document_frequency);
    m_cached_tokens = std::move(other.m_cached_tokens);
    m_query_count = other.m_query_count;
    m_hit_count = other.m_hit_count;
}

RetrievalIndex& RetrievalIndex::operator=(RetrievalIndex&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_term_counts = std::move(other.m_term_counts);
        m_document_frequency = std::move(other.m_document_frequency);
        m_cached_tokens = std::move(other.m_cached_tokens);
        m_query_count = other.m_query_count;
        m_hit_count = other.m_hit_count;
    }
    return *this;
}

namespace {

std::vector<std::string> normalise_tags(const std::vector<std::string>& tags) {
    std::vector<std::string> cleaned;
    cleaned.reserve(tags.size());
    std::unordered_set<std::string> seen;
    for (const auto& tag : tags) {
        std::string trimmed;
        trimmed.reserve(tag.size());
        bool in_space = false;
        for (unsigned char ch : tag) {
            if (std::isspace(ch) != 0) {
                if (!trimmed.empty()) {
                    in_space = true;
                }
                continue;
            }
            if (in_space) {
                trimmed.push_back(' ');
                in_space = false;
            }
            trimmed.push_back(static_cast<char>(ch));
        }
        while (!trimmed.empty() && std::isspace(static_cast<unsigned char>(trimmed.back())) != 0) {
            trimmed.pop_back();
        }
        if (trimmed.empty()) {
            continue;
        }
        if (seen.insert(trimmed).second) {
            cleaned.push_back(std::move(trimmed));
        }
    }
    return cleaned;
}

} // namespace

void RetrievalIndex::ingest_document(const std::string& id,
                                     const std::string& text,
                                     const std::vector<std::string>& tags) {
    auto tokens = m_tokenizer.encode(text);
    std::unordered_map<int, int> counts;
    for (int token : tokens) {
        ++counts[token];
    }

    std::scoped_lock lock(m_mutex);
    if (auto existing = m_term_counts.find(id); existing != m_term_counts.end()) {
        for (const auto& [token, _] : existing->second) {
            auto freq_it = m_document_frequency.find(token);
            if (freq_it != m_document_frequency.end()) {
                if (freq_it->second > 1) {
                    --freq_it->second;
                } else {
                    m_document_frequency.erase(freq_it);
                }
            }
        }
    }
    m_cached_tokens[id] = tokens;
    m_term_counts[id] = counts;
    for (const auto& [token, count] : counts) {
        (void)count;
        ++m_document_frequency[token];
    }
    m_document_tags[id] = normalise_tags(tags);
}

std::vector<RetrievalResult> RetrievalIndex::query(const std::string& text, std::size_t top_k) const {
    auto query_tokens = m_tokenizer.encode(text);
    std::unordered_map<int, int> query_counts;
    for (int token : query_tokens) {
        ++query_counts[token];
    }

    std::vector<RetrievalResult> results;
    std::scoped_lock lock(m_mutex);
    ++m_query_count;
    const double doc_count = static_cast<double>(m_term_counts.size());
    for (const auto& [doc_id, counts] : m_term_counts) {
        double score = 0.0;
        for (const auto& [token, qcount] : query_counts) {
            const auto df_it = m_document_frequency.find(token);
            if (df_it == m_document_frequency.end()) {
                continue;
            }
            const double idf = std::log((doc_count + 1.0) / (df_it->second + 1.0)) + 1.0;
            const auto tf_it = counts.find(token);
            if (tf_it != counts.end()) {
                score += (static_cast<double>(tf_it->second) * idf) * static_cast<double>(qcount);
            }
        }
        if (score > 0.0) {
            RetrievalResult result;
            result.document_id = doc_id;
            result.score = score;
            result.tokens = m_cached_tokens.at(doc_id);
            if (auto tag_it = m_document_tags.find(doc_id); tag_it != m_document_tags.end()) {
                result.tags = tag_it->second;
            }
            results.push_back(std::move(result));
        }
    }

    std::sort(results.begin(), results.end(), [](const RetrievalResult& a, const RetrievalResult& b) {
        return a.score > b.score;
    });
    if (results.size() > top_k) {
        results.resize(top_k);
    }
    if (!results.empty()) {
        ++m_hit_count;
    }
    return results;
}

double RetrievalIndex::hit_rate() const {
    std::scoped_lock lock(m_mutex);
    if (m_query_count == 0) {
        return 0.0;
    }
    return static_cast<double>(m_hit_count) / static_cast<double>(m_query_count);
}

void RetrievalIndex::save_metadata(const std::filesystem::path& path) const {
    namespace fs = std::filesystem;
    std::scoped_lock lock(m_mutex);

    JsonObject root;
    JsonObject stats;
    stats["query_count"] = Json(static_cast<double>(m_query_count));
    stats["hit_count"] = Json(static_cast<double>(m_hit_count));
    root["stats"] = Json(stats);

    JsonArray documents;
    documents.reserve(m_term_counts.size());
    for (const auto& [doc_id, counts] : m_term_counts) {
        JsonObject entry;
        entry["id"] = Json(doc_id);
        if (auto tag_it = m_document_tags.find(doc_id); tag_it != m_document_tags.end() && !tag_it->second.empty()) {
            JsonArray tags_json;
            tags_json.reserve(tag_it->second.size());
            for (const auto& tag : tag_it->second) {
                tags_json.emplace_back(Json(tag));
            }
            entry["tags"] = Json(tags_json);
        }
        if (auto token_it = m_cached_tokens.find(doc_id); token_it != m_cached_tokens.end()) {
            JsonArray tokens_json;
            tokens_json.reserve(token_it->second.size());
            for (int token : token_it->second) {
                tokens_json.emplace_back(Json(static_cast<double>(token)));
            }
            entry["tokens"] = Json(tokens_json);
        }
        documents.emplace_back(Json(entry));
    }
    root["documents"] = Json(documents);

    fs::create_directories(path.parent_path());
    std::ofstream out(path);
    if (!out) {
        return;
    }
    out << Json(root).dump();
}

void RetrievalIndex::load_metadata(const std::filesystem::path& path) {
    namespace fs = std::filesystem;
    if (!fs::exists(path)) {
        return;
    }
    std::ifstream in(path);
    if (!in) {
        return;
    }
    std::ostringstream buffer;
    buffer << in.rdbuf();
    if (buffer.str().empty()) {
        return;
    }

    Json parsed;
    try {
        parsed = Json::parse(buffer.str());
    }
    catch (...) {
        return;
    }
    if (!parsed.is_object()) {
        return;
    }

    auto extract_double = [](const Json& value) -> std::optional<double> {
        return std::visit(
            [](const auto& raw) -> std::optional<double> {
                using T = std::decay_t<decltype(raw)>;
                if constexpr (std::is_same_v<T, double>) {
                    return raw;
                }
                else if constexpr (std::is_same_v<T, bool>) {
                    return raw ? 1.0 : 0.0;
                }
                else if constexpr (std::is_same_v<T, std::string>) {
                    try {
                        std::size_t idx = 0;
                        double parsed = std::stod(raw, &idx);
                        if (idx == raw.size()) {
                            return parsed;
                        }
                    }
                    catch (...) {
                    }
                    return std::nullopt;
                }
                else {
                    return std::nullopt;
                }
            },
            value.value());
    };

    std::scoped_lock lock(m_mutex);
    const auto& obj = parsed.as_object();
    if (auto stats_it = obj.find("stats"); stats_it != obj.end() && stats_it->second.is_object()) {
        const auto& stats = stats_it->second.as_object();
        if (auto qc = stats.find("query_count"); qc != stats.end()) {
            if (auto numeric = extract_double(qc->second)) {
                m_query_count = static_cast<std::size_t>(*numeric);
            }
        }
        if (auto hc = stats.find("hit_count"); hc != stats.end()) {
            if (auto numeric = extract_double(hc->second)) {
                m_hit_count = static_cast<std::size_t>(*numeric);
            }
        }
    }

    m_term_counts.clear();
    m_document_frequency.clear();
    m_cached_tokens.clear();
    m_document_tags.clear();

    if (auto docs_it = obj.find("documents"); docs_it != obj.end() && docs_it->second.is_array()) {
        const auto& docs = docs_it->second.as_array();
        for (const auto& entry : docs) {
            if (!entry.is_object()) {
                continue;
            }
            const auto& doc_obj = entry.as_object();
            auto id_it = doc_obj.find("id");
            if (id_it == doc_obj.end() || !id_it->second.is_string()) {
                continue;
            }
            const std::string& id = id_it->second.as_string();
            std::vector<int> tokens;
            if (auto tokens_it = doc_obj.find("tokens"); tokens_it != doc_obj.end() && tokens_it->second.is_array()) {
                const auto& tokens_json = tokens_it->second.as_array();
                tokens.reserve(tokens_json.size());
                for (const auto& token : tokens_json) {
                    if (auto numeric = extract_double(token)) {
                        tokens.push_back(static_cast<int>(*numeric));
                    }
                }
            }

            std::unordered_map<int, int> counts;
            for (int token : tokens) {
                ++counts[token];
            }
            m_cached_tokens[id] = tokens;
            m_term_counts[id] = std::move(counts);
            for (const auto& [token, _] : m_term_counts[id]) {
                (void)_; 
                ++m_document_frequency[token];
            }

            std::vector<std::string> tag_list;
            if (auto tags_it = doc_obj.find("tags"); tags_it != doc_obj.end() && tags_it->second.is_array()) {
                const auto& tags_json = tags_it->second.as_array();
                tag_list.reserve(tags_json.size());
                for (const auto& tag_value : tags_json) {
                    if (tag_value.is_string()) {
                        tag_list.push_back(tag_value.as_string());
                    }
                }
            }
            m_document_tags[id] = normalise_tags(tag_list);
        }
    }
}

std::vector<std::string> RetrievalIndex::tags_for(const std::string& document_id) const {
    std::scoped_lock lock(m_mutex);
    auto it = m_document_tags.find(document_id);
    if (it == m_document_tags.end()) {
        return {};
    }
    return it->second;
}

} // namespace almondai

