#include "almondai/model.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>

#include "almondai/tokenizer_word.hpp"
#include "almondai/retrieval.hpp"
#include "almondai/governor.hpp"

namespace almondai {

DecoderModel::DecoderModel(ModelConfig cfg, AdapterRegistry &registry)
    : m_config(cfg),
      m_registry(registry),
      m_base_weights(std::vector<std::size_t>{cfg.vocab_size, cfg.hidden_dim}) {
    for (std::size_t i = 0; i < m_base_weights.size(); ++i) {
        m_base_weights[i] = static_cast<double>((i % 13) - 6) / 13.0;
    }
}

std::string DecoderModel::generate(const std::string &prompt,
                                   const TokenizerWord &tokenizer,
                                   RetrievalIndex &retrieval,
                                   PolicyGovernor &governor,
                                   const GenerationOptions &options) const {
    auto retrieval_results = retrieval.query(prompt, tokenizer, 2);
    std::string augmented = prompt;
    for (const auto &result : retrieval_results) {
        augmented += " \n" + result.document;
    }
    auto tokens = tokenizer.encode(augmented);
    auto hidden = forward_hidden(tokens);

    std::vector<double> logits = hidden;
    auto active_adapter = m_registry.active();
    if (active_adapter) {
        auto adapter_out = active_adapter->forward(hidden);
        for (std::size_t i = 0; i < logits.size() && i < adapter_out.size(); ++i) {
            logits[i] += adapter_out[i];
        }
    }

    retrieval.update_embedding(hidden);

    auto decoded = decode_logits(logits, tokenizer);
    auto governed = governor.inspect(decoded, options.blocklist_tokens, options.schema_json);
    if (!governed.allowed) {
        return "<rejected>";
    }
    return governed.output;
}

std::vector<double> DecoderModel::forward_hidden(const std::vector<int> &tokens) const {
    std::vector<double> hidden(m_config.hidden_dim, 0.0);
    if (m_config.vocab_size == 0 || m_config.hidden_dim == 0) {
        return hidden;
    }
    for (int token : tokens) {
        if (token < 0) {
            continue;
        }
        std::size_t index = static_cast<std::size_t>(token) % m_config.vocab_size;
        for (std::size_t h = 0; h < m_config.hidden_dim; ++h) {
            hidden[h] += m_base_weights.data()[index * m_config.hidden_dim + h];
        }
    }
    if (!tokens.empty()) {
        const double scale = 1.0 / static_cast<double>(tokens.size());
        for (double &value : hidden) {
            value *= scale;
        }
    }
    return hidden;
}

std::string DecoderModel::decode_logits(const std::vector<double> &logits,
                                        const TokenizerWord &tokenizer) const {
    std::vector<std::size_t> indices(logits.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::partial_sort(indices.begin(), indices.begin() + std::min<std::size_t>(3, indices.size()), indices.end(),
                      [&](std::size_t a, std::size_t b) { return logits[a] > logits[b]; });
    std::vector<int> token_ids;
    for (std::size_t i = 0; i < indices.size() && i < 3; ++i) {
        int token_id = static_cast<int>(indices[i] % std::max<std::size_t>(1, tokenizer.vocab_size()));
        token_ids.push_back(token_id);
    }
    return tokenizer.decode(token_ids);
}

void DecoderModel::log_weights(const std::string &path) const {
    std::ofstream out(path);
#ifdef _WIN32
    const char newline = '\r';
#else
    const char newline = '\n';
#endif
    for (std::size_t i = 0; i < m_base_weights.size(); ++i) {
        out << m_base_weights.data()[i] << newline;
    }
}

const ModelConfig &DecoderModel::config() const noexcept { return m_config; }

} // namespace almondai
