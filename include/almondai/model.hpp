#ifndef ALMONDAI_MODEL_HPP
#define ALMONDAI_MODEL_HPP

#include <string>
#include <vector>
#include <optional>
#include <functional>

#include "almondai/adapter.hpp"
#include "almondai/tensor.hpp"

namespace almondai {

class TokenizerWord;
class RetrievalIndex;
class PolicyGovernor;

struct ModelConfig {
    std::size_t vocab_size{0};
    std::size_t hidden_dim{64};
    std::size_t output_dim{64};
    std::size_t max_seq_len{128};
};

struct GenerationOptions {
    std::optional<std::string> schema_json;
    std::vector<std::string> blocklist_tokens;
};

class DecoderModel {
public:
    DecoderModel(ModelConfig cfg, AdapterRegistry &registry);

    std::string generate(const std::string &prompt,
                         const TokenizerWord &tokenizer,
                         RetrievalIndex &retrieval,
                         PolicyGovernor &governor,
                         const GenerationOptions &options) const;

    std::vector<double> forward_hidden(const std::vector<int> &tokens) const;
    std::string decode_logits(const std::vector<double> &logits,
                              const TokenizerWord &tokenizer) const;
    void log_weights(const std::string &path) const;
    const ModelConfig &config() const noexcept;

private:
    ModelConfig m_config;
    AdapterRegistry &m_registry;
    Tensor m_base_weights;
};

} // namespace almondai

#endif // ALMONDAI_MODEL_HPP
