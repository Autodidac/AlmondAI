#pragma once

#include "tensor.hpp"

#include <vector>
#include <string>
#include <unordered_map>
#include <optional>

namespace almondai {

struct ModelConfig {
    std::size_t vocab_size = 0;
    std::size_t hidden_size = 128;
    std::size_t num_layers = 2;
    std::size_t context_length = 256;
    double learning_rate = 1e-3;
};

class Adapter;

class BaseDecoder {
public:
    explicit BaseDecoder(ModelConfig config);

    const ModelConfig& config() const noexcept { return m_config; }

    std::vector<double> forward(const std::vector<int>& tokens) const;
    void apply_gradients(const std::vector<double>& gradient);

    const std::vector<Tensor>& weights() const noexcept { return m_weights; }

    bool save_weights(const std::string& path) const;
    bool load_weights(const std::string& path);
    void resize_vocab(std::size_t new_vocab_size);

    void attach_adapter(const Adapter* adapter);
    const Adapter* active_adapter() const noexcept { return m_active_adapter; }

private:
    ModelConfig m_config;
    std::vector<Tensor> m_weights;
    const Adapter* m_active_adapter = nullptr;

    std::vector<double> forward_layer(std::size_t layer, const std::vector<double>& input) const;
};

class StudentModel {
public:
    explicit StudentModel(BaseDecoder base);

    std::vector<double> forward(const std::vector<int>& tokens) const;
    void update(const std::vector<double>& gradient);

    BaseDecoder& base() noexcept { return m_base; }
    const BaseDecoder& base() const noexcept { return m_base; }

private:
    BaseDecoder m_base;
};

} // namespace almondai

