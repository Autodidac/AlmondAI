#include "almondai/model.hpp"
#include "almondai/adapter.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>

namespace almondai {

BaseDecoder::BaseDecoder(ModelConfig config) : m_config(config) {
    if (m_config.vocab_size == 0) {
        m_config.vocab_size = 4;
    }
    m_weights.reserve(m_config.num_layers + 2);
    m_weights.emplace_back(std::vector<std::size_t>{m_config.vocab_size, m_config.hidden_size}, 0.0);
    for (std::size_t i = 0; i < m_config.num_layers; ++i) {
        m_weights.emplace_back(std::vector<std::size_t>{m_config.hidden_size, m_config.hidden_size}, 0.0);
    }
    m_weights.emplace_back(std::vector<std::size_t>{m_config.hidden_size, m_config.vocab_size}, 0.0);

    const auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.02);
    for (auto& weight : m_weights) {
        for (double& value : weight.vector()) {
            value = dist(rng);
        }
    }
}

std::vector<double> BaseDecoder::forward(const std::vector<int>& tokens) const {
    if (tokens.empty()) {
        return std::vector<double>(m_config.vocab_size, 0.0);
    }
    std::vector<double> hidden(m_config.hidden_size, 0.0);
    const auto& embedding = m_weights.front().vector();
    for (int token : tokens) {
        std::size_t index = static_cast<std::size_t>(std::max(token, 0));
        if (index >= m_config.vocab_size) {
            index = 0;
        }
        for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
            hidden[h] += embedding[index * m_config.hidden_size + h];
        }
    }
    const double inv = 1.0 / static_cast<double>(tokens.size());
    for (double& value : hidden) {
        value *= inv;
    }

    for (std::size_t layer = 1; layer <= m_config.num_layers; ++layer) {
        hidden = forward_layer(layer, hidden);
    }

    if (m_active_adapter != nullptr) {
        const std::vector<double> delta = m_active_adapter->project(hidden);
        for (std::size_t i = 0; i < hidden.size(); ++i) {
            hidden[i] += delta[i];
        }
    }

    const Tensor& projection = m_weights.back();
    std::vector<double> logits(m_config.vocab_size, 0.0);
    const auto& proj_data = projection.vector();
    for (std::size_t v = 0; v < m_config.vocab_size; ++v) {
        double sum = 0.0;
        for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
            sum += proj_data[h * m_config.vocab_size + v] * hidden[h];
        }
        logits[v] = sum;
    }
    return logits;
}

std::vector<double> BaseDecoder::forward_layer(std::size_t layer, const std::vector<double>& input) const {
    const Tensor& weight = m_weights[layer];
    std::vector<double> output(m_config.hidden_size, 0.0);
    const auto& data = weight.vector();
    for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
        double sum = 0.0;
        for (std::size_t in = 0; in < m_config.hidden_size; ++in) {
            sum += data[in * m_config.hidden_size + h] * input[in];
        }
        output[h] = std::tanh(sum);
    }
    return output;
}

void BaseDecoder::apply_gradients(const std::vector<double>& gradient) {
    if (gradient.size() != m_config.hidden_size) {
        return;
    }
    Tensor& projection = m_weights.back();
    auto& proj = projection.vector();
    for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
        for (std::size_t v = 0; v < m_config.vocab_size; ++v) {
            const std::size_t idx = h * m_config.vocab_size + v;
            proj[idx] -= m_config.learning_rate * gradient[h];
        }
    }
}

void BaseDecoder::attach_adapter(const Adapter* adapter) {
    m_active_adapter = adapter;
}

StudentModel::StudentModel(BaseDecoder base) : m_base(std::move(base)) {}

std::vector<double> StudentModel::forward(const std::vector<int>& tokens) const {
    return m_base.forward(tokens);
}

void StudentModel::update(const std::vector<double>& gradient) {
    m_base.apply_gradients(gradient);
}

} // namespace almondai

