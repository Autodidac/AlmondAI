#include "../include/almondai/model.hpp"
#include "../include/almondai/adapter.hpp"
#include "../include/almondai/json.hpp"

#include <random>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <sstream>
#include <functional>
#include <variant>

namespace {

double json_to_double(const almondai::Json& value) {
    const auto& variant = value.value();
    if (std::holds_alternative<double>(variant)) {
        return std::get<double>(variant);
    }
    if (std::holds_alternative<bool>(variant)) {
        return std::get<bool>(variant) ? 1.0 : 0.0;
    }
    if (std::holds_alternative<std::string>(variant)) {
        try {
            return std::stod(std::get<std::string>(variant));
        } catch (...) {
            return 0.0;
        }
    }
    return 0.0;
}

std::vector<std::size_t> json_to_shape(const almondai::Json& value) {
    std::vector<std::size_t> shape;
    if (!value.is_array()) {
        return shape;
    }
    for (const auto& dim : value.as_array()) {
        shape.push_back(static_cast<std::size_t>(json_to_double(dim)));
    }
    return shape;
}

std::mt19937 create_rng() {
    static std::random_device rd;
    return std::mt19937(rd());
}

} // namespace

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

bool BaseDecoder::save_weights(const std::string& path) const {
    std::ofstream file(path, std::ios::trunc);
    if (!file) {
        return false;
    }
    JsonObject root;
    JsonObject cfg;
    cfg["vocab_size"] = Json(static_cast<int>(m_config.vocab_size));
    cfg["hidden_size"] = Json(static_cast<int>(m_config.hidden_size));
    cfg["num_layers"] = Json(static_cast<int>(m_config.num_layers));
    cfg["context_length"] = Json(static_cast<int>(m_config.context_length));
    cfg["learning_rate"] = Json(m_config.learning_rate);
    root["config"] = Json(cfg);

    JsonArray weights;
    for (const auto& tensor : m_weights) {
        JsonObject tensor_obj;
        JsonArray shape;
        for (std::size_t dim : tensor.shape()) {
            shape.emplace_back(Json(static_cast<int>(dim)));
        }
        tensor_obj["shape"] = Json(shape);
        JsonArray data;
        for (double value : tensor.vector()) {
            data.emplace_back(Json(value));
        }
        tensor_obj["data"] = Json(data);
        weights.emplace_back(Json(tensor_obj));
    }
    root["weights"] = Json(weights);
    file << Json(root).dump();
    return true;
}

bool BaseDecoder::load_weights(const std::string& path) {
    std::ifstream file(path);
    if (!file) {
        return false;
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    try {
        Json parsed = Json::parse(buffer.str());
        if (!parsed.is_object()) {
            return false;
        }
        const auto& obj = parsed.as_object();
        if (auto cfg_it = obj.find("config"); cfg_it != obj.end() && cfg_it->second.is_object()) {
            const auto& cfg = cfg_it->second.as_object();
            if (auto it = cfg.find("vocab_size"); it != cfg.end()) {
                m_config.vocab_size = static_cast<std::size_t>(json_to_double(it->second));
            }
            if (auto it = cfg.find("hidden_size"); it != cfg.end()) {
                m_config.hidden_size = static_cast<std::size_t>(json_to_double(it->second));
            }
            if (auto it = cfg.find("num_layers"); it != cfg.end()) {
                m_config.num_layers = static_cast<std::size_t>(json_to_double(it->second));
            }
            if (auto it = cfg.find("context_length"); it != cfg.end()) {
                m_config.context_length = static_cast<std::size_t>(json_to_double(it->second));
            }
            if (auto it = cfg.find("learning_rate"); it != cfg.end()) {
                m_config.learning_rate = json_to_double(it->second);
            }
        }
        auto weights_it = obj.find("weights");
        if (weights_it == obj.end() || !weights_it->second.is_array()) {
            return false;
        }
        const auto& weights_arr = weights_it->second.as_array();
        std::vector<Tensor> loaded;
        loaded.reserve(weights_arr.size());
        for (const auto& tensor_json : weights_arr) {
            if (!tensor_json.is_object()) {
                return false;
            }
            const auto& tensor_obj = tensor_json.as_object();
            auto shape_it = tensor_obj.find("shape");
            auto data_it = tensor_obj.find("data");
            if (shape_it == tensor_obj.end() || data_it == tensor_obj.end()) {
                return false;
            }
            if (!shape_it->second.is_array() || !data_it->second.is_array()) {
                return false;
            }
            std::vector<std::size_t> shape = json_to_shape(shape_it->second);
            Tensor tensor(shape);
            auto& data = tensor.vector();
            const auto& data_arr = data_it->second.as_array();
            if (data_arr.size() != data.size()) {
                return false;
            }
            for (std::size_t i = 0; i < data.size(); ++i) {
                data[i] = json_to_double(data_arr[i]);
            }
            loaded.emplace_back(std::move(tensor));
        }
        if (!loaded.empty()) {
            m_weights = std::move(loaded);
        }
    } catch (...) {
        return false;
    }
    return true;
}

void BaseDecoder::resize_vocab(std::size_t new_vocab_size) {
    if (new_vocab_size <= m_config.vocab_size || new_vocab_size == 0) {
        return;
    }
    const std::size_t old_vocab = m_config.vocab_size;
    std::mt19937 rng = create_rng();
    std::normal_distribution<double> dist(0.0, 0.02);

    Tensor new_embedding({new_vocab_size, m_config.hidden_size}, 0.0);
    const auto& old_embedding = m_weights.front().vector();
    auto& embed_data = new_embedding.vector();
    const std::size_t old_embed_size = old_vocab * m_config.hidden_size;
    std::copy(old_embedding.begin(), old_embedding.begin() + std::min(old_embed_size, embed_data.size()), embed_data.begin());
    for (std::size_t v = old_vocab; v < new_vocab_size; ++v) {
        for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
            embed_data[v * m_config.hidden_size + h] = dist(rng);
        }
    }

    Tensor new_projection({m_config.hidden_size, new_vocab_size}, 0.0);
    const auto& old_projection = m_weights.back().vector();
    auto& proj_data = new_projection.vector();
    for (std::size_t h = 0; h < m_config.hidden_size; ++h) {
        for (std::size_t v = 0; v < old_vocab; ++v) {
            proj_data[h * new_vocab_size + v] = old_projection[h * old_vocab + v];
        }
        for (std::size_t v = old_vocab; v < new_vocab_size; ++v) {
            proj_data[h * new_vocab_size + v] = dist(rng);
        }
    }

    m_weights.front() = std::move(new_embedding);
    m_weights.back() = std::move(new_projection);
    m_config.vocab_size = new_vocab_size;
}

StudentModel::StudentModel(BaseDecoder base) : m_base(std::move(base)) {}

std::vector<double> StudentModel::forward(const std::vector<int>& tokens) const {
    return m_base.forward(tokens);
}

void StudentModel::update(const std::vector<double>& gradient) {
    m_base.apply_gradients(gradient);
}

} // namespace almondai

