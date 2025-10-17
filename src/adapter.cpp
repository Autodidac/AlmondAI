#include "almondai/adapter.hpp"

#include <random>

namespace almondai {

namespace {
std::vector<double> random_weights(std::size_t count) {
    std::mt19937 rng{std::random_device{}()};
    std::uniform_real_distribution<double> dist(-0.05, 0.05);
    std::vector<double> values(count, 0.0);
    for (double &v : values) {
        v = dist(rng);
    }
    return values;
}
} // namespace

AdapterState::AdapterState(std::string name, AdapterConfig cfg)
    : m_name(std::move(name)),
      m_config(cfg),
      m_up_weights(random_weights(cfg.rank * cfg.output_dim)),
      m_down_weights(random_weights(cfg.input_dim * cfg.rank)),
      m_previous_weights(m_up_weights.size() + m_down_weights.size(), 0.0) {}

const std::string &AdapterState::name() const noexcept { return m_name; }

const AdapterConfig &AdapterState::config() const noexcept { return m_config; }

std::vector<double> AdapterState::forward(const std::vector<double> &hidden) const {
    std::vector<double> rank(m_config.rank, 0.0);
    for (std::size_t r = 0; r < m_config.rank; ++r) {
        double sum = 0.0;
        for (std::size_t i = 0; i < m_config.input_dim; ++i) {
            sum += hidden[i] * m_down_weights[i * m_config.rank + r];
        }
        rank[r] = sum;
    }
    std::vector<double> projected(m_config.output_dim, 0.0);
    for (std::size_t o = 0; o < m_config.output_dim; ++o) {
        double sum = 0.0;
        for (std::size_t r = 0; r < m_config.rank; ++r) {
            sum += rank[r] * m_up_weights[r * m_config.output_dim + o];
        }
        projected[o] = sum;
    }
    return projected;
}

void AdapterState::apply_update(const std::vector<double> &gradient) {
    const std::size_t total = m_up_weights.size() + m_down_weights.size();
    if (gradient.size() != total) {
        return;
    }
    m_previous_weights.reserve(total);
    if (m_previous_weights.size() != total) {
        m_previous_weights.assign(total, 0.0);
    }
    std::size_t index = 0U;
    for (std::size_t i = 0; i < m_down_weights.size(); ++i, ++index) {
        m_previous_weights[index] = m_down_weights[i];
        m_down_weights[i] -= m_config.learning_rate * gradient[index];
    }
    for (std::size_t i = 0; i < m_up_weights.size(); ++i, ++index) {
        m_previous_weights[index] = m_up_weights[i];
        m_up_weights[i] -= m_config.learning_rate * gradient[index];
    }
}

void AdapterState::regularize(const std::vector<double> &prior) {
    if (prior.size() != m_up_weights.size() + m_down_weights.size()) {
        return;
    }
    std::size_t index = 0U;
    for (std::size_t i = 0; i < m_down_weights.size(); ++i, ++index) {
        const double delta = m_down_weights[i] - prior[index];
        m_down_weights[i] -= m_config.ewc_lambda * delta;
    }
    for (std::size_t i = 0; i < m_up_weights.size(); ++i, ++index) {
        const double delta = m_up_weights[i] - prior[index];
        m_up_weights[i] -= m_config.ewc_lambda * delta;
    }
}

const std::vector<double> &AdapterState::weights() const noexcept {
    return m_up_weights;
}

std::vector<double> AdapterState::combined_weights() const {
    std::vector<double> combined;
    combined.reserve(m_down_weights.size() + m_up_weights.size());
    combined.insert(combined.end(), m_down_weights.begin(), m_down_weights.end());
    combined.insert(combined.end(), m_up_weights.begin(), m_up_weights.end());
    return combined;
}

void AdapterRegistry::register_adapter(const AdapterState &adapter) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_adapters[adapter.name()] = adapter;
}

void AdapterRegistry::unregister_adapter(const std::string &name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    m_adapters.erase(name);
    if (m_active_name && *m_active_name == name) {
        m_active_name.reset();
    }
}

std::optional<AdapterState> AdapterRegistry::get(const std::string &name) const {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_adapters.find(name);
    if (it == m_adapters.end()) {
        return std::nullopt;
    }
    return it->second;
}

void AdapterRegistry::promote(const std::string &name) {
    std::lock_guard<std::mutex> lock(m_mutex);
    auto it = m_adapters.find(name);
    if (it == m_adapters.end()) {
        return;
    }
    if (m_active_name) {
        auto prev = m_adapters.find(*m_active_name);
        if (prev != m_adapters.end()) {
            m_previous_active = prev->second;
        }
    }
    m_active_name = name;
}

std::optional<AdapterState> AdapterRegistry::active() const {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_active_name) {
        return std::nullopt;
    }
    auto it = m_adapters.find(*m_active_name);
    if (it == m_adapters.end()) {
        return std::nullopt;
    }
    return it->second;
}

void AdapterRegistry::rollback() {
    std::lock_guard<std::mutex> lock(m_mutex);
    if (!m_previous_active) {
        return;
    }
    m_adapters[m_previous_active->name()] = *m_previous_active;
    m_active_name = m_previous_active->name();
    m_previous_active.reset();
}

} // namespace almondai
