#include "../include/almondai/adapter.hpp"

#include <random>
#include <chrono>
#include <numeric>
#include <algorithm>
#include <cmath>
#include <mutex>
#include <utility>

namespace almondai {

Adapter::Adapter(std::string name, std::size_t hidden_size, AdapterConfig config)
    : m_name(std::move(name)), m_config(config),
      m_down({hidden_size, config.rank}, 0.0),
      m_up({config.rank, hidden_size}, 0.0),
      m_fisher_diagonal(hidden_size, 1.0) {
    const auto seed = static_cast<unsigned>(
        std::chrono::high_resolution_clock::now().time_since_epoch().count());
    std::mt19937 rng(seed);
    std::normal_distribution<double> dist(0.0, 0.02);
    for (double& value : m_down.vector()) {
        value = dist(rng);
    }
    for (double& value : m_up.vector()) {
        value = dist(rng);
    }
}

Adapter::Adapter(Adapter&& other) noexcept
    : m_name(std::move(other.m_name)),
      m_config(other.m_config),
      m_down(std::move(other.m_down)),
      m_up(std::move(other.m_up)),
      m_fisher_diagonal(std::move(other.m_fisher_diagonal)) {}

Adapter& Adapter::operator=(Adapter&& other) noexcept {
    if (this != &other) {
        std::scoped_lock lock(m_mutex, other.m_mutex);
        m_name = std::move(other.m_name);
        m_config = other.m_config;
        m_down = std::move(other.m_down);
        m_up = std::move(other.m_up);
        m_fisher_diagonal = std::move(other.m_fisher_diagonal);
    }
    return *this;
}

void Adapter::update_statistics(const std::vector<double>& activations) {
    if (activations.size() != m_fisher_diagonal.size() || activations.empty()) {
        return;
    }
    std::scoped_lock lock(m_mutex);
    const double norm = std::inner_product(activations.begin(), activations.end(), activations.begin(), 0.0);
    const double scaled = norm / static_cast<double>(activations.size());
    for (double& value : m_fisher_diagonal) {
        value = 0.9 * value + 0.1 * scaled;
    }
}

std::vector<double> Adapter::project(const std::vector<double>& activations) const {
    std::scoped_lock lock(m_mutex);
    std::vector<double> down_proj(m_config.rank, 0.0);
    const auto& down_data = m_down.vector();
    const auto& up_data = m_up.vector();
    const std::size_t hidden = activations.size();
    for (std::size_t r = 0; r < m_config.rank; ++r) {
        double sum = 0.0;
        for (std::size_t h = 0; h < hidden; ++h) {
            sum += down_data[h * m_config.rank + r] * activations[h];
        }
        down_proj[r] = sum;
    }
    std::vector<double> result(hidden, 0.0);
    for (std::size_t h = 0; h < hidden; ++h) {
        double sum = 0.0;
        for (std::size_t r = 0; r < m_config.rank; ++r) {
            sum += up_data[r * hidden + h] * down_proj[r];
        }
        result[h] = sum * (m_config.alpha / static_cast<double>(m_config.rank));
    }
    return result;
}

void Adapter::apply_gradient(const std::vector<double>& activations,
                             const std::vector<double>& gradient) {
    if (activations.size() != gradient.size() || activations.size() != m_fisher_diagonal.size()) {
        return;
    }

    std::scoped_lock lock(m_mutex);
    const std::size_t hidden = activations.size();
    if (hidden == 0 || m_config.rank == 0) {
        return;
    }

    auto& down_data = m_down.vector();
    auto& up_data = m_up.vector();
    const std::vector<double> up_snapshot = up_data;

    std::vector<double> scaled_grad(hidden, 0.0);
    for (std::size_t h = 0; h < hidden; ++h) {
        const double fisher = m_fisher_diagonal[h];
        scaled_grad[h] = gradient[h] / (fisher + m_config.ewc_lambda);
    }

    std::vector<double> down_projection(m_config.rank, 0.0);
    for (std::size_t r = 0; r < m_config.rank; ++r) {
        double sum = 0.0;
        for (std::size_t h = 0; h < hidden; ++h) {
            sum += down_data[h * m_config.rank + r] * activations[h];
        }
        down_projection[r] = sum;
    }

    std::vector<double> back_projection(m_config.rank, 0.0);
    for (std::size_t r = 0; r < m_config.rank; ++r) {
        double sum = 0.0;
        for (std::size_t h = 0; h < hidden; ++h) {
            sum += scaled_grad[h] * up_snapshot[r * hidden + h];
        }
        back_projection[r] = sum;
    }

    constexpr double kAdapterLearningRate = 0.01;
    const double scale = m_config.alpha / static_cast<double>(m_config.rank);

    for (std::size_t r = 0; r < m_config.rank; ++r) {
        for (std::size_t h = 0; h < hidden; ++h) {
            const double grad_up = scaled_grad[h] * down_projection[r] * scale;
            up_data[r * hidden + h] -= kAdapterLearningRate * grad_up;
        }
    }

    for (std::size_t h = 0; h < hidden; ++h) {
        for (std::size_t r = 0; r < m_config.rank; ++r) {
            const double grad_down = activations[h] * back_projection[r] * scale;
            down_data[h * m_config.rank + r] -= kAdapterLearningRate * grad_down;
        }
    }
}

double Adapter::norm() const {
    std::scoped_lock lock(m_mutex);
    double sum = 0.0;
    for (double value : m_down.vector()) {
        sum += value * value;
    }
    for (double value : m_up.vector()) {
        sum += value * value;
    }
    return std::sqrt(sum);
}

void Adapter::set_base_fisher(const std::vector<double>& fisher) {
    std::scoped_lock lock(m_mutex);
    m_fisher_diagonal = fisher;
}

void AdapterManager::register_adapter(Adapter adapter) {
    m_adapters.push_back(std::move(adapter));
    if (m_active_index >= m_adapters.size()) {
        m_active_index = static_cast<std::size_t>(-1);
    }
}

const Adapter* AdapterManager::active_adapter() const noexcept {
    if (m_active_index >= m_adapters.size()) {
        return nullptr;
    }
    return &m_adapters[m_active_index];
}

Adapter* AdapterManager::active_adapter() {
    if (m_active_index >= m_adapters.size()) {
        return nullptr;
    }
    return &m_adapters[m_active_index];
}

void AdapterManager::activate(const std::string& name) {
    m_active_index = static_cast<std::size_t>(-1);
    for (std::size_t i = 0; i < m_adapters.size(); ++i) {
        if (m_adapters[i].name() == name) {
            m_active_index = i;
            break;
        }
    }
}

void AdapterManager::deactivate() {
    m_active_index = static_cast<std::size_t>(-1);
}

std::vector<std::string> AdapterManager::names() const {
    std::vector<std::string> result;
    result.reserve(m_adapters.size());
    for (const auto& adapter : m_adapters) {
        result.push_back(adapter.name());
    }
    return result;
}

} // namespace almondai

