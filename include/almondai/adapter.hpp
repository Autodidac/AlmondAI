#pragma once

#include "almondai/tensor.hpp"

#include <vector>
#include <string>
#include <optional>
#include <mutex>
#include <cstddef>

namespace almondai {

struct AdapterConfig {
    std::size_t rank = 8;
    double alpha = 16.0;
    double ewc_lambda = 0.1;
};

class Adapter {
public:
    Adapter(std::string name, std::size_t hidden_size, AdapterConfig config);
    Adapter(const Adapter&) = delete;
    Adapter& operator=(const Adapter&) = delete;
    Adapter(Adapter&& other) noexcept;
    Adapter& operator=(Adapter&& other) noexcept;

    const std::string& name() const noexcept { return m_name; }
    const AdapterConfig& config() const noexcept { return m_config; }

    void update_statistics(const std::vector<double>& activations);
    std::vector<double> project(const std::vector<double>& activations) const;

    void apply_gradient(const std::vector<double>& gradient);

    double norm() const;

    void set_base_fisher(const std::vector<double>& fisher);

private:
    std::string m_name;
    AdapterConfig m_config;
    Tensor m_down;
    Tensor m_up;
    std::vector<double> m_fisher_diagonal;
    mutable std::mutex m_mutex;
};

class AdapterManager {
public:
    void register_adapter(Adapter adapter);
    const Adapter* active_adapter() const noexcept;
    Adapter* active_adapter();
    void activate(const std::string& name);
    void deactivate();
    std::vector<std::string> names() const;

private:
    std::vector<Adapter> m_adapters;
    std::size_t m_active_index = static_cast<std::size_t>(-1);
};

} // namespace almondai

