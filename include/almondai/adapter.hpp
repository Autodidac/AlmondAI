#ifndef ALMONDAI_ADAPTER_HPP
#define ALMONDAI_ADAPTER_HPP

#include <string>
#include <vector>
#include <map>
#include <optional>
#include <mutex>

#include "almondai/tensor.hpp"

namespace almondai {

struct AdapterConfig {
    std::size_t input_dim{0};
    std::size_t output_dim{0};
    std::size_t rank{1};
    double learning_rate{1e-2};
    double ewc_lambda{1e-2};
};

class AdapterState {
public:
    AdapterState() = default;
    AdapterState(std::string name, AdapterConfig cfg);

    const std::string &name() const noexcept;
    const AdapterConfig &config() const noexcept;

    std::vector<double> forward(const std::vector<double> &hidden) const;
    void apply_update(const std::vector<double> &gradient);
    void regularize(const std::vector<double> &prior);
    const std::vector<double> &weights() const noexcept;
    std::vector<double> combined_weights() const;

private:
    std::string m_name;
    AdapterConfig m_config;
    std::vector<double> m_up_weights;
    std::vector<double> m_down_weights;
    std::vector<double> m_previous_weights;
};

class AdapterRegistry {
public:
    void register_adapter(const AdapterState &adapter);
    void unregister_adapter(const std::string &name);
    std::optional<AdapterState> get(const std::string &name) const;
    void promote(const std::string &name);
    std::optional<AdapterState> active() const;
    void rollback();

private:
    mutable std::mutex m_mutex;
    std::map<std::string, AdapterState> m_adapters;
    std::optional<std::string> m_active_name;
    std::optional<AdapterState> m_previous_active;
};

} // namespace almondai

#endif // ALMONDAI_ADAPTER_HPP
