#pragma once

#include <cstddef>
#include <vector>

namespace almondai {

class AdamWOptimizer {
public:
    struct Params {
        double learning_rate = 3e-4;
        double beta1 = 0.9;
        double beta2 = 0.98;
        double epsilon = 1e-8;
        double weight_decay = 0.01;
    };

    AdamWOptimizer() = default;
    explicit AdamWOptimizer(std::size_t parameter_count);
    AdamWOptimizer(std::size_t parameter_count, Params params);

    void reset(std::size_t parameter_count);
    void set_params(Params params) { m_params = params; }
    const Params& params() const noexcept { return m_params; }

    void step(std::vector<double>& parameters,
              const std::vector<double>& gradients,
              double learning_rate_scale = 1.0);

    void zero_state();
    std::size_t step_index() const noexcept { return m_step; }

private:
    Params m_params;
    std::vector<double> m_moment1;
    std::vector<double> m_moment2;
    std::size_t m_step = 0;
};

} // namespace almondai

