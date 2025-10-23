#include "../include/almondai/optim_adamw.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace almondai {

AdamWOptimizer::AdamWOptimizer(std::size_t parameter_count)
    : AdamWOptimizer(parameter_count, Params{}) {}

AdamWOptimizer::AdamWOptimizer(std::size_t parameter_count, Params params)
    : m_params(params) {
    reset(parameter_count);
}

void AdamWOptimizer::reset(std::size_t parameter_count) {
    m_moment1.assign(parameter_count, 0.0);
    m_moment2.assign(parameter_count, 0.0);
    m_step = 0;
}

void AdamWOptimizer::zero_state() {
    std::fill(m_moment1.begin(), m_moment1.end(), 0.0);
    std::fill(m_moment2.begin(), m_moment2.end(), 0.0);
    m_step = 0;
}

void AdamWOptimizer::step(std::vector<double>& parameters,
                          const std::vector<double>& gradients,
                          double learning_rate_scale) {
    if (parameters.size() != gradients.size()) {
        throw std::invalid_argument("adamw parameter/gradient size mismatch");
    }
    if (m_moment1.size() != parameters.size()) {
        reset(parameters.size());
    }

    ++m_step;
    const double lr = m_params.learning_rate * learning_rate_scale;
    const double bias_correction1 = 1.0 - std::pow(m_params.beta1, static_cast<double>(m_step));
    const double bias_correction2 = 1.0 - std::pow(m_params.beta2, static_cast<double>(m_step));

    for (std::size_t i = 0; i < parameters.size(); ++i) {
        const double grad = gradients[i];
        m_moment1[i] = m_params.beta1 * m_moment1[i] + (1.0 - m_params.beta1) * grad;
        m_moment2[i] = m_params.beta2 * m_moment2[i] + (1.0 - m_params.beta2) * (grad * grad);

        const double m_hat = m_moment1[i] / bias_correction1;
        const double v_hat = m_moment2[i] / bias_correction2;
        const double denom = std::sqrt(v_hat) + m_params.epsilon;
        const double update = m_hat / denom;
        const double decay = m_params.weight_decay * parameters[i];
        parameters[i] -= lr * (update + decay);
    }
}

} // namespace almondai

