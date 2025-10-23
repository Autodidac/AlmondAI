#include "../include/almondai/scheduler.hpp"

#include <algorithm>
#include <cmath>

namespace almondai {

WarmupCosineScheduler::WarmupCosineScheduler(double base_learning_rate,
                                             std::size_t warmup_steps,
                                             std::size_t total_steps,
                                             double min_lr_ratio)
    : m_base_learning_rate(base_learning_rate)
    , m_warmup_steps(warmup_steps)
    , m_total_steps(total_steps)
    , m_min_lr_ratio(min_lr_ratio) {}

double WarmupCosineScheduler::learning_rate_scale(std::size_t step) const {
    if (m_total_steps == 0) {
        return 1.0;
    }
    if (step < m_warmup_steps && m_warmup_steps > 0) {
        return static_cast<double>(step + 1) / static_cast<double>(m_warmup_steps);
    }
    if (m_total_steps <= m_warmup_steps) {
        return m_min_lr_ratio;
    }
    const std::size_t decay_steps = m_total_steps - m_warmup_steps;
    const std::size_t decay_step = std::min(step - std::min(step, m_warmup_steps), decay_steps - 1);
    const double progress = static_cast<double>(decay_step) / static_cast<double>(decay_steps - 1);
    const double cosine = 0.5 * (1.0 + std::cos(progress * std::acos(-1.0)));
    return m_min_lr_ratio + (1.0 - m_min_lr_ratio) * cosine;
}

} // namespace almondai

