#pragma once

#include <cstddef>

namespace almondai {

class WarmupCosineScheduler {
public:
    WarmupCosineScheduler(double base_learning_rate = 3e-4,
                          std::size_t warmup_steps = 250,
                          std::size_t total_steps = 10000,
                          double min_lr_ratio = 0.1);

    void set_total_steps(std::size_t steps) noexcept { m_total_steps = steps; }
    void set_warmup_steps(std::size_t steps) noexcept { m_warmup_steps = steps; }
    void set_min_ratio(double ratio) noexcept { m_min_lr_ratio = ratio; }

    [[nodiscard]] double learning_rate_scale(std::size_t step) const;
    [[nodiscard]] double base_learning_rate() const noexcept { return m_base_learning_rate; }

private:
    double m_base_learning_rate;
    std::size_t m_warmup_steps;
    std::size_t m_total_steps;
    double m_min_lr_ratio;
};

} // namespace almondai

