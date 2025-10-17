#pragma once

#include <vector>
#include <string>
#include <initializer_list>
#include <cstddef>
#include <span>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <cmath>

namespace almondai {

class Tensor {
public:
    Tensor() = default;
    Tensor(std::vector<std::size_t> shape, double fill = 0.0)
        : m_shape(std::move(shape)), m_data(size_from_shape(m_shape), fill) {}

    Tensor(std::initializer_list<std::size_t> shape, double fill = 0.0)
        : Tensor(std::vector<std::size_t>(shape), fill) {}

    std::size_t size() const noexcept { return m_data.size(); }

    const std::vector<std::size_t>& shape() const noexcept { return m_shape; }

    double* data() noexcept { return m_data.data(); }
    const double* data() const noexcept { return m_data.data(); }

    double& operator[](std::size_t index) { return m_data.at(index); }
    const double& operator[](std::size_t index) const { return m_data.at(index); }

    double& at(std::size_t index) { return m_data.at(index); }
    const double& at(std::size_t index) const { return m_data.at(index); }

    std::vector<double>& vector() noexcept { return m_data; }
    const std::vector<double>& vector() const noexcept { return m_data; }

    Tensor& operator+=(const Tensor& other) {
        require_same_shape(other);
        for (std::size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] += other.m_data[i];
        }
        return *this;
    }

    Tensor& operator-=(const Tensor& other) {
        require_same_shape(other);
        for (std::size_t i = 0; i < m_data.size(); ++i) {
            m_data[i] -= other.m_data[i];
        }
        return *this;
    }

    Tensor& operator*=(double scalar) {
        for (double& value : m_data) {
            value *= scalar;
        }
        return *this;
    }

    static Tensor zeros(std::initializer_list<std::size_t> shape) {
        return Tensor(shape, 0.0);
    }

    static Tensor random(std::initializer_list<std::size_t> shape, double scale = 0.02);

    double l2_norm() const {
        double sum = 0.0;
        for (double value : m_data) {
            sum += value * value;
        }
        return std::sqrt(sum);
    }

private:
    std::vector<std::size_t> m_shape;
    std::vector<double> m_data;

    static std::size_t size_from_shape(const std::vector<std::size_t>& shape) {
        if (shape.empty()) {
            return 0;
        }
        return std::accumulate(shape.begin(), shape.end(), std::size_t{1}, std::multiplies<>());
    }

    void require_same_shape(const Tensor& other) const {
        if (m_shape != other.m_shape) {
            throw std::invalid_argument("tensor shape mismatch");
        }
    }
};

} // namespace almondai

