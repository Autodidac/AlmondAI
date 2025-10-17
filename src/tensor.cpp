#include "almondai/tensor.hpp"

#include <stdexcept>

namespace almondai {

Tensor::Tensor(std::vector<std::size_t> shape_) : m_shape(std::move(shape_)) {
    std::size_t total = 1;
    for (std::size_t s : m_shape) {
        total *= s;
    }
    m_data.assign(total, 0.0);
}

Tensor::Tensor(std::vector<std::size_t> shape_, std::vector<double> data_)
    : m_shape(std::move(shape_)), m_data(std::move(data_)) {}

Tensor::Tensor(std::initializer_list<double> values)
    : m_shape{values.size()}, m_data(values) {}

const std::vector<std::size_t> &Tensor::shape() const noexcept { return m_shape; }

const std::vector<double> &Tensor::data() const noexcept { return m_data; }

std::vector<double> &Tensor::data() noexcept { return m_data; }

std::size_t Tensor::size() const noexcept { return m_data.size(); }

double &Tensor::operator[](std::size_t idx) { return m_data.at(idx); }

double Tensor::operator[](std::size_t idx) const { return m_data.at(idx); }

void Tensor::fill(double value) { std::fill(m_data.begin(), m_data.end(), value); }

Tensor Tensor::matmul(const Tensor &other) const {
    if (m_shape.size() != 2 || other.m_shape.size() != 2) {
        throw std::invalid_argument("matmul expects two 2D tensors");
    }
    if (m_shape[1] != other.m_shape[0]) {
        throw std::invalid_argument("matmul dimension mismatch");
    }

    Tensor result(std::vector<std::size_t>{m_shape[0], other.m_shape[1]});
    for (std::size_t i = 0; i < m_shape[0]; ++i) {
        for (std::size_t j = 0; j < other.m_shape[1]; ++j) {
            double sum = 0.0;
            for (std::size_t k = 0; k < m_shape[1]; ++k) {
                sum += m_data[i * m_shape[1] + k] * other.m_data[k * other.m_shape[1] + j];
            }
            result.m_data[i * other.m_shape[1] + j] = sum;
        }
    }
    return result;
}

Tensor Tensor::add(const Tensor &other) const {
    if (m_data.size() != other.m_data.size()) {
        throw std::invalid_argument("add dimension mismatch");
    }
    Tensor result(m_shape, m_data);
    for (std::size_t i = 0; i < m_data.size(); ++i) {
        result.m_data[i] += other.m_data[i];
    }
    return result;
}

Tensor Tensor::scale(double factor) const {
    Tensor result(m_shape, m_data);
    for (double &value : result.m_data) {
        value *= factor;
    }
    return result;
}

Tensor Tensor::transpose(std::size_t rows, std::size_t cols) const {
    if (rows * cols != m_data.size()) {
        throw std::invalid_argument("transpose dimension mismatch");
    }
    Tensor result(std::vector<std::size_t>{cols, rows});
    for (std::size_t r = 0; r < rows; ++r) {
        for (std::size_t c = 0; c < cols; ++c) {
            result.m_data[c * rows + r] = m_data[r * cols + c];
        }
    }
    return result;
}

Tensor outer_product(const std::vector<double> &a, const std::vector<double> &b) {
    Tensor result(std::vector<std::size_t>{a.size(), b.size()});
    for (std::size_t i = 0; i < a.size(); ++i) {
        for (std::size_t j = 0; j < b.size(); ++j) {
            result.data()[i * b.size() + j] = a[i] * b[j];
        }
    }
    return result;
}

double dot(const std::vector<double> &a, const std::vector<double> &b) {
    if (a.size() != b.size()) {
        throw std::invalid_argument("dot dimension mismatch");
    }
    double result = 0.0;
    for (std::size_t i = 0; i < a.size(); ++i) {
        result += a[i] * b[i];
    }
    return result;
}

} // namespace almondai
