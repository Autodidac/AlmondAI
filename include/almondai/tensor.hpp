#ifndef ALMONDAI_TENSOR_HPP
#define ALMONDAI_TENSOR_HPP

#include <vector>
#include <string>
#include <cstddef>
#include <initializer_list>
#include <numeric>

namespace almondai {

class Tensor {
public:
    Tensor() = default;
    explicit Tensor(std::vector<std::size_t> shape_);
    Tensor(std::vector<std::size_t> shape_, std::vector<double> data_);
    Tensor(std::initializer_list<double> values);

    const std::vector<std::size_t> &shape() const noexcept;
    const std::vector<double> &data() const noexcept;
    std::vector<double> &data() noexcept;
    std::size_t size() const noexcept;

    double &operator[](std::size_t idx);
    double operator[](std::size_t idx) const;

    void fill(double value);
    Tensor matmul(const Tensor &other) const;
    Tensor add(const Tensor &other) const;
    Tensor scale(double factor) const;
    Tensor transpose(std::size_t rows, std::size_t cols) const;

private:
    std::vector<std::size_t> m_shape{};
    std::vector<double> m_data{};
};

Tensor outer_product(const std::vector<double> &a, const std::vector<double> &b);
double dot(const std::vector<double> &a, const std::vector<double> &b);

} // namespace almondai

#endif // ALMONDAI_TENSOR_HPP
