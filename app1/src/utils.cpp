// utils.cpp
#include "../include/utils.hpp"

namespace safe_io
{
    // Implementation of out() function
    std::ostream& out() noexcept
    {
        return std::cout;
    }
}
