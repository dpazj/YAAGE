#pragma once

#include <random> 

std::mt19937 gen(1337);

namespace czy{
namespace utils
{
    double get_rand_double(double min, double max)
    {
        std::uniform_real_distribution<double> dis(min, max);
        return dis(gen);
    }

} // namespace utils
}//namespace czy



