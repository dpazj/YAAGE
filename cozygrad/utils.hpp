#pragma once

#include <random> 
#include <algorithm>

std::mt19937 gen(1337);

namespace czy{
namespace utils
{
    template <typename T>
    T get_rand_double(T min, T max)
    {
        std::uniform_real_distribution<T> dis(min, max);
        return dis(gen);
    }


    

    


} // namespace utils
}//namespace czy



