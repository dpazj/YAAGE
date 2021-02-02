#pragma once

#include <random> 
#include <algorithm>
#include <iostream>

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


    template <typename T>
    void print_vec(const std::vector<T>& vec)
    {
        for(const auto& x : vec)
        {
            std::cout << x << " ";
        }
        std::cout << std::endl;
    }
    


} // namespace utils
}//namespace czy



