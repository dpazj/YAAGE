#pragma once

#include <random> 
#include <algorithm>
#include <iostream>

std::mt19937 gen;

namespace czy{
namespace utils
{

    void set_random_seed(unsigned int seed_val)
    {
        gen.seed(seed_val);
    }


    template <typename T>
    T get_rand_double(T min, T max)
    {
        std::uniform_real_distribution<T> dis(min, max);
        return dis(gen);
    }


    template <typename T>
    std::string vec_to_string(const std::vector<T>& vec)
    {
        std::string out; 
        for(const auto& x : vec)
        {
            out += std::to_string(x) + " ";
        }
        return out;
    }

    template <typename T>
    void print_vec(const std::vector<T>& vec)
    {
        std::cout << vec_to_string(vec) << std::endl;
    }
    


} // namespace utils
}//namespace czy



