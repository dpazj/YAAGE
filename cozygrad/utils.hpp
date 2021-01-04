#pragma once

#include "session.hpp"
#include "node.hpp"

#include <random> 

std::mt19937 gen(1337);

class Session;
class node;

namespace utils
{
    // void clean_session()
    // {
    //     Session& session = Session::get_session();
    //     for(node* x : session.get_session_nodes())
    //     {
    //         delete x;
    //     }
    // } 


    double get_rand_double(double min, double max)
    {
        std::uniform_real_distribution<double> dis(min, max);
        return dis(gen);
    }

} // namespace utils


