#pragma once

#include "../autograd/node.hpp"


namespace czy{
namespace loss{

    template <typename T>
    node<T>& hinge(node<T>& y, node<T>& yhat)
    {   
        return (( (T) 1 + (-y*yhat)).relu()).sum();
    }

    template <typename T>
    node<T>& binary_cross_entropy(node<T>& y, node<T>& yhat)
    {
        //−(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
        auto& a = y * yhat.log();
        auto& b = ( (T)1 - y) * ( (T)1 - yhat).log();
        auto& c = a + b;
        auto& mean = c.mean();

        return -mean;
    }


    // template <typename T>
    // node<T>& categorical_cross_entropy(node<T>& y, node<T>& yhat)
    // {
    //     return -((y * yhat.log()).sum());
    // }

    template <typename T>
    node<T>& mean_squared_error(node<T>& y, node<T>& yhat)
    {
        return (y - yhat).pow(2).mean();
    }

}//namespace loss
}//namespace czy



