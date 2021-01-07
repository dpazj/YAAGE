#pragma once

#include "../autograd/node.hpp"


namespace czy{
namespace loss{

    node& hinge(node& y, node& yhat)
    {   
        return ((1 + (-y*yhat)).relu()).sum();
    }

    node& binary_cross_entropy(node& y, node& yhat)
    {
        //−(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
        return -(  (y*yhat.log()) + ((1 -y)*(1-yhat).log()) );
    }

    node& mean_squared_error(node& y, node& yhat)
    {

       // double n = y.data()->rows(); // change this in future :) 
        return (y - yhat).pow(2).mean();// / n;
    }

}//namespace loss
}//namespace czy



