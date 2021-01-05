#pragma once

#include "../autograd/node.hpp"


namespace czy{
namespace loss{

    node& hinge(node& y, node& yh)
    {   
        return ((1 + (-y*yh)).relu()).sum();
    }

    node& binary_cross_entropy(node& y, node& yhat)
    {
        //−(𝑦log(𝑝)+(1−𝑦)log(1−𝑝))
        return -(  (y*yhat.log()) + ((1 -y)*(1-yhat).log()) );
    }

}//namespace loss
}//namespace czy



