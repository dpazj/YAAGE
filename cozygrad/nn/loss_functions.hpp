#pragma once

#include "../autograd/node.hpp"


namespace czy{
namespace nn{
namespace loss{

    autograd::node& hinge(autograd::node& y, autograd::node& yh)
    {   
        return ((1 + (-y*yh)).relu()).sum();
    }
    
}//namespace loss
}//namespace nn
}//namespace czy



