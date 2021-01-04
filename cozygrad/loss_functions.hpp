#pragma once

#include "node.hpp"

namespace czy{

    node& hinge(node& y, node& yh)
    {   
        return ((1 + (-y*yh)).relu()).sum();
    }


}//namespace czy



