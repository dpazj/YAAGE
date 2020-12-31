#pragma once

#include "tensor.hpp"

#include <stdexcept>
#include <math.h>

namespace op{

    tensor max(tensor& a, double y);



    tensor max(tensor& a, double y)
    {
        tensor c(a.rows(), a.columns());
        double* a_data = a.data();
        double * c_data = c.data();
        for(size_t i=0; i<a.size(); i++)
        {
            c_data[i] = std::max(a_data[i], y);
        }
        return c;
    }

}