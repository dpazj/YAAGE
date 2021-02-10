#pragma once

#include "tensor_shape.hpp"

namespace czy{

//returns the resulting tensor shape of the broadcast, also prepends x_shape and y_shape with ones
tensor_shape calculate_dot_broadcast_shape(tensor_shape& x, tensor_shape& y)
{
    if(x.size() < 2 || y.size() < 2)
    {
        throw std::runtime_error("dot: tensors must have a rank of at least 2!");
    }

    if(x[x.size() - 1] != y[y.size() - 2]) //x col != y row
    {
        throw std::runtime_error("dot: last dim of x must be the same size as the second last dim of y!");
    }

    tensor_shape out_shape;
    size_t n_dims = std::max(x.size(), y.size());

    auto prepend_ones = [](tensor_shape& x, size_t dims){
        while(x.size() < dims){x.insert(x.begin(), 1);}
        return x;
    };
    
    x = prepend_ones(x, n_dims);
    y = prepend_ones(y, n_dims);

    for(size_t i=0; i < n_dims-2; i++)
    {
        if( (x[i] != 1 && y[i] != 1) && (x[i] != y[i]))
        {
            throw std::runtime_error("Unbroadcastable shapes!");
        }
        out_shape.push_back(std::max(x[i], y[i]));
    }

    out_shape.insert(out_shape.end(), x[x.size() - 2]);
    out_shape.insert(out_shape.end(), y[y.size() - 1]);

    return out_shape;
}

std::vector<size_t> calculate_dimension_offsets(tensor_shape& shape)
{
    size_t acc = 1;
    std::vector<size_t> c;
    for(size_t i=shape.size(); i > 0; i--)
    {
        acc *= shape[i-1];
        c.push_back(acc);
    }
    std::reverse(c.begin(), c.end());
    return c;
}

//returns the resulting tensor shape of the broadcast, also prepends x_shape and y_shape with ones
tensor_shape calculate_broadcast_shapes(tensor_shape& x_shape, tensor_shape& y_shape)
{
    tensor_shape out_shape;
    size_t n_dims = std::max(x_shape.size(), y_shape.size());

    auto prepend_ones = [](tensor_shape& x, size_t dims){
        while(x.size() < dims){x.insert(x.begin(), 1);}
        return x;
    };

    x_shape = prepend_ones(x_shape, n_dims);
    y_shape = prepend_ones(y_shape, n_dims);

    for(size_t i=0; i < n_dims; i++)
    {
        if( (x_shape[i] != 1 && y_shape[i] != 1) && (x_shape[i] != y_shape[i]) )
        {
            throw std::runtime_error("Unbroadcastable shapes! got x:" + utils::vec_to_string(x_shape) + " and y:" + utils::vec_to_string(y_shape) );
        }
        out_shape.push_back(std::max(x_shape[i], y_shape[i]));
    }
    return out_shape;
}

}//namespace czy