#pragma once

#include "tensor_broadcasting_utils.hpp"
#include "tensor.hpp"


#include <stdexcept>
#include <cmath>
#include <algorithm>

namespace czy{
namespace op{



//UNARY OPS
template <typename T>
tensor<T> exp(const tensor<T>& a)
{
    return a.unary_operation([](T a){return std::exp(a);});
}

template <typename T>
tensor<T> log(const tensor<T>& a)
{
    return a.unary_operation([](T a){return std::log(a);});
}

//BROADCAST OPS
//max
template <typename T>
tensor<T> max(const tensor<T>& x, const tensor<T>& y)
{
    return x.broadcast( y, [](T& a, T& b)
    {
        return std::max(a, b);
    });
}

template <typename T>
tensor<T> max(const tensor<T>& x, T y)
{
    return max(x, tensor<T>(y));
}

template <typename T>
tensor<T> max(T x,const tensor<T>& y)
{
    return max(tensor<T>(x), y);
}

//pow
template <typename T>
tensor<T> pow(const tensor<T>& x, const tensor<T>& y)
{
    return x.broadcast(y, [](T a, T b)
    {
        return std::pow(a, b);
    });
}

template <typename T>
tensor<T> pow(const tensor<T>& x, T y)
{
    return pow(x, tensor<T>(y));
}

template <typename T>
tensor<T> pow(T x,const tensor<T>& y)
{
    return pow(tensor<T>(x), y);
}

//reduce ops
template <typename T>
tensor<T> sum(const tensor<T>& x) 
{
    T* data_ptr = x.data();
    T acc = 0;
    for(size_t i=0; i<x.size();i++)
    {
        acc += data_ptr[i];
    }
    return tensor<T>(acc);
}

template <typename T>
tensor<T> sum(const tensor<T>& x, unsigned int axis, bool reshape = true) //reshape removes the summed axis from the resulting tensor e.g (5,5,3) summed on axis 1 -> (5,3)
{ 
    tensor_shape out_shape = x.shape();
    tensor_shape x_shape = x.shape();

    unsigned int max_axis = x_shape.size() -1;
    if(axis > max_axis)
    {
        throw std::runtime_error("axis " + std::to_string(axis) + " out of bounds for tensor of dimension " + std::to_string(max_axis + 1));
    }

    out_shape[axis] = 1;

    tensor<T> out(out_shape);
    std::vector<size_t> x_offsets = calculate_dimension_offsets(x_shape);
    std::vector<size_t> o_offsets = calculate_dimension_offsets(out_shape);
    
    T* x_data = x.data();

    size_t out_idx = 0;
    size_t x_offset = 0;

    size_t ets = x_shape[axis] / out_shape[axis];
    size_t ets_offset = axis == max_axis ? 1 : x_offsets[axis + 1];
    
    std::function<void(unsigned int)> recursive_sum_dimension = [&](unsigned int dim)
    {
        if(dim == axis)
        {
            for(size_t i=0; i < o_offsets[axis]; i++)
            {
                T acc = 0;
                for(size_t j=0; j < ets; j++)
                {
                    size_t offset = (ets_offset * j) + i + x_offset;
                    acc += x_data[offset];
                }
                out[out_idx] = acc;
                out_idx++;
            }
            return;
        }

        for(size_t i=0; i<x_shape[dim]; i++)
        {
            recursive_sum_dimension(dim+1);
            x_offset += x_offsets[dim+1];
        }
        x_offset -= x_offsets[dim];
    };
    recursive_sum_dimension(0);

    if(reshape)
    {
        out_shape.erase(out_shape.begin() + axis);
        out.reshape(out_shape);
    }
    
    return out;
}

template <typename T>
tensor<T> sum(const tensor<T>& x, std::vector<unsigned int> axes) 
{
    tensor_shape out_shape = x.shape();
    tensor_shape x_shape = x.shape();
    unsigned int max_axis = out_shape.size() - 1;

    for(const auto& x : axes)
    {
        if(x > max_axis)
        {
            throw std::runtime_error("axis " + std::to_string(x) + " out of bounds for tensor of dimension " + std::to_string(max_axis + 1));
        }
        out_shape[x] = 1;
    }
    tensor<T> out = sum(x, axes[0],false);
    axes.erase(axes.begin());
    
    for(const auto& axis : axes)
    {
        out = sum(out, axis, false);        
    }

    //this assumes that each dimension's shape is not zero - which it cant be, but might want to change in future!
    tensor_shape new_shape;
    for(const auto& axis : axes)
    {
        out_shape[axis] = 0;
    }
    for(const auto& x: out_shape)
    {
        if(x != 0)
        {
            new_shape.push_back(x);
        }
    }
    out.reshape(new_shape);
    return out;
}


template <typename T>
tensor<T> dot(const tensor<T>& x, const tensor<T>& y)
{
    auto print_vec = [](tensor_shape to_print){
        for(const auto& x: to_print){
            std::cout << x << " ";
        }
        std::cout << std::endl;
    };

    tensor_shape x_shape = x.shape();
    tensor_shape y_shape = y.shape();
    tensor_shape out_shape = calculate_dot_broadcast_shape(x_shape,y_shape);

    tensor<T> out(out_shape);
    print_vec(out_shape);

    size_t M = x_shape[x_shape.size() - 2]; //x rows
    size_t K = x_shape[x_shape.size() - 1]; //x cols
    size_t N = y_shape[y_shape.size() - 1]; //y cols

    std::vector<size_t> x_dim_offsets = calculate_dimension_offsets(x_shape);     
    std::vector<size_t> y_dim_offsets = calculate_dimension_offsets(y_shape);    
    std::vector<size_t> o_dim_offsets = calculate_dimension_offsets(out_shape);    

    size_t x_offset, y_offset, o_offset; 
    x_offset = y_offset = o_offset = 0;

    T* x_data = x.data();
    T* y_data = x.data();
    T* o_data = out.data();

    std::function<void(unsigned int)> recursive_dot_broadcast = [&](unsigned int dim)
    {
        if(dim == out_shape.size() - 2)
        {
            size_t tmp_x = 0;
            size_t tmp_o = 0;
            for(size_t i=0; i<M;i++)
            {
                
                for (size_t j = 0; j < N; j++)
                {
                    T acc = 0.0f;
                    //size_t tmp_x = K * i;

                    for (size_t k = 0; k < K; k++)
                    {
                        acc += x_data[tmp_x + k + x_offset] * y_data[k * N + j + y_offset];
                    }

                    o_data[tmp_o + j + o_offset] = acc;
                }
                tmp_x+=K;
                tmp_o+=N;
            }
            return;
        }

        for(size_t i=0; i<out_shape[dim]; i++)
        {
            recursive_dot_broadcast(dim+1);
            x_offset += x_dim_offsets[dim+1] * !(1 == x_shape[dim]); //if shape is one multiply by 0      
            y_offset += y_dim_offsets[dim+1] * !(1 == y_shape[dim]);           
            o_offset += o_dim_offsets[dim+1]; 
        }
        x_offset -= x_dim_offsets[dim] * !(1 == x_shape[dim]);        
        y_offset -= y_dim_offsets[dim] * !(1 == y_shape[dim]);           
        o_offset -= o_dim_offsets[dim];     
    };
    recursive_dot_broadcast(0);

    

    


    return out;
}

// tensor dot(const tensor& a, const tensor& b)
// {
//     size_t M = a.rows();
//     size_t K = a.columns();
//     size_t N = b.columns();

//     if(a.columns() != b.rows())
//     {
//         throw std::runtime_error("dot: Matrix a rows != Matrix b columns! got: " + std::to_string(a.columns()) + " and " + std::to_string(b.rows()));
//     }
//     tensor c(M,N);
//     double* a_data = a.data();
//     double* b_data = b.data();
//     double* c_data = c.data();

//     for(size_t i=0; i<M;i++)
//     {
//         for (size_t j = 0; j < N; j++)
//         {
//             double acc = 0.0f;
//             size_t a_offset = K * i;

//             for (size_t k = 0; k < K; k++)
//             {
//                 acc += a_data[a_offset + k] * b_data[k * N + j];
//             }
//             c_data[(i*N) + j] = acc;
//         }
//     }
//     return c;
// }

// tensor transpose(const tensor& a)
// {
//     tensor a_t(a.columns(), a.rows());
//     double* a_t_data = a_t.data();
//     double* a_data = a.data();

//     size_t rows = a.rows();
//     size_t columms = a.columns();

//     for(size_t i=0; i<rows; i++)
//     {
//         size_t a_offset = i * columms;
//         for (size_t j = 0; j < columms; j++)
//         {
//             a_t_data[j * rows +  i] = a_data[a_offset + j];
//         }
//     }
//     return a_t;
// }

}//namespace op
}//namespace czy

