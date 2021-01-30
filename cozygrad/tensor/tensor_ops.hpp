#pragma once

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
tensor<T> sum(const tensor<T>& x, std::initializer_list<unsigned int> axis_il) 
{
    tensor_shape out_shape = x.shape();
    tensor_shape x_shape = x.shape();
    unsigned int max_axis = out_shape.size() - 1;
    std::vector<unsigned int> axes = axis_il;
    std::sort(axes.begin(), axes.end());
    for(const auto& x : axes)
    {
        if(x > max_axis)
        {
            throw std::runtime_error("axis " + std::to_string(x) + " out of bounds for tensor of dimension " + std::to_string(max_axis + 1));
        }
        out_shape[x] = 1;
    }

    tensor<T> out(out_shape);

    auto x_offsets = x.calculate_dimension_offsets(x_shape);

    std::function<void(std::vector<size_t>&)> print_vec = [](std::vector<size_t>& to_print){for(const auto& x: to_print){std::cout << x << " ";} std::cout << std::endl;};
    print_vec(x_offsets);

    // for(const auto& axis : axes)
    // {
    //     // size_t offset = x_offsets[axis + 1];
    //     // size_t dim_shape = x_shape[axis];
    //     // size_t iterations = x_offsets[axis] / dim_shape;

    //     // T* x_data_ptr = x.data();

   // unsinged int axis = 0;


        
    // }


    return out;
}

template <typename T>
tensor<T> sum(const tensor<T>& x, unsigned int axis) 
{ 
    std::function<void(std::vector<size_t>&)> print_vec = [](std::vector<size_t>& to_print){for(const auto& x: to_print){std::cout << x << " ";} std::cout << std::endl;};



    tensor_shape out_shape = x.shape();
    tensor_shape x_shape = x.shape();

    unsigned int max_axis = x_shape.size() -1;
    if(axis > max_axis)
    {
        throw std::runtime_error("axis " + std::to_string(axis) + " out of bounds for tensor of dimension " + std::to_string(max_axis + 1));
    }

    out_shape[axis] = 1;

    tensor<T> out(out_shape);
    std::vector<size_t> x_offsets = x.calculate_dimension_offsets(x_shape);
    std::vector<size_t> o_offsets = x.calculate_dimension_offsets(out_shape);
    
    T* x_data = x.data();

    print_vec(x_offsets);
    print_vec(o_offsets);
    print_vec(x_shape);
    print_vec(out_shape);
    std::cout << std::endl;

    size_t out_idx = 0;
    size_t x_idx = 0;

    std::function<void(unsigned int)> test = [&](unsigned int dim)
    {
        if(dim == axis)
        {
            std::cout << "offset" << o_offsets[axis] << std::endl;
            for(size_t i=0;i<o_offsets[axis];i++)
            {
                T acc = 0;
                size_t j_offset = 0;
                for(size_t j=0; j<8;j++)
                {
                    acc += x_data[j_offset + x_idx];
                    std::cout << j_offset + x_idx << std::endl;
                    j_offset += 6; 
                }
                std::cout << "Break" << std::endl;
                out[out_idx] = acc;
                out_idx++;
                x_idx++;
            }


            return;
        }

        for(size_t i=0; i<x_shape[dim]; i++)
        {
            test(dim+1);
        }
    };
    test(0);

    return out;
}

// j is number of elements we are adding together
// offset_j is the offset between the numbers we are adding together

// [[[1, 1, 1], [2, 2, 2]], [[3, 3, 3], [4, 4, 4]], [[5, 5, 5], [6, 6, 6]], [[7, 7, 7], [8, 8, 8]]] (4,2,3)

// [[ 3,  6], [ 9, 12], [15, 18], [21, 24]] axis=2 | ets=3 etsoffset=1 i_offset= 0,3,9,12,15,18,21 

// [[ 3,  3,  3], [ 7,  7,  7],[11, 11, 11], [15, 15, 15]] axis=1 | ets=2 etsoffset=3 i_offset= 0,1,2, 6,7,8 , 12,13,14, 18,19,20 

// [[16, 16, 16],[20, 20, 20]] axis=0 ets=4 etsoffset=6 i_offset= 0,1,2,3,4,5 












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

