#pragma once

#include "tensor.hpp"

#include <stdexcept>
#include <cmath>

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



// tensor sum(const tensor& a) 
// {
//     tensor c(1,1);
//     double* a_data = a.data();
//     double* c_data = c.data();
//     double acc = 0.0f;
//     for(size_t i=0; i < a.rows(); i++)
//     {
//         size_t offset = a.columns() * i;
//         for(size_t j=0; j < a.columns(); j++)
//         {
//             acc += a_data[offset + j];
//         }
//     }
//     c_data[0] = acc;
//     return c;
// }



// tensor max(const tensor& a, double y)
// {
//     tensor c(a.rows(), a.columns());
//     double* a_data = a.data();
//     double * c_data = c.data();
//     for(size_t i=0; i<a.size(); i++)
//     {
//         c_data[i] = std::max(a_data[i], y);
//     }
//     return c;
// }

// tensor pow(const tensor& a, double e)
// {
//     tensor c(a.rows(), a.columns());
//     double* a_data = a.data();
//     double * c_data = c.data();
//     for(size_t i=0; i<a.size(); i++)
//     {
//         c_data[i] = std::pow(a_data[i], e);
//     }
//     return c;
// }

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

