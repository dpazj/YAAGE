#pragma once

#include "tensor.hpp"

#include <stdexcept>
#include <math.h>

namespace op{

    tensor max(const tensor& a, double y);
    tensor pow(const tensor& a, double e);
    tensor dot(const tensor& a, const tensor& b);
    tensor transpose(const tensor& a);


    tensor max(const tensor& a, double y)
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

    tensor pow(const tensor& a, double e)
    {
        tensor c(a.rows(), a.columns());
        double* a_data = a.data();
        double * c_data = c.data();
        for(size_t i=0; i<a.size(); i++)
        {
            c_data[i] = std::pow(a_data[i], e);
        }
        return c;
    }

    tensor dot(const tensor& a, const tensor& b)
    {
        size_t M = a.rows();
        size_t K = a.columns();
        size_t N = b.columns();

        if(a.columns() != b.rows())
        {
            throw std::runtime_error("dot: Matrix a rows != Matrix b columns!");
        }
        tensor c(M,N);
        double* a_data = a.data();
        double* b_data = b.data();
        double* c_data = c.data();

        for(size_t i=0; i<M;i++)
        {
            for (size_t j = 0; j < N; j++)
            {
                double acc = 0.0f;

                size_t a_offset = K * i;

                for (size_t k = 0; k < K; k++)
                {
                    acc += a_data[a_offset + k] * b_data[k * N + j];
                }
                c_data[(i*N) + j] = acc;
            }
        }
        return c;
    }

    tensor transpose(const tensor& a)
    {
        tensor a_t(a.rows(), a.columns());
        double* a_t_data = a_t.data();
        double* a_data = a.data();
        size_t columms = a.columns();

        for(size_t i=0; i<a_t.rows(); i++)
        {
            size_t a_offset = i * columms;
            for (size_t j = 0; j < columms; j++)
            {
                a_t_data[j * columms +  i] = a_data[a_offset + j];
            }
        
        }
        return a_t;
    }

}