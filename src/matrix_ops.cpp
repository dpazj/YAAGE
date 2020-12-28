#include "matrix_ops.h"

#include <stdexcept>
#include <iostream>

#include <math.h>

namespace op
{

Matrix Add(const Matrix& a, const Matrix& b)
{
    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
    {
        throw std::runtime_error("add: Tensors not of the same shape!");
    }

    Matrix c(a.Rows(), a.Columns());

    double* a_data = a.Data();
    double* b_data = b.Data();
    double* c_data = c.Data();

    for(size_t i=0; i<a.Size(); i++)
    {
        c_data[i] = a_data[i] + b_data[i];
    }
    return c;
}

Matrix Sub(const Matrix& a, const Matrix& b)
{
    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
    {
        throw std::runtime_error("Sub: Tensors not of the same shape!");
    }

    Matrix c(a.Rows(), a.Columns());

    double* a_data = a.Data();
    double* b_data = b.Data();
    double* c_data = c.Data();

    for(size_t i=0; i<a.Size(); i++)
    {
        c_data[i] = a_data[i] - b_data[i];
    }
    return c;
}

Matrix Mul(const Matrix& a, const Matrix& b)
{
    if(a.Columns() != b.Columns() || a.Rows() != b.Rows())
    {
        throw std::runtime_error("Mul: Tensors not of the same shape!");
    }

    Matrix c(a.Rows(), a.Columns());

    double* a_data = a.Data();
    double* b_data = b.Data();
    double* c_data = c.Data();

    for(size_t i=0; i<a.Size(); i++)
    {
        c_data[i] = a_data[i] * b_data[i];
    }
    return c;
}

Matrix Mul(const Matrix& a, double b)
{
   
    Matrix c(a.Rows(), a.Columns());

    double* a_data = a.Data();
    double* c_data = c.Data();

    for(size_t i=0; i<c.Size(); i++)
    {
        c_data[i] = a_data[i] * b;
    }
    return c;
}

Matrix Dot(const Matrix& a, const Matrix& b)
{
    size_t M = a.Rows();
    size_t K = a.Columns();
    size_t N = b.Columns();

    if(a.Columns() != b.Rows())
    {
        throw std::runtime_error("dot: Tensors a rows != Matrix b columns!");
    }
    Matrix c(M,N);
    double* a_data = a.Data();
    double* b_data = b.Data();
    double* c_data = c.Data();

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

Matrix Pow(const Matrix& a, double exp)
{
    
    Matrix c(a.Rows(), a.Columns());
    double* a_data = a.Data();
    double* c_data = c.Data();

    for(size_t i=0; i<a.Size(); i++)
    {
        c_data[i] = pow(a_data[i],exp);
    }
    return c;
}

Matrix Max(const Matrix& a, double val)
{
    Matrix c(a.Rows(), a.Columns());
    double* a_data = a.Data();
    double * c_data = c.Data();
    for(size_t i=0; i<a.Size(); i++)
    {
        c_data[i] = std::max(a_data[i], val);
    }
    return c;
}

Matrix Transpose(const Matrix& a)
{
    Matrix a_t(a.Columns(), a.Rows());
    double * a_t_data = a_t.Data();
    double * a_data = a.Data();

    for(size_t i=0; i<a.Rows(); i++)
    {
        size_t a_offset = i * a.Columns();
        for (size_t j = 0; j < a.Columns(); j++)
        {
            a_t_data[j * a_t.Columns() +  i] = a_data[a_offset + j];
        }
        
    }
    return a_t;
}

} //namespace op
