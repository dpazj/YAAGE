#include "matrix_ops.h"

#include <stdexcept>
#include <iostream>

#include <math.h>

namespace op
{

Matrix Add(Matrix& a, Matrix& b)
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

Matrix Sub(Matrix& a, Matrix& b)
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

Matrix Mul(Matrix& a, Matrix& b)
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

Matrix Mul(Matrix& a, double b)
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

Matrix Dot(Matrix& a, Matrix& b)
{
    size_t M = a.Rows();
    size_t K = a.Columns();
    size_t N = b.Columns();

    if(a.Columns() != b.Rows())
    {
        throw std::runtime_error("dot: Tensors a rows != Matrix b columns!");
    }

    Matrix c(M,N);

    for(size_t i=0; i<M;i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            double acc = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    return c;
}

Matrix Pow(Matrix& a, double exp)
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

Matrix Sum(Matrix& a)
{
    
    Matrix c(1, 1);
    double* a_data = a.Data();

    double acc = 0.0f;
    
    for(size_t i=0; i<a.Size(); i++)
    {
        acc += a_data[i];
    }
    c[0][0] = acc;
    return c;
}


Matrix Max(Matrix& a, double val)
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

} //namespace op
