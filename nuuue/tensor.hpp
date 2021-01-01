#pragma once

#include "tensor_shape.hpp"

#include <initializer_list>
#include <memory>
#include <iostream>
#include <cstring>

class tensor
{
    public:
        tensor();
        tensor(const tensor& x);
        tensor(size_t m, size_t n);
        tensor(std::initializer_list<double> il);
        tensor(std::initializer_list<std::initializer_list<double>> il);
        ~tensor();

        double* data() const;
        size_t size() const;
        void print();

        //we need to change this to shape at some point
        size_t columns() const;
        size_t rows() const;

        tensor& operator=(const tensor& rhs);
        tensor operator+(const tensor& rhs);
        tensor operator-(const tensor& rhs);
        tensor operator-();
        tensor operator*(const tensor& rhs);
        tensor operator*(const double rhs);
        tensor operator>(double val);
        tensor operator<(double val);

    private:

        double* m_data = nullptr;
        size_t m_size;
        size_t m_rows;
        size_t m_columns;
};

tensor::~tensor()
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }
}

tensor::tensor()
{
    //so we can create uninitialized tensors that can be given life later :)
}

tensor::tensor(const tensor& x)
{
    *this = x;
}

tensor::tensor(size_t m, size_t n)
{
    m_rows = m;
    m_columns = n;
    m_size = m * n;
    m_data = new double[m_size];
}

tensor::tensor(std::initializer_list<double> il)
{
 
    m_rows = 1;
    m_columns = il.size();
    m_size = m_rows * m_columns;
    m_data = new double[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        m_data[i] = row;
        i++;
    }
}

tensor::tensor(std::initializer_list<std::initializer_list<double>> il)
{
    m_rows = il.size();
    m_columns = il.begin()->size();
    m_size = m_rows * m_columns;
    
    m_data = new double[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        size_t j = 0;
        if(row.size() != m_columns){
            throw std::runtime_error("init list rows must be of the same length!");
        }
        for(auto col : row)
        {
            m_data[(i * m_columns) + j] = col;
            j++;
        }
        i++;
    }
}

double* tensor::data() const {return m_data;}
size_t tensor::size() const {return m_size;}  
size_t tensor::rows() const {return m_rows;}  
size_t tensor::columns() const {return m_columns;}  

void tensor::print()
{


    for(size_t i=0;i<m_rows;i++)
    {
        size_t offset = i * m_columns;
        for(size_t j=0; j<m_columns;j++)
        {
            std::cout << m_data[offset + j] << " ";
        }
        std::cout << std::endl;
    }
}

//operators
tensor& tensor::operator=(const tensor& rhs)
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }

    m_rows = rhs.m_rows;
    m_columns = rhs.m_columns;
    m_size = rhs.m_size;
    m_data = new double[m_size];
    std::memcpy(m_data, rhs.m_data, m_size * sizeof(double));
    return *this;
}

tensor tensor::operator+(const tensor& rhs)
{
    if(m_size != rhs.m_size)
    {
        throw std::runtime_error("tensor shapes not the same");
    }

    tensor out(m_rows, m_columns);
    double* out_data = out.data();
    double* rhs_data = rhs.data();

    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] + rhs_data[i];
    }
    return out;
}

tensor tensor::operator-(const tensor& rhs)
{
    if(m_size != rhs.m_size)
    {
        throw std::runtime_error("tensor shapes not the same");
    }

    tensor out(m_rows, m_columns);
    double* out_data = out.data();
    double* rhs_data = rhs.data();

    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] - rhs_data[i];
    }
    return out;
}

tensor tensor::operator-()
{

    tensor out(m_rows, m_columns);
    double* out_data = out.data();

    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = -this->m_data[i];
    }
    return out;
}

tensor tensor::operator*(const tensor& rhs)
{
    if(m_size != rhs.m_size)
    {
        throw std::runtime_error("tensor shapes not the same");
    }

    tensor out(m_rows, m_columns);
    double* out_data = out.data();
    double* rhs_data = rhs.data();

    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] * rhs_data[i];
    }
    return out;
}

tensor tensor::operator*(double rhs)
{
    tensor out(m_rows, m_columns);
    double* out_data = out.data();

    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] * rhs;
    }
    return out;
}

tensor tensor::operator>(double val)
{
    tensor out(m_rows, m_columns);
    double* out_data = out.data();
    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] > val;
    }
    return out;
}

tensor tensor::operator<(double val)
{
    tensor out(m_rows, m_columns);
    double* out_data = out.data();
    for(size_t i=0; i < m_size; i++)
    {
        out_data[i] = this->m_data[i] < val;
    }
    return out;
}