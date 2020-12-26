#include "tensor.h"
#include <iostream>
#include <cstring>

Tensor::Tensor(const Tensor& tensor)
{
    m_rows = tensor.m_rows;
    m_columns = tensor.m_columns;
    m_size = tensor.m_size;
    std::memcpy(m_data, tensor.m_data, m_size * sizeof(double));
}

Tensor::Tensor(std::initializer_list<std::initializer_list<double>> il)
{
    m_rows = il.size();
    m_columns =  il.begin()->size();
    m_size = m_rows * m_columns;
    m_data = new double[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        size_t j = 0;
        if(row.size() != m_columns){
            throw std::runtime_error("Tensor rows must be of the same length!");
        }
        for(auto col : row)
        {
            m_data[(i * m_columns) + j] = col;
            j++;
        }
        i++;
    }
}

Tensor::Tensor(std::initializer_list<double> il)
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

Tensor::Tensor(size_t m, size_t n, double init_val)
{
    m_rows = m;
    m_columns = n;
    m_size = m_rows * m_rows;
    m_data = new double[m_size];
    //init zeros
    for(size_t i=0; i<m_size; i++)
    {
        m_data[i] = init_val;
    }
}

Tensor::~Tensor()
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }
}

size_t Tensor::Columns(){return m_columns;}
size_t Tensor::Rows(){return m_rows;}
size_t Tensor::Size(){return m_size;}
double* Tensor::Data(){return m_data;}


void Tensor::Print()
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
    std::cout << std::endl;
}

double* Tensor::operator[](size_t i) //get the row
{
    return m_data + (i * m_columns);
}

Tensor& Tensor::operator=(const Tensor& rhs)
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
