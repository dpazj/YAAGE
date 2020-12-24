#include "tensor.h"
#include <iostream>



Tensor::Tensor(std::initializer_list<std::initializer_list<double>> il)
{
    m_rows = il.size();
    m_columns = il.begin()->size();

    m_data = new double*[m_rows];

    size_t i = 0;
    for(auto row : il)
    {
        m_data[i] = new double[m_columns];
        //init with zeros
        size_t j = 0;
        for(auto col : row)
        {
            m_data[i][j] = col;
            j++;
        }
        i++;
    }
}

Tensor::Tensor(size_t m, size_t n)
{
    m_rows = m;
    m_columns = n;
    m_data = new double*[m];

    //allocate memory
    for(size_t i=0; i<m_rows; i++)
    {
        m_data[i] = new double[n];
        //init with zeros
        for(size_t j=0; j<m_columns;j++)
        {
            m_data[i][j] = 0.0f;
        }
    }
}

Tensor::~Tensor()
{
    for(size_t i=0; i< m_rows; i++)
    {
        delete[] m_data[i];
    }
    delete[] m_data;
}

size_t Tensor::GetColums(){return m_columns;}
size_t Tensor::GetRows(){return m_rows;}

double* Tensor::operator[](size_t i)
{
    return m_data[i];
}

void Tensor::Print()
{
    for(size_t i=0;i<m_rows;i++)
    {
        for(size_t j=0; j<m_columns;j++)
        {
            std::cout << m_data[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

