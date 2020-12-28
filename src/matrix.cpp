#include "matrix.h"
#include <iostream>
#include <cstring>

Matrix::Matrix(const Matrix& matrix)
{
    m_rows = matrix.m_rows;
    m_columns = matrix.m_columns;
    m_size = matrix.m_size;
    std::memcpy(m_data, matrix.m_data, m_size * sizeof(double));
}

Matrix::Matrix(std::initializer_list<std::initializer_list<double>> il)
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
            throw std::runtime_error("Matrix rows must be of the same length!");
        }
        for(auto col : row)
        {
            m_data[(i * m_columns) + j] = col;
            j++;
        }
        i++;
    }
}

Matrix::Matrix(std::initializer_list<double> il)
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

Matrix::Matrix(size_t m, size_t n)
{
    m_rows = m;
    m_columns = n;
    m_size = m_rows * m_rows;
    m_data = new double[m_size];
}

Matrix::Matrix(size_t m, size_t n, double init_val)
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

Matrix::~Matrix()
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }
}

size_t Matrix::Columns()const{return m_columns;}
size_t Matrix::Rows()const{return m_rows;}
size_t Matrix::Size()const{return m_size;}
double* Matrix::Data()const{return m_data;}


void Matrix::Print()
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


Matrix& Matrix::operator=(const Matrix& rhs)
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
