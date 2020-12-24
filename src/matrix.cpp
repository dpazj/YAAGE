#include "matrix.h"

Matrix::Matrix(size_t m, size_t n)
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

Matrix::~Matrix()
{
    for(size_t i=0; i< m_rows; i++)
    {
        delete[] m_data[i];
    }
    delete[] m_data;
}

size_t Matrix::GetColums(){return m_columns;}
size_t Matrix::GetRows(){return m_rows;}

double* Matrix::operator[](size_t i)
{
    return m_data[i];
}

void Matrix::Print()
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