#pragma once
#include <iostream>


class Matrix
{
    public:
        Matrix(size_t m, size_t n=1);
        ~Matrix();

        size_t GetColums();
        size_t GetRows();

        void Print();

        double* operator[](size_t i);
        
    private:
        double** m_data;
        size_t m_rows;
        size_t m_columns;
};