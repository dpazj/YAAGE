#pragma once

typedef unsigned long size_t;

#include <initializer_list>

class Tensor
{
    public:
        Tensor(size_t m, size_t n=1);
        Tensor(std::initializer_list<std::initializer_list<double>> il);
        Tensor();
        ~Tensor();

        size_t GetColums();
        size_t GetRows();

        void Print();

        double* operator[](size_t i);
        
    private:
        double** m_data;
        size_t m_rows;
        size_t m_columns;
};