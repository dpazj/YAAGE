#pragma once

typedef unsigned long size_t;

#include <initializer_list>

class Matrix
{
    public:
        Matrix(const Matrix& matrix);
        Matrix(size_t m, size_t n);
        Matrix(size_t m, size_t n, double init_val);
        Matrix(std::initializer_list<std::initializer_list<double>> il);
        Matrix(std::initializer_list<double> il);

        ~Matrix();

        size_t Columns();
        size_t Rows();
        size_t Size();

        double* Data();

        void Print();
        
        double* operator[](size_t i);
        Matrix& operator=(const Matrix& rhs);

    private:
        double* m_data = nullptr;
        size_t m_size;
        size_t m_rows;
        size_t m_columns;
};

