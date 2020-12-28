#pragma once

typedef unsigned long size_t;

#include <initializer_list>

class Tensor
{
    public:
        Tensor(const Tensor& tensor);
        Tensor(size_t m, size_t n);
        Tensor(size_t m, size_t n, double init_val);
        Tensor(std::initializer_list<std::initializer_list<double>> il);
        Tensor(std::initializer_list<double> il);

        ~Tensor();

        size_t Columns();
        size_t Rows();
        size_t Size();

        double* Data();

        void Print();
        
        double* operator[](size_t i);
        Tensor& operator=(const Tensor& rhs);

    private:
        double* m_data = nullptr;
        size_t m_size;
        size_t m_rows;
        size_t m_columns;
};

