#pragma once

#include "../utils.hpp"

#include <initializer_list>
#include <memory>
#include <ostream>
#include <cstring>
#include <random>
#include <functional>


typedef unsigned long size_t;
typedef std::vector<size_t> tensor_shape;

namespace czy{


template <typename T>
class tensor
{
    public:
        tensor();
        tensor(const tensor& x);
        tensor(tensor_shape& shape);

        tensor(std::vector<char>& buf, std::initializer_list<size_t> shape);
        //tensor(std::vector<double> buf, std::initializer_list<size_t> shape);

        tensor(std::initializer_list<T> il);
        tensor(std::initializer_list<std::initializer_list<T>> il);
        tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> il);
        tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> il);

        ~tensor();

        void of_value(double val);
        void zeros();
        void ones();
        void random(double min = -1, double max = 1);

        T* data() const;
        size_t size() const;
        void print(std::ostream& os) const;
        void print_shape(std::ostream& os) const;

        tensor_shape shape() const;
        


        //operators
        template <typename TT>
        friend std::ostream& operator<<(std::ostream& os, const tensor<TT>& ten);

        T& operator[](size_t i);
        tensor<T>& operator=(const tensor<T>& rhs);

        // tensor operator+(const tensor& rhs);

        // tensor operator-(const tensor& rhs);
        // tensor operator-();

        // tensor operator*(const tensor& rhs);
        // tensor operator*(const double rhs);

        // tensor operator/(const tensor& rhs);
        // tensor operator/(const double rhs);

        // tensor operator>(double val);
        // tensor operator<(double val);

    private:

        size_t calculate_size();
        T* m_data = nullptr;
        size_t m_size = 0;
        std::vector<size_t> m_shape;

};

template <typename T>
tensor<T>::~tensor()
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }
}

template <typename T>
tensor<T>::tensor() : m_shape()
{

}

template <typename T>
tensor<T>::tensor(const tensor<T>& x)
{
    *this = x;
}

template <typename T>
tensor<T>::tensor(tensor_shape& shape)
{
    m_shape = shape;
    m_size = calculate_size();
    m_data = new T[m_size];
}

template <typename T>
tensor<T>::tensor(std::vector<char>& buf, std::initializer_list<size_t> shape)
{
    m_shape = shape;
    m_size = calculate_size();
    m_data = new T[m_size];

    if(buf.size() != m_size * sizeof(T))
    {
        throw std::runtime_error("Buffer size does not match size given by shape!");
    }

    std::memcpy(m_data, buf.data(),buf.size());
}

template <typename T>
tensor<T>::tensor(std::initializer_list<T> il)
{
    m_shape = {il.size()};
    m_size = calculate_size();
    m_data = new T[m_size];

    size_t i = 0;
    for(auto x : il)
    {
        m_data[i] = x;
        i++;
    }
}

template <typename T>
tensor<T>::tensor(std::initializer_list<std::initializer_list<T>> il)
{
    m_shape = {il.size(), il.begin()->size()};
    m_size = calculate_size();
    m_data = new T[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        size_t j = 0;
        if(row.size() != il.begin()->size()){
            throw std::runtime_error("init list rows must be of the same length!");
        }
        for(auto col : row)
        {
            m_data[(i * il.begin()->size()) + j] = col;
            j++;
        }
        i++;
    }
}

template <typename T>
tensor<T>::tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> il)
{
    m_shape = {il.size(), il.begin()->size(), il.begin()->begin()->size()};
    m_size = calculate_size();
    m_data = new T[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        size_t dim_1_offset = i * il.begin()->size() * il.begin()->begin()->size();
        size_t j = 0;
        if(row.size() != il.begin()->size()){
            throw std::runtime_error("init list rows must be of the same length!");
        }

        for(auto col : row)
        {
            size_t dim_2_offset = j * il.begin()->begin()->size();
            size_t k = 0;
            for(auto depth : col)
            {
                m_data[ dim_1_offset + dim_2_offset + k] = depth;
                k++;
            }
            j++;
        }
        i++;
    }
}

template <typename T>
tensor<T>::tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> il)
{
    m_shape = {il.size(), il.begin()->size(), il.begin()->begin()->size(), il.begin()->begin()->begin()->size()};
    m_size = calculate_size();
    m_data = new T[m_size];

    size_t i = 0;
    for(auto row : il)
    {
        size_t dim_1_offset = i * il.begin()->size() * il.begin()->begin()->size() * il.begin()->begin()->begin()->size();
        size_t j = 0;
        if(row.size() != il.begin()->size()){
            throw std::runtime_error("init list rows must be of the same length!");
        }
        for(auto col : row)
        {
            size_t dim_2_offset = j * il.begin()->begin()->size() * il.begin()->begin()->begin()->size();
            size_t k = 0;
            for(auto depth : col)
            {
                size_t l = 0;
                size_t dim_3_offset = k * il.begin()->begin()->begin()->size();
                for(auto time : depth)
                {
                    m_data[dim_1_offset + dim_2_offset + dim_3_offset + l] = time;
                }
                k++;
            }
            j++;
        }
        i++;
    }
}

template <typename T>
void tensor<T>::of_value(double val)
{
    for(size_t i=0; i< m_size; i++)
    {
        m_data[i] = val;
    }
}

template <typename T>
void tensor<T>::zeros(){of_value(0.0f);}

template <typename T>
void tensor<T>::ones(){of_value(1.0f);}

template <typename T>
void tensor<T>::random(double min, double max)
{
    for(size_t i=0; i< m_size; i++)
    {
        m_data[i] = utils::get_rand_double(min, max);
    }
}

template <typename T>
T* tensor<T>::data() const {return m_data;}
template <typename T>
size_t tensor<T>::size() const {return m_size;}
template <typename T>
tensor_shape tensor<T>::shape() const {return m_shape;};

template <typename T>
void tensor<T>::print(std::ostream& os) const
{
    size_t offset = 0;

    std::function<void(int)> print_tensor = [&, this](int idx)
    {
        os << '[';
        for(size_t i=0; i < m_shape[idx]; i++)
        {
            if(idx == (int) m_shape.size()-1)
            {
                os << m_data[offset];
                offset++;
                
            }
            else
            {
                print_tensor(idx + 1);
            }

            if(i < m_shape[idx] -1)
            {
                os << ", ";
            }
        }

        os << ']';
    };
    
    print_tensor(0);
}

template <typename T>
void tensor<T>::print_shape(std::ostream& os) const
{
    os << '(';
    for(size_t i =0; i < m_shape.size(); i++)
    {
        os << m_shape[i];
        if(i != m_shape.size() -1)
        {
            os << ",";
        }
    }
    os << ')';
}

template <typename T>
size_t tensor<T>::calculate_size()
{
    size_t size = 0;
    for(const auto& x : m_shape)
    {
        if(size == 0){size = 1;}
        size *= x;
    }
    return size;
}

//operators
template <typename T>
std::ostream& operator<<(std::ostream& os, const tensor<T>& ten)
{
    os << "tensor(";
    ten.print(os);
    os << ", shape=";
    ten.print_shape(os);
    os << ")";
    return os;
}

template <typename T>
T& tensor<T>::operator[](size_t i){return m_data[i];};

template <typename T>
tensor<T>& tensor<T>::operator=(const tensor<T>& rhs)
{
    if(m_data != nullptr)
    {
        delete[] m_data;
    }

    m_shape = rhs.m_shape;
    m_size = rhs.m_size;
    m_data = new T[m_size];
    std::memcpy(m_data, rhs.m_data, m_size * sizeof(T));
    return *this;
}

// //adding
// tensor tensor::operator+(const tensor& rhs)
// {
//     if(m_size != rhs.m_size)
//     {
//         throw std::runtime_error("add: tensor shapes not the same " + std::to_string(m_size) + " " +  std::to_string(rhs.m_size));
//     }
//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] + rhs_data[i];
//     }
//     return out;
// }

// tensor operator+(double lhs,const tensor& rhs)
// {
//     tensor out(rhs.rows(), rhs.columns());
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();
//     for(size_t i=0; i < rhs.size(); i++)
//     {
//         out_data[i] = lhs + rhs_data[i];
//     }
//     return out;
// }

// tensor operator+(const tensor& lhs, double rhs)
// {
//     return rhs + lhs;
// }

// //subbing
// tensor tensor::operator-(const tensor& rhs)
// {
//     if(m_size != rhs.m_size)
//     {
//         throw std::runtime_error("sub: tensor shapes not the same");
//     }

//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] - rhs_data[i];
//     }
//     return out;
// }

// tensor operator-(double lhs, const tensor& rhs)
// {
//     tensor out(rhs.rows(), rhs.columns());
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();
//     for(size_t i=0; i < rhs.size(); i++)
//     {
//         out_data[i] = lhs - rhs_data[i];
//     }
//     return out;
// }

// tensor operator-(const tensor& lhs, double rhs)
// {
//     tensor out(lhs.rows(), lhs.columns());
//     double* out_data = out.data();
//     double* rhs_data = lhs.data();
//     for(size_t i=0; i < lhs.size(); i++)
//     {
//         out_data[i] = rhs_data[i] - rhs;
//     }
//     return out;
// }

// tensor tensor::operator-()
// {
//     return *this * -1;
// }

// //multiplying
// tensor tensor::operator*(const tensor& rhs)
// {
//     if(m_size != rhs.m_size)
//     {
//         throw std::runtime_error("mul: tensor shapes not the same: got " + std::to_string(m_size) + " and " + std::to_string(rhs.m_size) );
//     }

//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] * rhs_data[i];
//     }
//     return out;
// }

// tensor tensor::operator*(double rhs)
// {
//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] * rhs;
//     }
//     return out;
// }

// tensor operator*(double lhs, tensor& rhs)
// {
//     return rhs * lhs;
// }

// //dividing
// tensor tensor::operator/(const tensor& rhs)
// {
//     if(m_size != rhs.m_size)
//     {
//         std::cout << m_size << " " << rhs.m_size << std::endl;
//         throw std::runtime_error("tensor shapes not the same");
//     }

//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] / rhs_data[i];
//     }
//     return out;
// }

// tensor tensor::operator/(double rhs)
// {
//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();

//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] / rhs;
//     }
//     return out;
// }

// tensor operator/(double lhs, const tensor& rhs)
// {
//     tensor out(rhs.rows(), rhs.columns());
//     double* out_data = out.data();
//     double* rhs_data = rhs.data();

//     for(size_t i=0; i < rhs.size(); i++)
//     {
//         out_data[i] = lhs / rhs_data[i];
//     }
//     return out;
// }

// //greater than
// tensor tensor::operator>(double val)
// {
//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] > val;
//     }
//     return out;
// }
// //less than
// tensor tensor::operator<(double val)
// {
//     tensor out(m_rows, m_columns);
//     double* out_data = out.data();
//     for(size_t i=0; i < m_size; i++)
//     {
//         out_data[i] = this->m_data[i] < val;
//     }
//     return out;
// }

}//namespace czy




