#pragma once
#include "tensor.hpp"


namespace czy{


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

template <typename T>
bool tensor<T>::operator==(const tensor<T>& other)
{
    if(m_size != other.m_size) return false;
    if(other.m_shape != m_shape) return false;

    for(size_t i=0; i<m_size;i++)
    {
        if(m_data[i] != other.m_data[i]) return false;
    }
    return true;
}

template <typename T>
bool tensor<T>::operator!=(const tensor<T>& other)
{
    return ! (*this==other);
}


//adding
template <typename T>
tensor<T> tensor<T>::operator+(const tensor<T>& y)
{
    return broadcast( y, [](T& a, T& b){
        return a+b;
    });
}

template <typename T>
tensor<T> operator+(T x, const tensor<T>& y)
{
    return (tensor<T>) x + y;
}

//subbing
template <typename T>
tensor<T> tensor<T>::operator-(const tensor<T>& y)
{
    return broadcast(y, [](T& a, T& b){
        return a-b;
    });
}

template <typename T>
tensor<T> operator-(T x, const tensor<T>& y)
{
    return (tensor<T>) x - y;
}

//multiplying
template <typename T>
tensor<T> tensor<T>::operator*(const tensor<T>& y)
{
    return broadcast(y, [](T& a, T& b){
        return a*b;
    });
}

template <typename T>
tensor<T> operator*(T x, const tensor<T>& y)
{
    return y * x;
}

// //dividing

template <typename T>
tensor<T> operator/(T x, const tensor<T>& y)
{
    return (tensor<T>) x / y;
}

template <typename T>
tensor<T> tensor<T>::operator/(const tensor<T>& y)
{
    return broadcast(y, [](T& a, T& b){
        return a/b;
    });
}

template <typename T>
tensor<T> tensor<T>::operator-()
{
    return unary_operation([](T& a){return -a;});
}


}//namespace czy