#pragma once

#include "../utils.hpp"

#include "tensor_shape.hpp"
#include "tensor_broadcasting_utils.hpp"

#include <initializer_list>
#include <memory>
#include <ostream>
#include <cstring>
#include <random>
#include <functional>
#include <algorithm>

namespace czy{


template <typename T>
class tensor
{
    public:
        tensor();
        tensor(T x); 
        tensor(const tensor& x);
        tensor(tensor_shape& shape);

        tensor(std::vector<char>& buf, std::initializer_list<size_t> shape) : tensor(buf, tensor_shape(shape)){};
        tensor(std::vector<char>& buf, tensor_shape shape);

        tensor(std::initializer_list<T> il);
        tensor(std::initializer_list<std::initializer_list<T>> il);
        tensor(std::initializer_list<std::initializer_list<std::initializer_list<T>>> il);
        tensor(std::initializer_list<std::initializer_list<std::initializer_list<std::initializer_list<T>>>> il);

        ~tensor();

        void reshape(tensor_shape new_shape);
        tensor<T> slice(int start, int end=-1);

        tensor<T> broadcast(const tensor<T>& other, std::function<T(T&,T&)> operation) const;
        
        tensor<T> unary_operation(std::function<T(T&)> operation) const;
        void map(std::function<T(T&)> operation);
        void of_value(T val);
        void zeros();
        void ones();
        void random(T min = -1, T max = 1);

        T* data() const;
        size_t size() const;
        void print(std::ostream& os) const;
        void print_shape(std::ostream& os) const;

        tensor_shape shape() const;
        
        //operators - defined in tensor operators
        T& operator[](size_t i);
        tensor<T>& operator=(const tensor<T>& y);
        tensor<T>& operator=(T y);
        bool operator==(const tensor<T>& y);
        bool operator!=(const tensor<T>& y);

        tensor<T> operator+(const tensor<T>& y);
        tensor<T> operator-(const tensor& y);
        tensor<T> operator*(const tensor<T>& y);
        tensor<T> operator/(const tensor<T>& y);
        tensor<T> operator>(const tensor<T>& y);
        tensor<T> operator<(const tensor<T>& y);
    
        tensor<T> operator-();

        //Friends
        template <typename TT>
        friend std::ostream& operator<<(std::ostream& os, const tensor<TT>& ten);

    private:
        size_t calculate_size();


        T* m_data = nullptr;
        size_t m_size = 0;
        tensor_shape m_shape;

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
tensor<T>::tensor(T x)
{
    m_shape = {1};
    m_size = 1;
    m_data = new T[m_size];
    *m_data = x;
}; 


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
tensor<T>::tensor(std::vector<char>& buf, tensor_shape shape)
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
void tensor<T>::reshape(tensor_shape new_shape)
{
    tensor_shape old = m_shape;
    size_t old_size = m_size;

    m_shape = new_shape;
    m_size = calculate_size();

    if(m_size != old_size)
    {
        m_shape = old;
        m_size = calculate_size();
        throw std::runtime_error("New shape not valid!");
    }
}

//maybe change this to a view of the data instead so that we dont need to copy data?
template <typename T>
tensor<T> tensor<T>::slice(int start, int end)
{
    size_t start_idx = start;
    size_t end_idx = end; 
    if(end == -1){ end_idx = m_shape.front();}

    if(start_idx > m_shape.front() || end_idx > m_shape.front()){throw std::runtime_error("Slice index out of range!");}
    if(start_idx == end_idx){throw std::runtime_error("Slice start index cannot equal slice end index!");}
    if(start_idx > end_idx){throw std::runtime_error("Slice start index cannot be greater than slice end index!");}

    tensor_shape new_shape = m_shape;
    new_shape[0] = end_idx - start_idx;

    size_t offset = (m_size / m_shape[0]) * start_idx;

    tensor<T> tensor_slice(new_shape);
    std::memcpy(tensor_slice.m_data, m_data + offset, tensor_slice.m_size * sizeof(T));

    return tensor_slice;
}

template <typename T>
tensor<T> tensor<T>::unary_operation(std::function<T(T&)> operation) const
{
    tensor_shape new_shape = m_shape;
    tensor<T> out(new_shape);
    for(size_t i=0; i < out.m_size; i++)
    {
        out[i] = operation(m_data[i]);
    }
    return out;
}

template <typename T>
void tensor<T>::map(std::function<T(T&)> operation)
{
    for(size_t i=0; i< m_size; i++)
    {
        m_data[i] = operation(m_data[i]);
    }
}

template <typename T>
void tensor<T>::of_value(T val)
{
    map([&val](T&){return val;});
}

template <typename T>
void tensor<T>::zeros(){of_value(0);}

template <typename T>
void tensor<T>::ones(){of_value(1);}

template <typename T>
void tensor<T>::random(T min, T max)
{
    map([&min, &max](T&){return utils::get_rand_double(min, max);});
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
        if(i != m_shape.size()-1)
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
    if(m_shape.size() > 0) size = 1;
    for(const auto& x : m_shape)
    {
        size *= x;
    }
    return size;
}

template <typename T>
tensor<T> tensor<T>::broadcast(const tensor<T>& y, std::function<T(T&,T&)> operation) const
{
    tensor_shape x_shape = m_shape;
    tensor_shape y_shape = y.m_shape;
    tensor_shape out_shape = calculate_broadcast_shapes(x_shape,y_shape);

    tensor<T> out(out_shape);
    
    //see if we can do a few tricks to make the operation faster
    if(x_shape == y_shape)
    {
        for(size_t i=0; i < out.size(); i++)
        {
            T a = m_data[i];
            T b = y.m_data[i];
            out[i] = operation(a,b);
        }
    }
    else if(m_size == 1 || y.m_size == 1)
    {        
        for(size_t i=0; i < out.size(); i++)
        {
            T a = m_data[i % m_size];
            T b = y.m_data[i % y.m_size];
            out[i] = operation(a,b);
        }
    }
    else
    {
        std::vector<size_t> x_dim_offsets = calculate_dimension_offsets(x_shape);     
        std::vector<size_t> y_dim_offsets = calculate_dimension_offsets(y_shape);    
        std::vector<size_t> o_dim_offsets = calculate_dimension_offsets(out_shape);    

        size_t x_offset, y_offset, o_offset; 
        x_offset = y_offset = o_offset = 0;

        std::function<void(size_t)> recurse_broadcasting = [&](size_t dim){

            if(dim == out_shape.size() - 1)
            {
                for(size_t j=0; j<out_shape[dim];j++)
                {
                    T a = m_data[x_offset + (j % x_shape[dim])];
                    T b = y.m_data[y_offset + (j % y_shape[dim])];
                    out[o_offset + j] = operation(a,b);
                }
                return;
            }
            
            for(size_t i=0;i<out_shape[dim];i++)
            {   
                recurse_broadcasting(dim + 1);   
                x_offset += x_dim_offsets[dim+1] * !(1 == x_shape[dim]); //if shape is one multiply by 0      
                y_offset += y_dim_offsets[dim+1] * !(1 == y_shape[dim]);           
                o_offset += o_dim_offsets[dim+1]; 
            }
            x_offset -= x_dim_offsets[dim] * !(1 == x_shape[dim]);        
            y_offset -= y_dim_offsets[dim] * !(1 == y_shape[dim]);           
            o_offset -= o_dim_offsets[dim];           
        };
        recurse_broadcasting(0);
    }
    return out;
}


}//namespace czy

