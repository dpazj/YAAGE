#pragma once

#include "../utils.hpp"

#include <initializer_list>
#include <memory>
#include <ostream>
#include <cstring>
#include <random>
#include <functional>
#include <algorithm>


typedef unsigned long size_t;
typedef std::vector<size_t> tensor_shape;

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

        void reshape(std::initializer_list<size_t> new_shape);
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
        
        //operators
        T& operator[](size_t i);
        tensor<T>& operator=(const tensor<T>& rhs);
        tensor<T>& operator=(T rhs);

        tensor<T> operator+(const tensor<T>& rhs);
        tensor<T> operator-(const tensor& rhs);
        tensor<T> operator*(const tensor<T>& rhs);
        tensor<T> operator/(const tensor<T>& rhs);

        tensor<T> operator-();

        //Friends
        template <typename TT>
        friend std::ostream& operator<<(std::ostream& os, const tensor<TT>& ten);

    private:
        size_t calculate_size();

        //broadcast util functions
        std::vector<size_t> calculate_dimension_offsets(tensor_shape& shape) const;
        size_t calculate_offset(std::vector<size_t>& counter, std::vector<size_t>& offsets) const;

        //broadcasting functions
        tensor_shape calculate_broadcast_shapes(tensor_shape& x_shape, tensor_shape& y_shape) const;


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
void tensor<T>::reshape(std::initializer_list<size_t> new_shape)
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
    for(const auto& x : m_shape)
    {
        if(size == 0){size = 1;}
        size *= x;
    }
    return size;
}

template <typename T>
std::vector<size_t> tensor<T>::calculate_dimension_offsets(tensor_shape& shape) const
{
    size_t acc = 1;
    std::vector<size_t> c;
    for(size_t i=shape.size(); i > 0; i--)
    {
        acc *= shape[i-1];
        c.push_back(acc);
    }
    std::reverse(c.begin(), c.end());
    return c;
}

template <typename T>
size_t tensor<T>::calculate_offset(std::vector<size_t>& counter, std::vector<size_t>& offsets) const
{
    size_t acc = 0;
    for(size_t i=0; i<counter.size()-1;i++)
    {
        acc += counter[i] * offsets[i+1];
    }
    return acc;
}

//returns the resulting tensor shape of the broadcast, also prepends x_shape and y_shape with ones
template <typename T>
tensor_shape tensor<T>::calculate_broadcast_shapes(tensor_shape& x_shape, tensor_shape& y_shape) const
{
    tensor_shape out_shape;
    size_t n_dims = std::max(x_shape.size(), y_shape.size());

    auto prepend_ones = [](tensor_shape& x, size_t dims){
        while(x.size() < dims){x.insert(x.begin(), 1);}
        return x;
    };

    x_shape = prepend_ones(x_shape, n_dims);
    y_shape = prepend_ones(y_shape, n_dims);

    for(size_t i=0; i < n_dims; i++)
    {
        if( (x_shape[i] != 1 && y_shape[i] != 1) && (x_shape[i] != y_shape[i]) )
        {
            throw std::runtime_error("Unbroadcastable shapes!");
        }
        out_shape.push_back(std::max(x_shape[i], y_shape[i]));
    }
    return out_shape;
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
        //this can be optimized by adding the offset at each loop instead of calculating on the last dim
        std::vector<size_t> xdim_counter(out_shape.size(), 0);
        std::vector<size_t> ydim_counter(out_shape.size(), 0);
        std::vector<size_t> o_dim_counter(out_shape.size(), 0); 

        std::vector<size_t> x_dim_offsets = calculate_dimension_offsets(x_shape);     
        std::vector<size_t> y_dim_offsets = calculate_dimension_offsets(y_shape);    
        std::vector<size_t> o_dim_offsets = calculate_dimension_offsets(out_shape);     

        std::function<void(size_t)> recurse_broadcasting = [&](size_t dim){

            if(dim == out_shape.size() - 1)
            {
                size_t x_offset = calculate_offset(xdim_counter, x_dim_offsets); 
                size_t y_offset = calculate_offset(ydim_counter, y_dim_offsets);
                size_t o_offset = calculate_offset(o_dim_counter, o_dim_offsets);
                
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
                o_dim_counter[dim] = i;
                xdim_counter[dim] = i % x_shape[dim];
                ydim_counter[dim] = i % y_shape[dim];
                recurse_broadcasting(dim + 1);   
            }              
        };
        recurse_broadcasting(0);
    }
    return out;
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

//adding
template <typename T>
tensor<T> tensor<T>::operator+(const tensor<T>& y)
{
    return broadcast(y, [](T& a, T& b){
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
tensor<T> tensor<T>::operator/(const tensor<T>& y)
{
    return broadcast(y, [](T& a, T& b){
        return a/b;
    });
}

template <typename T>
tensor<T> operator/(T x, const tensor<T>& y)
{
    return (tensor<T>) x / y;
}

//UNARY OPERATORS
template <typename T>
tensor<T> tensor<T>::operator-()
{
    return unary_operation([](T& a){return -a;});
}


}//namespace czy




