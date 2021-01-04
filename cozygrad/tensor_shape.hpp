#pragma once
#include <initializer_list>
#include <vector>
#include <exception>

namespace czy{
    
class tensor_shape
{
    public:
        tensor_shape(std::initializer_list<size_t> il) : m_strides(il){};
        tensor_shape(std::vector<size_t> shape) : m_strides(shape){};

        size_t calculate_size();
        size_t get_stride(int dim);

        std::vector<size_t> shape() const; 

        bool operator==(tensor_shape& rhs);
        bool operator!=(tensor_shape& rhs);

        void print() const;

    private:
        std::vector<size_t> m_strides;
};


size_t tensor_shape::calculate_size()
{
    size_t size = 0;
    for(const auto& x : m_strides)
    {
        size += x;
    }
    return size;
}

size_t tensor_shape::get_stride(int dim)
{
    if(dim < 0 || (size_t) dim > m_strides.size())
    {
        throw std::runtime_error("get_stride: shape does not have " + std::to_string(dim) + "dimension.");
    }
    return m_strides.at(dim);
}

std::vector<size_t> tensor_shape::shape() const
{
    return m_strides;
}

void tensor_shape::print() const
{
    for(const auto& x : m_strides)
    {
        std::cout << x << " ";
    }
    std::cout << std::endl;
}

bool tensor_shape::operator==(tensor_shape& rhs)
{
    
    if(m_strides.size() != rhs.m_strides.size())
    {
        return false;
    }

    for(size_t i= 0; i < m_strides.size(); i++)
    {
        if(m_strides[i] != rhs.m_strides[i])
        {
            return false;
        }
    }
    return true;
}

bool tensor_shape::operator!=(tensor_shape& rhs)
{
    bool equal = *this == rhs;
    return !equal;
}

}//namespace czy

