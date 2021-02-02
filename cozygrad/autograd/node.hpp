#pragma once

#include "session.hpp"
#include "../tensor/tensor.hpp"
#include "../tensor/tensor_ops.hpp"

#include <functional>
#include <vector>
#include <math.h>

namespace czy{

template <typename T>
class node
{
    public:
        std::string name = "node";

        node();
        node(const node& x);
        node(tensor<T>* x, bool updatable = true);
        node(tensor<T>& x, bool updatable = true) : node(&x, updatable){};
        ~node();

        void forward();
        void backward();

        void add_child(node *);
        void set_data(tensor<T>* x);
        const tensor<T>& data();
        const tensor<T>& gradient();
        void set_gradient(const tensor<T>& grad);
        std::vector<node*> children();
    
        bool updatable();

        //operators
        node& operator=(const node& other);
        node& operator+(node& other);
        node& operator+(double other);
        node& operator-();
        node& operator-(node& other);
        node& operator-(double other);
        node& operator*(double other);
        node& operator*(node& other);
        node& operator/(double other);
        node& operator/(node& other);

        node& relu();
        node& dot(node& other);
        node& pow(double exponent);
        node& sum();
        node& mean();
        node& exp();
        node& log();
        node& sigmoid();
        node& softmax();
        node& logsoftmax();

    private:

        tensor<T>* m_data;
        tensor<T>* m_gradient;

        bool m_owns_data = true;
        bool m_owns_gradient = true;
        bool m_data_updatable = false;

        std::function<void()> m_forward = [](){};
        std::function<void()> m_backward = [](){};

        std::vector<node*> m_children;
        std::vector<node*> m_node_references;
        node* create_node();
};

template <typename T>
node<T>::node()
{
    m_data = new tensor<T>();
    m_gradient = new tensor<T>();
}

template <typename T>
node<T>::node(const node<T>& x)
{
    *this = x;
}

template <typename T>
node<T>::node(tensor<T>* x, bool updatable)
{
    m_data_updatable = updatable;
    m_owns_data = false;
    m_data = x;
    m_gradient = new tensor<T>();
}

template <typename T>
node<T>::~node()
{
    if(m_owns_data && m_data != nullptr)
    {
        delete m_data;
    }

    if(m_owns_gradient && m_gradient != nullptr)
    {
        delete m_gradient;
    }  
}

template <typename T>
void node<T>::forward(){m_forward();}

template <typename T>
void node<T>::backward(){m_backward();}

template <typename T>
void node<T>::set_data(tensor<T>* x)
{
    if(m_owns_data && m_data != nullptr)
    {
        delete m_data;
    }
    m_data = x;
    m_owns_data = false;
}

template <typename T>
bool node<T>::updatable(){return m_data_updatable;}
template <typename T>
const tensor<T>& node<T>::data(){return *m_data;}
template <typename T>
const tensor<T>& node<T>::gradient(){return *m_gradient;}
template <typename T>
void node<T>::set_gradient(const tensor<T>& grad){ *m_gradient = grad;}


template <typename T>
std::vector<node<T>*> node<T>::children(){return m_children;}

template <typename T>
void node<T>::add_child(node<T> * x)
{
    //copy of refers to itself if it is not a copy
    m_children.push_back(x);
}

template <typename T>
node<T>* node<T>::create_node()
{
    auto out = new node<T>();

    auto& session = Session<T>::get_session();
    session.add_node(out);

    return out;
}

//------------OPERATORS----------------
template <typename T>
node<T>& node<T>::operator=(const node<T>& other)
{
    m_data = other.m_data;
    m_gradient = other.m_gradient;
    m_owns_data = false;
    m_owns_gradient = false;
    m_children = other.m_children;
    m_backward = other.m_backward;
    m_forward = other.m_forward;
    m_data_updatable = other.m_data_updatable;

    return *this;
}

template <typename T>
node<T>& node<T>::operator+(node<T>& other)
{
    node<T>* out = create_node();

    out->add_child(this);
    out->add_child(&other);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = *m_data + *other.m_data;
    };

    std::function<void()> backward = [&, out](){
        *m_gradient = m_gradient->size() == 0 ? *out->m_gradient : *m_gradient + *out->m_gradient;
        *other.m_gradient = other.m_gradient->size() == 0 ? *out->m_gradient : *other.m_gradient + *out->m_gradient;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "add";

    return *out;
}

template <typename T>
node<T>& node<T>::operator+(double other)
{
    auto out = create_node();

    out->add_child(this);

    std::function<void()> forward = [&, out, other](){ 
        *out->m_data = *m_data + other;
    };

    std::function<void()> backward = [&, out, other](){
        *m_gradient = m_gradient->size() == 0 ? *out->m_gradient : *m_gradient + *out->m_gradient;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "add";

    return *out;
}

template <typename T>
node<T>& operator+(double other, node<T>& rhs){ return rhs + other;}

template <typename T>
node<T>& node<T>::operator-(){ return *this * -1;}
template <typename T>
node<T>& node<T>::operator-(node<T>& other){ return *this + (-other);}
template <typename T>
node<T>& node<T>::operator-(double other){ return *this + (-other);}
template <typename T>
node<T>& operator-(double other, node<T>& rhs){ return (-rhs) + other;}

template <typename T>
node<T>& node<T>::operator*(node<T>& other)
{
    auto out = create_node();
    out->add_child(this);
    out->add_child(&other);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = *m_data * *other.m_data;
    };

    std::function<void()> backward = [&, out](){
        auto a_derivative = *out->m_gradient * *other.m_data;
        auto b_derivative = *out->m_gradient * *m_data;

        *m_gradient = m_gradient->size() == 0 ? a_derivative : *m_gradient + a_derivative;
        *other.m_gradient = other.m_gradient->size() == 0 ? b_derivative : *other.m_gradient + b_derivative;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "mul";

    return *out;
}

template <typename T>
node<T>& node<T>::operator*(double other)
{
    auto out = create_node();
    out->add_child(this);
    //out->add_child(&other);

    std::function<void()> forward = [&, out, other](){ 
        *out->m_data = *m_data * other;
    };

    std::function<void()> backward = [&, out, other](){
        auto a_derivative = *out->m_gradient * other;
        *m_gradient = m_gradient->size() == 0 ? a_derivative : *m_gradient + a_derivative;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "mul";

    return *out;
}

template <typename T>
node<T>& operator*(double other, node<T>& node)
{
    return node * other;
}

//div
template <typename T>
node<T>& node<T>::operator/(double other)
{
    return *this * std::pow(other, -1);
}
template <typename T>
node<T>& node<T>::operator/(node<T>& other)
{
    return *this * other.pow(-1);
}
template <typename T>
node<T>& operator/(double other, node<T>& node)
{
    return other * node.pow(-1);
}

//relu
template <typename T>
node<T>& node<T>::relu()
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::max(*m_data, (T) 0);
    };

    std::function<void()> backward = [&, out](){
        auto a_grad = *out->m_gradient * (*m_data > 0);
        *m_gradient = m_gradient->size() == 0 ? a_grad : *m_gradient + a_grad;
    };
    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "relu";

    return *out;
}

template <typename T>
node<T>& node<T>::dot(node<T>& other)
{
    auto out = create_node();

    out->add_child(this);
    out->add_child(&other);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::dot(*m_data,*other.m_data);
    };

    std::function<void()> backward = [&, out](){
        
        auto a = op::dot(*out->m_gradient, op::transpose(*other.m_data));    
        auto b = op::dot(op::transpose(*m_data), *out->m_gradient);

        *m_gradient = m_gradient->size() == 0 ? a: *m_gradient + a;
        *other.m_gradient = other.m_gradient->size() == 0 ? b : *other.m_gradient + b;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "dot";

    return *out;
}

template <typename T>
node<T>& node<T>::pow(double exponent)
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out, exponent](){ 
        *out->m_data = op::pow(*m_data, exponent);
    };

    std::function<void()> backward = [&, out, exponent](){
        auto der = *out->m_gradient * op::pow(*m_data, exponent - 1.0f) * exponent;
        *m_gradient = m_gradient->size() == 0 ? der : *m_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "pow";

    return *out;
}

//this will need to be changed in future to work with more dimensions
template <typename T>
node<T>& node<T>::sum()
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::sum(*m_data);
    };

    std::function<void()> backward = [&, out](){
        //this is the bit that will need changed : )
        double x = out->m_gradient->data()[0];
        auto der = tensor<T>(m_data->shape());
        der.of_value(x);
        *m_gradient = m_gradient->size() == 0 ? der : *m_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "sum";

    return *out;
}

//this will need to be changed in future to work with more dimensions
template <typename T>
node<T>& node<T>::mean()
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::sum(*m_data) / m_data->size();
    };

    std::function<void()> backward = [&, out](){
        //this is the bit that will need changed : )
        double x = out->m_gradient->data()[0] / m_data->size();
        auto der = tensor<T>(m_data->shape());
        der.of_value(x);
        *m_gradient = m_gradient->size() == 0 ? der : *m_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "mean";

    return *out;

}

template <typename T>
node<T>& node<T>::exp()
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::exp(*m_data);
    };

    std::function<void()> backward = [&, out](){
        auto der = *out->m_data * *out->m_gradient;
        *m_gradient = m_gradient->size() == 0 ? der : *m_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "exp";

    return *out;
}

template <typename T>
node<T>& node<T>::log()
{
    auto out = create_node();
    out->add_child(this);

    std::function<void()> forward = [&, out](){ 
        *out->m_data = op::log(*m_data);
    };

    std::function<void()> backward = [&, out](){
        auto der = *out->m_gradient / *m_data;
        *m_gradient = m_gradient->size() == 0 ? der : *m_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;
    out->name = "log";

    return *out;
}

template <typename T>
node<T>& node<T>::sigmoid()
{
    return 1 / (1 + (-*this).exp());
}

template <typename T>
node<T>& node<T>::softmax()
{
    //need to add broadcasting for this to work : )
    return this->exp() / this->exp().sum();
}

template <typename T>
node<T>& node<T>::logsoftmax()
{
    return this->softmax().log();
}
 
}//namespace czy

