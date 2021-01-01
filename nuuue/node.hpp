#pragma once

#include "tensor.hpp"
#include "tensor_ops.hpp"

#include <functional>
#include <vector>

class node
{
    public:
        node();
        node(const node& x);
        node(tensor* x);
        node(tensor& x) : node(&x){};
        ~node();

        void forward();
        void backward();

        tensor* data();
        tensor* gradient();
        std::vector<node*> children();
    
        //operators
        node& operator=(const node& rhs);
        node& operator+(node& rhs);
        node& operator-(node& rhs);
        node& operator*(node& rhs);

        node& relu();
        node& dot(node& rhs);
        node& pow(double exponent);

    private:
        void add_child(node *);

        tensor* m_data;
        tensor* m_gradient;

        bool m_owns_data = true;
        bool m_owns_gradient = true;

        std::function<void()> m_forward = [](){};
        std::function<void()> m_backward = [](){};

        std::vector<node*> m_children;
        std::vector<node*> m_node_references;
        node* create_node();
};



node::node()
{
    m_data = new tensor();
    m_gradient = new tensor();
}

node::node(const node& x)
{
    *this = x;
}

node::node(tensor* x)
{
    m_owns_data = false;
    m_data = x;
    m_gradient = new tensor();
}

node::~node()
{
    if(m_owns_data && m_data != nullptr)
    {
        delete m_data;
    }

    if(m_owns_gradient && m_gradient != nullptr)
    {
       delete m_gradient;
    }

    for(const auto& x : m_node_references)
    {
        delete x;
    }
}

void node::forward(){m_forward();}
void node::backward(){m_backward();}

tensor* node::data(){return m_data;}
tensor* node::gradient(){return m_gradient;}
std::vector<node*> node::children(){return m_children;}

void node::add_child(node * x){m_children.push_back(x);}

node* node::create_node()
{
    node* out = new node();
    m_node_references.push_back(out);
    return out;
}

node& node::operator=(const node& rhs)
{
    
    m_data = rhs.m_data;
    m_gradient = rhs.m_gradient;
    m_owns_data = false;
    m_owns_gradient = false;
    m_children = rhs.m_children;
    m_backward = rhs.m_backward;
    m_forward = rhs.m_forward;

    return *this;
}


node& node::operator+(node& rhs)
{
    node* out = create_node();

    out->add_child(this);
    out->add_child(&rhs);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;
    tensor* b_data = rhs.m_data;
    tensor* b_gradient = rhs.m_gradient;
    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data, b_data](){ 
        *out_data = *a_data + *b_data;
    };

    std::function<void()> backward = [out_gradient, a_gradient, b_gradient](){
        *a_gradient = a_gradient->size() == 0 ? *out_gradient : *a_gradient + *out_gradient;
        *b_gradient = b_gradient->size() == 0 ? *out_gradient : *b_gradient + *out_gradient;
    };

    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}

node& node::operator-(node& rhs)
{
    node* out = create_node();

    out->add_child(this);
    out->add_child(&rhs);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;
    tensor* b_data = rhs.m_data;
    tensor* b_gradient = rhs.m_gradient;
    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data, b_data](){ 
        *out_data = *a_data - *b_data;
    };

    std::function<void()> backward = [out_gradient, a_gradient, b_gradient](){
        *a_gradient = a_gradient->size() == 0 ? *out_gradient : *a_gradient + *out_gradient;
        *b_gradient = b_gradient->size() == 0 ? -*out_gradient : *b_gradient - *out_gradient;
    };

    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}

node& node::operator*(node& rhs)
{
    node* out = create_node();

    out->add_child(this);
    out->add_child(&rhs);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;
    tensor* b_data = rhs.m_data;
    tensor* b_gradient = rhs.m_gradient;
    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data, b_data](){ 
        *out_data = *a_data * *b_data;
    };

    std::function<void()> backward = [out_gradient, a_gradient, b_gradient, b_data, a_data](){
        auto a_grad = *out_gradient * *b_data;
        auto b_grad = *out_gradient * *a_data;

        *a_gradient = a_gradient->size() == 0 ? a_grad : *a_gradient + a_grad;
        *b_gradient = b_gradient->size() == 0 ? b_grad : *b_gradient + b_grad;
    };

    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}

node& node::relu()
{
    node* out = create_node();
    out->add_child(this);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;

    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data](){ 
        *out_data = op::max(*a_data, 0.0f);
    };

    std::function<void()> backward = [out_gradient, a_gradient, a_data](){
        auto a_grad = *out_gradient * (*a_data > 0.0f);
        *a_gradient = a_gradient->size() == 0 ? a_grad : *a_gradient + a_grad;
    };
    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}

node& node::dot(node& rhs)
{
    node* out = create_node();

    out->add_child(this);
    out->add_child(&rhs);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;
    tensor* b_data = rhs.m_data;
    tensor* b_gradient = rhs.m_gradient;
    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data, b_data](){ 
        *out_data = op::dot(*a_data,*b_data);
    };

    std::function<void()> backward = [out_gradient, a_gradient, b_gradient, b_data, a_data](){
        auto a= op::dot(*out_gradient, op::transpose(*b_data));
        auto b= op::dot(op::transpose(*a_data), *out_gradient);

        *a_gradient = a_gradient->size() == 0 ? a: *a_gradient + a;
        *b_gradient = b_gradient->size() == 0 ? b : *b_gradient + b;
    };

    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}

node& node::pow(double exponent)
{
    node* out = create_node();
    out->add_child(this);

    tensor* a_data = m_data;
    tensor* a_gradient = m_gradient;
    tensor* out_data = out->m_data;
    tensor* out_gradient = out->m_gradient;

    std::function<void()> forward = [out_data, a_data, exponent](){ 
        *out_data = op::pow(*a_data, exponent);
    };

    std::function<void()> backward = [out_gradient, a_gradient, a_data, exponent](){
        auto der = *out_gradient * op::pow(*a_data, exponent - 1.0f) * exponent;
        *a_gradient = a_gradient->size() == 0 ? der : *a_gradient + der;
    };

    out->m_forward = forward;
    out->m_backward = backward;

    return *out;
}