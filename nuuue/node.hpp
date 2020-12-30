#pragma once

#include "tensor.hpp"

#include <functional>
#include <vector>

class node
{
    public:
        node();
        node(tensor* x);
        node(tensor& x) : node(&x){};
        ~node();

        void forward();
        void backward();

        tensor* data();
        tensor* gradient();
        std::vector<node*> children();

        node& operator=(const node& rhs);
        node& operator+(node& rhs);

    private:

        void add_child(node *);

        tensor* m_data; //shared ptr?
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

node::node(tensor* x)
{
    m_owns_data = false;
    m_data = x;
    m_gradient = new tensor();
}

node::~node()
{

   // std::cout << "still called - this causes the seg fault" << std::endl;
    if(m_owns_data == true && m_data != nullptr)
    {
        //delete m_data;
    }
    //do something with gradient
    //do something with node references
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