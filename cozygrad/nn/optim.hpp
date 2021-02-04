#pragma once

#include "../autograd/node.hpp"


namespace czy{

template <typename T>
class optimizer
{   
    public:
        optimizer(double lr);
        virtual void step(std::vector<node<T>*> graph_nodes) = 0;

    protected:
        double m_lr;
};

template <typename T>
optimizer<T>::optimizer(double lr)
{
    m_lr = lr;
}


template <typename T>
class SDG : public optimizer<T>
{
    public:
        SDG(double lr = 0.01) : optimizer<T>(lr){};
        void step(std::vector<node<T>*> graph_nodes);
};



template <typename T>
void SDG<T>::step(std::vector<node<T>*> graph_nodes)
{
    for(auto& x : graph_nodes)
    {
        if(x->updatable())
        {
            auto lr_grad = this->m_lr * x->gradient();
            auto x_data = x->data(); 

            auto new_data = x_data - lr_grad;
            x->set_data( new_data );
        }
    }
}

}//namespace czy

