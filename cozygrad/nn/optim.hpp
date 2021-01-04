#pragma once

#include "../autograd/node.hpp"


namespace czy{
namespace nn{

class optimizer
{   
    public:
        optimizer(double lr);
        virtual void step(std::vector<autograd::node*> graph_nodes) = 0;

    protected:
        double m_lr;
};

optimizer::optimizer(double lr)
{
    m_lr = lr;
}

class SDG : public optimizer
{
    public:
        SDG(double lr = 0.01) : optimizer(lr){};
        void step(std::vector<autograd::node*> graph_nodes);
};

void SDG::step(std::vector<autograd::node*> graph_nodes)
{
    for(auto& x : graph_nodes)
    {
        if(x->updatable())
        {
            tensor* data = x->data(); 
            tensor* grad = x->gradient();
            *data = *data - (m_lr * *grad);
        }
    }
}

}//namespace nn
}//namespace czy

