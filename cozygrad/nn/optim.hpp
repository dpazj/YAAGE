#pragma once

#include "../autograd/node.hpp"


namespace czy{

class optimizer
{   
    public:
        optimizer(double lr);
        virtual void step(std::vector<node*> graph_nodes) = 0;

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
        void step(std::vector<node*> graph_nodes);
};

void SDG::step(std::vector<node*> graph_nodes)
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

}//namespace czy

