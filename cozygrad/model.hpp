#pragma once

#include "optim.hpp"
#include "tensor.hpp"
#include "node.hpp"
#include "graph.hpp"

#include <vector>
#include <functional>

namespace czy{
    
class model
{
    public: 
        ~model();

        void train(std::vector<tensor>& x_train, std::vector<tensor>& y_train, optimizer& optim, size_t epochs, std::function<node&(node&,node&)> loss_fn);
        //void evaluate();
        virtual node& create_model() = 0;

    protected:
        //model builder api
        node& create_input_node();
        node& create_model_param(size_t m, size_t n); //shape

    private:
        std::vector<tensor*> m_model_parameters;
        node* m_input_node = nullptr;
};

model::~model()
{
    for(const auto& x : m_model_parameters)
    {
        delete x;
    }
}

node& model::create_input_node()
{
    node* input = new node();

    Session& session = Session::get_session();
    session.add_node(input);

    m_input_node = input;

    return *input;
}

node& model::create_model_param(size_t m, size_t n)
{
    tensor* param = new tensor(m,n);
    param->random(-1.0f,1.0f); // initialises the parameters to random (-1, 1) 
    m_model_parameters.push_back(param);

    node* out = new node(param);

    Session& session = Session::get_session();
    session.add_node(out);

    return *out;
}

void model::train(std::vector<tensor>& x_train, std::vector<tensor>& y_train, optimizer& optim, size_t epochs, std::function<node&(node&,node&)> loss_fn)
{
    auto& model = create_model();
    node label;

    auto& loss = loss_fn(label, model);
 
    graph g(loss);

    for(size_t k=0; k < epochs; k++){
        tensor av_loss = {0};
        for(size_t i=0; i < x_train.size();i++)
        {
            m_input_node->set_data(&x_train[i]);
            label.set_data(&y_train[i]);

            g.forwards();
            g.backwards();
            optim.step(g.nodes());
            g.zero_gradients();

            //stats
            av_loss = av_loss + *loss.data();
        }
        std::cout << "epoch " + std::to_string(k + 1) +" average loss: " << av_loss.data()[0] / x_train.size() << std::endl;
    }

}


}//namespace czy


