#pragma once

#include "optim.hpp"
#include "../tensor/tensor.hpp"
#include "../autograd/node.hpp"
#include "../autograd/graph.hpp"
#include "../autograd/session.hpp"

#include <vector>
#include <functional>
#include <exception>

namespace czy{

class model
{
    public: 
        ~model();

        void train(std::vector<tensor>& x_train, std::vector<tensor>& y_train, optimizer& optim, unsigned int epochs, std::function<node&(node&,node&)> loss_fn);
        void evaluate(std::vector<tensor>& x_test, std::vector<tensor>& y_test);


        virtual node& create_model() = 0;

    protected:
        //model builder api
        node& create_input_node();
        node& create_model_param(size_t m, size_t n); //shape

    private:
        std::vector<tensor*> m_model_parameters;
        node* m_input_node = nullptr;
        node* m_model = nullptr;

        std::function<node&(node&,node&)> m_loss_fn;
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
    input->name = "input";

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

void model::train(std::vector<tensor>& x_train, std::vector<tensor>& y_train, optimizer& optim, unsigned int epochs, std::function<node&(node&,node&)> loss_fn)
{
    std::cout << "started training" << std::endl;
    auto& model = create_model();

    m_model = &model;
    m_loss_fn = loss_fn;

    node label;

    auto& loss = loss_fn(label, model);
    graph g(loss);


    for(size_t k=0; k < epochs; k++){
        std::cout << "start epoch " << k + 1 << std::endl;
        tensor av_loss = {0};

        for(size_t i=0; i < x_train.size();i++)
        {
            m_input_node->set_data(&x_train[i]);
            label.set_data(&y_train[i]);

            g.forwards();

            model.data()->print();

            g.backwards();
            optim.step(g.nodes());
            g.zero_gradients();

            //stats
            av_loss = av_loss + *loss.data();
        }
        std::cout << "epoch " + std::to_string(k + 1) +" average loss: " << av_loss.data()[0] / x_train.size() << std::endl;
    }
}

void model::evaluate(std::vector<tensor>& x_test, std::vector<tensor>& y_test)
{
    if(m_model == nullptr)
    {
        throw std::runtime_error("Model has not been trained yet : )");
    }

    auto& model = *m_model;
    node label;
    tensor av_loss = {0};


    auto& loss = m_loss_fn(label, model);

    graph g(loss);


    for(size_t i=0; i < x_test.size();i++)
    {
        m_input_node->set_data(&x_test[i]);
        label.set_data(&y_test[i]);
        g.forwards();

        av_loss = av_loss + *loss.data();
        
        //stats
    }

    std::cout << "Evaluation average loss: " << av_loss.data()[0] / x_test.size() << std::endl;
        
}

}//namespace czy


