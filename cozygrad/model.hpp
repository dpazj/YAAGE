#pragma once

#include "tensor.hpp"
#include "node.hpp"
#include "graph.hpp"

#include <vector>

class model
{
    public: 
        model(); //optimiser, batch size, loss function
        ~model();

        void train(std::vector<tensor>& x_train, std::vector<tensor>& y_train);
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

model::model()
{

}

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
    param->random(); // initialises the parameters to random (-1, 1) might need to change this location
    m_model_parameters.push_back(param);

    node* out = new node(param);

    Session& session = Session::get_session();
    session.add_node(out);

    return *out;
}

void model::train(std::vector<tensor>& x_train, std::vector<tensor>& y_train)
{
    auto& model = create_model();

    //create loss function
    //loss = -((labels * torch.log(output)) + (1 - labels) * torch.log(1 - output)).sum()
    //node label;
    
   // auto& loss = (label * model.log()) +  

    graph g(*m_input_node,model);

    for(size_t i=0; i < x_train.size();i++)
    {
        m_input_node->set_data(&x_train[i]);
       // label.set_data(&y_train[i]);

        g.forwards();
        g.backwards();
        g.zero_gradients();
        model.data()->print();
    }

}