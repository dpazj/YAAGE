#pragma once

#include "tensor.hpp"
#include "node.hpp"

#include <vector>

class model
{
    public: 
        model(); //optimiser, batch size, loss function

        void train();
        void evaluate();

        virtual node create_model()
        {
            node x;
            return x.relu();
        }

    protected:
        //model builder api
        node& create_input_node();
        node& create_model_param(size_t m, size_t n); //shape

    private:
        std::vector<tensor*> m_model_parameters;
        std::vector<node*> m_created_nodes;

        node* m_input_node = nullptr;

        //

};