#pragma once

#include "node.hpp"

#include <unordered_set>
#include <vector>
#include <algorithm>

class graph
{

    public:
        graph(node& input, node& output) : graph(&input, &output){};
        graph(node * input, node * output);

        void forwards();
        void backwards();

    private:
        node* m_input_node;
        node* m_output_node;

        void populate_exec_order(node* node);
 
        std::unordered_set<node*> m_visited;
        std::vector<node *> m_exec_order;
};


graph::graph(node * input, node * output)
{
    m_input_node = input;
    m_output_node = output;
}

void graph::forwards()
{
    m_visited.clear();
    m_exec_order.clear();
    populate_exec_order(m_output_node); 

    for(auto& node : m_exec_order)
    {
        node->forward();
        node->data()->print();
    }

    std::cout << "answer: ";
    m_output_node->data()->print();  
}

void graph::backwards()
{
    *m_output_node->gradient() = tensor({1});

    m_output_node->gradient()->print();

    std::reverse(m_exec_order.begin(), m_exec_order.end());
    for(auto& node : m_exec_order)
    {
        node->backward();
        //node->data()->print();
    }

    std::cout << "dy/dx answer: ";
    m_input_node->gradient()->print();  
}


void graph::populate_exec_order(node* node)
{
    if(m_visited.find(node) == m_visited.end()) //if child not in visited
    {
        m_visited.insert(node);
        for(const auto& child : node->children())
        {
            //child->data()->print();
            populate_exec_order(child);
        }
        m_exec_order.push_back(node);
    }
} 
