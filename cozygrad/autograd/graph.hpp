#pragma once

#include "node.hpp"

#include <unordered_set>
#include <vector>
#include <algorithm>

namespace czy{
namespace autograd{

class graph
{

    public:
        graph(node& output) : graph(&output){};
        graph(node * output);

        void forwards();
        void backwards();
        void zero_gradients();

        std::vector<node*> nodes();

    private:
        node* m_output_node;

        void populate_exec_order(node* node);
 
        std::unordered_set<node*> m_visited;
        std::vector<node *> m_exec_order;
        std::vector<node *> m_reverse_exec_order;
};

graph::graph(node * output)
{
    m_output_node = output;

    populate_exec_order(m_output_node); 
    m_reverse_exec_order = m_exec_order;
    std::reverse(m_reverse_exec_order.begin(), m_reverse_exec_order.end());
}

std::vector<node*> graph::nodes(){return m_exec_order;};

void graph::forwards()
{
    for(auto& node : m_exec_order)
    {
        node->forward();
    }
}

void graph::backwards()
{
    *m_output_node->gradient() = tensor({1});

    for(auto& node : m_reverse_exec_order)
    {
        node->backward();
    }
}

void graph::zero_gradients()
{
    for(auto& node : m_exec_order)
    {
        node->gradient()->zeros(); 
    }
}

void graph::populate_exec_order(node* node)
{
    if(m_visited.find(node) == m_visited.end()) //if child not in visited
    {
        m_visited.insert(node);

        for(const auto& child : node->children())
        {
            populate_exec_order(child);
        }
        m_exec_order.push_back(node);
    }
} 


} //namespace autograd
}//namespace czy




