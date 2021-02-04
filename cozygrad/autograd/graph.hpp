#pragma once

#include "node.hpp"

#include <unordered_set>
#include <vector>
#include <algorithm>

namespace czy{

template <typename T>
class graph
{

    public:
        graph(node<T>& output) : graph(&output){};
        graph(node<T>* output);

        void forwards();
        void backwards();
        void zero_gradients();

        std::vector<node<T>*> nodes();

    private:
        node<T>* m_output_node;

        void populate_exec_order(node<T>* node);
 
        std::unordered_set<node<T>*> m_visited;
        std::vector<node<T>*> m_exec_order;
        std::vector<node<T>*> m_reverse_exec_order;
};

template <typename T>
graph<T>::graph(node<T>* output)
{
    m_output_node = output;

    populate_exec_order(m_output_node); 
    m_reverse_exec_order = m_exec_order;
    std::reverse(m_reverse_exec_order.begin(), m_reverse_exec_order.end());
}

template <typename T>
std::vector<node<T>*> graph<T>::nodes(){return m_exec_order;};

template <typename T>
void graph<T>::forwards()
{
    for(auto& node : m_exec_order)
    {
        node->forward();
    }
}

template <typename T>
void graph<T>::backwards()
{

    if(m_output_node->data().size() != 1)
    {
        throw std::runtime_error("backwards: gradient can only be implicitly created for scalar outputs");
    }

    m_output_node->set_gradient(tensor<T>({1}));

    for(auto& node : m_reverse_exec_order)
    {
        node->backward();
    }
}

template <typename T>
void graph<T>::zero_gradients()
{
    for(auto x : m_exec_order)
    {
        x->get_gradient().zeros(); 
    }
}

template <typename T>
void graph<T>::populate_exec_order(node<T>* node)
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


}//namespace czy




