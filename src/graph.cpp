#include "graph.h"

#include <algorithm>

Graph::Graph(Node * input, Node * output)
{
    m_input_node = input;
    m_output_node = output;
}

Tensor* Graph::Forward()
{
    TraverseBackwards(m_output_node); 

    for(const auto& node : m_exec_order)
    {
        //std::cout << node->Name() << std::endl;
        node->Forward();
        //node->Data()->Print();
    }

    return m_output_node->Data();  
}

void Graph::TraverseBackwards(Node* start)
{
    auto children = start->Children();
    m_exec_order.insert(start);

    for(const auto& child : children)
    {
        if(m_exec_order.find(child) == m_exec_order.end())
        {
            m_exec_order.insert(child);
            TraverseBackwards(child);
        }
  
    }
}