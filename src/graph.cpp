#include "graph.h"

#include <algorithm>
#include <iostream>

Graph::Graph(Node * input, Node * output)
{
    m_input_node = input;
    m_output_node = output;
}

Tensor* Graph::Forward()
{

    m_visited.clear();
    m_exec_order.clear();
    PopulateExecOrder(m_output_node); 

    for(const auto& node : m_exec_order)
    {
        node->Forward();
        //std::cout << node->Name() << std::endl;
        //node->Name();

    }
    return m_output_node->Data();  
}

Tensor* Graph::Backward()
{
    Tensor* answer = m_output_node->Data();
    m_output_node->AllocateGradientMem(answer->Rows(), answer->Columns(),1.0f);
    
    std::reverse(m_exec_order.begin(),m_exec_order.end());
    for(const auto& node : m_exec_order)
    {
        node->Backward();
    }
    return m_input_node->Gradient();
}


void Graph::PopulateExecOrder(Node* node)
{
    if(m_visited.find(node) == m_visited.end()) //if child not in visited
    {
        m_visited.insert(node);
        
        for(const auto& child : node->Children())
        {
            PopulateExecOrder(child);
        }
        m_exec_order.push_back(node);
    }
    
} 
    



