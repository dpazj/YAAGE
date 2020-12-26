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

    VisitForwards(m_output_node); 

    for(const auto& node : m_visited)
    {
        node->Forward();
    }

    return m_output_node->Data();  
}

Tensor* Graph::Backward()
{
    Tensor* answer = m_output_node->Data();
    m_output_node->AllocateGradientMem(answer->Rows(), answer->Columns(),1.0f);
    
    m_visited.clear();
    VisitBackwards(m_output_node);

    return m_input_node->Gradient();
}

void Graph::PopulateExecOrder(Node* start)
{
    auto children = start->Children();
    m_visited.insert(start);

    for(const auto& child : children)
    {
        if(m_visited.find(child) == m_visited.end())
        {
            m_visited.insert(child);
            PopulateExecOrder(child);
        }
  
    }
}

void Graph::VisitForwards(Node* node)
{
    auto children = node->Children();

    m_visited.insert(node);

    for(const auto& child : children)
    {
        if(m_visited.find(child) == m_visited.end())
        {
            
            m_visited.insert(child);
            VisitForwards(child);
        }
    }
}

void Graph::VisitBackwards(Node* node)
{
    auto children = node->Children();
    node->Backward();

    m_visited.insert(node);

    for(const auto& child : children)
    {
        if(m_visited.find(child) == m_visited.end())
        {
            
            m_visited.insert(child);
            VisitBackwards(child);
        }
    }
}


