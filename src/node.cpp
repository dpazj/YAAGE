#include "node.h"

#include <stdexcept>

//NODE

void Node::AddOutputNode(std::shared_ptr<Node> node)
{
    m_output_nodes.push_back(node);
}

void Node::SetOutput(Matrix* output)
{
    m_output = output;
}

NodeType Node::GetType(){return m_node_type;}


//OPERATION
Operation::Operation(std::vector<std::shared_ptr<Node>> input_nodes)
{
    m_input_nodes = input_nodes;

    for(auto& input_node : m_input_nodes)
    {
        input_node->AddOutputNode(std::shared_ptr<Node>(this));
    }
}

//VARIABLE
Variable::Variable(Matrix* val)
{
    m_output = val;
    m_node_type = NodeType::Variable;
}

//Placeholder
Placeholder::Placeholder()
{
    m_node_type = NodeType::Placeholder;
}

//OPERATIONS
//ADD
Matrix* Add::Forward(Matrix* x, Matrix* y)
{
    if(x->GetColums() != y->GetColums() || x->GetRows() != y->GetRows())
    {
        throw std::runtime_error("Matricies not same dims");
    }

    Matrix* c = new Matrix(x->GetRows(), x->GetColums());

    for(size_t i=0; i < x->GetRows(); i++)
    {
        for(size_t j=0; j < x->GetColums(); j++)
        {
            (*c)[i][j] = (*x)[i][j] + (*y)[i][j];
        }
    }
    return c;
}