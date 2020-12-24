#include "node.h"

#include <stdexcept>
#include <math.h>

//NODE
Node::~Node()
{
    //delete m_data;
}

Matrix* Node::Data(){return m_data;}
const std::string Node::Name(){return m_name;}
std::vector<Node*> Node::Children(){return m_input_nodes;};

void Node::Connect(Node* node)
{
    m_input_nodes.push_back(node);
    node->AddOutput(this);
}

void Node::AddOutput(Node* node)
{
    m_output_nodes.push_back(node);
}


//Value
Value::Value(Matrix* val)
{
    m_data = val;
    m_name = "Value";
}

void Value::Forward(){}

//OPERATIONS

//ADD

Add::Add(Node * x, Node * y)
{
    Connect(x);
    Connect(y);
    m_name = "Add";
}

void Add::Forward()
{
    //we can assume these exist
    Matrix* a = m_input_nodes[0]->Data(); 
    Matrix* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly :) ");
    }
    if(a->GetColums() != b->GetColums() || a->GetRows() != b->GetRows())
    {
        throw std::runtime_error("ADD: Matrix dims are not the same");
    }

    m_data = new Matrix(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = (*a)[i][j] + (*b)[i][j]; 
        }
    }
}

//POW
Pow::Pow(Node * x, double exponent)
{
    Connect(x);
    m_exponent = exponent;
    m_name = "Pow";
}

void Pow::Forward()
{
    Matrix* a = m_input_nodes[0]->Data();
    if(a == nullptr)
    {
        throw std::runtime_error("POW: Data of input nodes == nullptr. Check nodes are connected properly :) ");
    }
    m_data = new Matrix(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = pow( (*a)[i][j], m_exponent); 
        }
    }
}

