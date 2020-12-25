#include "node.h"

#include <stdexcept>
#include <math.h>
#include <iostream>

//NODE
Node::~Node()
{
    if(m_owns_memory && m_data != nullptr)
    {
        delete m_data;
    }

    if(m_gradient != nullptr)
    {
        delete m_gradient;
    }
}

Tensor* Node::Data(){return m_data;}
Tensor* Node::Gradient(){return m_gradient;}
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
Value::Value(std::initializer_list<std::initializer_list<double>> il)
{
    m_data = new Tensor(il);
    m_owns_memory = true;
}

Value::Value(Tensor* val)
{
    m_data = val;
    m_name = "Value";
    m_owns_memory = false;
}

void Value::Forward(){}
void Value::Backward(){}

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
    //we can assume these nodes exist
    Tensor* a = m_input_nodes[0]->Data(); 
    Tensor* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    if(a->GetColums() != b->GetColums() || a->GetRows() != b->GetRows())
    {
        throw std::runtime_error("ADD: Matrix dims are not the same");
    }

    m_data = new Tensor(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = (*a)[i][j] + (*b)[i][j]; 
        }
    }
}


//SUB
Sub::Sub(Node * x, Node * y)
{
    Connect(x);
    Connect(y);
    m_name = "SUB";
}

void Sub::Forward()
{
    //we can assume these nodes exist
    Tensor* a = m_input_nodes[0]->Data(); 
    Tensor* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("SUB: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    if(a->GetColums() != b->GetColums() || a->GetRows() != b->GetRows())
    {
        throw std::runtime_error("SUB: Matrix dims are not the same");
    }

    m_data = new Tensor(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = (*a)[i][j] - (*b)[i][j]; 
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
    Tensor* a = m_input_nodes[0]->Data();
    if(a == nullptr)
    {
        throw std::runtime_error("POW: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    m_data = new Tensor(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = pow( (*a)[i][j], m_exponent); 
        }
    }
}

//DOT
Dot::Dot(Node * x, Node * y)
{
    Connect(x);
    Connect(y);
    m_name = "Dot";
}

void Dot::Forward()
{
    //we can assume these exist
    Tensor* a = m_input_nodes[0]->Data(); 
    Tensor* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("MatMul: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }

    size_t M = a->GetRows();
    size_t K = a->GetColums();
    size_t N = b->GetColums();

    if(K != b->GetRows())
    {
        throw std::runtime_error("MatMul: Matricies do not share common dim");
    }

    m_data = new Tensor(a->GetRows(), b->GetColums());

    for(size_t i=0; i<M;i++)
    {
        for (size_t j = 0; j < N; j++)
        {
            double acc = 0.0f;
            for (size_t k = 0; k < K; k++)
            {
                acc += (*a)[i][k] * (*b)[k][j];
            }
            (*m_data)[i][j] = acc; 
        }
    }
}

//MUL
Mul::Mul(Node * x, Node * y)
{
    Connect(x);
    Connect(y);
    m_name = "Dot";
}

void Mul::Forward()
{
    Tensor* a = m_input_nodes[0]->Data(); 
    Tensor* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    if(a->GetColums() != b->GetColums() || a->GetRows() != b->GetRows())
    {
        throw std::runtime_error("ADD: Matrix dims are not the same");
    }

    m_data = new Tensor(a->GetRows(), a->GetColums());
    for(size_t i=0; i<a->GetRows();i++)
    {
        for (size_t j = 0; j < a->GetColums(); j++)
        {
            (*m_data)[i][j] = (*a)[i][j] * (*b)[i][j]; 
        }
    }
}
