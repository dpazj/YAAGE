#include "node.h"
#include "matrix_ops.h"

#include <stdexcept>
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

Matrix* Node::Data(){return m_data;}
Matrix* Node::Gradient(){return m_gradient;}
std::vector<Node*> Node::Children(){return m_input_nodes;};

void Node::AllocateGradientMem(size_t row, size_t col, double val)
{
    if(m_gradient != nullptr){return;} //gradient already been allocated
    m_gradient = new Matrix(row,col, val);
}

void Node::Connect(Node* node)
{
    m_input_nodes.push_back(node);
    node->AddOutput(this);
}

void Node::AddOutput(Node* node)
{
    m_output_nodes.push_back(node);
}

//TWO INPUT NODE
TwoInputNode::TwoInputNode(Node * x, Node * y)
{
    Connect(x);
    Connect(y);
}


//Value
const std::string Value::Name(){return "Value";}

Value::Value(std::initializer_list<double> il)
{
    m_data = new Matrix(il);
    m_owns_memory = true;
}

Value::Value(std::initializer_list<std::initializer_list<double>> il)
{
    m_data = new Matrix(il);
    m_owns_memory = true;
}

Value::Value(Matrix* val)
{
    m_data = val;
    m_owns_memory = false;
}

void Value::Forward(){}
void Value::Backward(){}

//OPERATIONS

//ADD
const std::string Add::Name(){return "Add";}

void Add::Forward()
{
    //we can assume these nodes exist
    Matrix* a = m_input_nodes[0]->Data(); 
    Matrix* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
  
    m_data = new Matrix(op::Add(*a, *b));
}

void Add::Backward()
{

    Node* a = m_input_nodes[0];
    Node* b = m_input_nodes[1];

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD back: a or b is nullprt");
    }
    
    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());
    b->AllocateGradientMem(m_data->Rows(), m_data->Columns());

    Matrix* a_grad = a->Gradient(); 
    Matrix* b_grad = b->Gradient(); 

    *a_grad = op::Add(*a_grad, *m_gradient);
    *b_grad = op::Add(*b_grad, *m_gradient);
}

//SUB
const std::string Sub::Name(){return "Sub";}

void Sub::Forward()
{
    //we can assume these nodes exist
    Matrix* a = m_input_nodes[0]->Data(); 
    Matrix* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("SUB: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }

    m_data = new Matrix(op::Sub(*a, *b));
}

void Sub::Backward()
{
    Node* a = m_input_nodes[0];
    Node* b = m_input_nodes[1];
    //input nodes gradients

    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());
    b->AllocateGradientMem(m_data->Rows(), m_data->Columns());

    Matrix* a_grad = a->Gradient(); 
    Matrix* b_grad = b->Gradient(); 

    *a_grad = op::Add(*a_grad, *m_gradient);
    *b_grad = op::Sub(*b_grad, *m_gradient);
}

//POW
const std::string Pow::Name(){return "Pow";}

Pow::Pow(Node * node, double exponent)
{
    m_exponent = exponent;
    Connect(node);
}

void Pow::Forward()
{
    Matrix* a = m_input_nodes[0]->Data();
    if(a == nullptr)
    {
        throw std::runtime_error("POW: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    
    m_data = new Matrix(op::Pow(*a, m_exponent));
}

void Pow::Backward()
{
    Node* a = m_input_nodes[0];

    //input nodes gradients
    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());

    Matrix* a_grad = a->Gradient(); 
    Matrix* a_data = a->Data(); 

    auto pow = op::Pow(*a_data,m_exponent-1);
    auto mul = op::Mul(pow, m_exponent);
    auto der = op::Mul(mul, *m_gradient);

    *a_grad = op::Add(*a_grad, der);
}

// //DOT

// const std::string Dot::Name(){return "Dot";}

// void Dot::Forward()
// {
//     //we can assume these exist
//     Matrix* a = m_input_nodes[0]->Data(); 
//     Matrix* b = m_input_nodes[1]->Data(); 

//     if(a == nullptr || b == nullptr)
//     {
//         throw std::runtime_error("Dot: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
//     m_data = new Matrix(op::Dot(*a, *b));
// }

// void Dot::Backward()
// {
    
// }

// //MUL

const std::string Mul::Name(){return "Mul";}

void Mul::Forward()
{
    Matrix* a = m_input_nodes[0]->Data(); 
    Matrix* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    m_data = new Matrix(op::Mul(*a, *b));
}

void Mul::Backward()
{
    Node* a = m_input_nodes[0];
    Node* b = m_input_nodes[1];
    //input nodes gradients

    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());
    b->AllocateGradientMem(m_data->Rows(), m_data->Columns());

    Matrix* a_grad = a->Gradient(); 
    Matrix* b_grad = b->Gradient(); 
    Matrix* a_data = a->Data(); 
    Matrix* b_data = b->Data(); 

    auto x = op::Mul(*b_data,*m_gradient);
    auto y = op::Mul(*a_data,*m_gradient);

    *a_grad = op::Add(*a_grad,  x);
    *b_grad = op::Add(*b_grad, y);
}

// //Sum
// Sum::Sum(Node * node)
// {
//     Connect(node);
// }

// const std::string Sum::Name(){return "Mul";}

// void Sum::Forward()
// {
//     Matrix* a = m_input_nodes[0]->Data(); 

//     if(a == nullptr)
//     {
//         throw std::runtime_error("Sum: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
//     m_data = new Matrix(op::Sum(*a));
// }

// void Sum::Backward()
// {
    
// }

//ReLU
ReLU::ReLU(Node* node)
{
    Connect(node);
}

const std::string ReLU::Name(){return "Relu";}

void ReLU::Forward()
{
    Matrix* a = m_input_nodes[0]->Data(); 

    if(a == nullptr)
    {
        throw std::runtime_error("Relu: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
    m_data = new Matrix(op::Max(*a,0.0f));
}

void ReLU::Backward()
{
    Node* a = m_input_nodes[0];

    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());

    Matrix* a_grad = a->Gradient(); 
    Matrix* a_data = a->Data(); 

    Matrix tmp(a_data->Rows(), a_data->Columns());
    double* tmp_data = tmp.Data();
    double* a_data_ptr = a_data->Data();

    for(size_t i=0; i<tmp.Size();i++)
    { 
        tmp_data[i] = a_data_ptr[i] > 0 ? 1.0f : 0.0f;
    }

    auto grad = op::Mul(*m_gradient, *tmp_data);

    *a_grad = op::Add(*a_grad, grad);
}