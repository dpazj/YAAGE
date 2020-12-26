#include "node.h"
#include "tensor_ops.h"

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

Tensor* Node::Data(){return m_data;}
Tensor* Node::Gradient(){return m_gradient;}
std::vector<Node*> Node::Children(){return m_input_nodes;};

void Node::AllocateGradientMem(size_t row, size_t col, double val)
{
    if(m_gradient != nullptr){return;} //gradient already been allocated
    m_gradient = new Tensor(row,col, val);
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
    m_data = new Tensor(il);
    m_owns_memory = true;
}

Value::Value(std::initializer_list<std::initializer_list<double>> il)
{
    m_data = new Tensor(il);
    m_owns_memory = true;
}

Value::Value(Tensor* val)
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
    Tensor* a = m_input_nodes[0]->Data(); 
    Tensor* b = m_input_nodes[1]->Data(); 

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
    }
  
    m_data = new Tensor(op::Add(*a, *b));
}

void Add::Backward()
{

    Node* a = m_input_nodes[0];
    Node* b = m_input_nodes[1];

    if(a == nullptr || b == nullptr)
    {
        throw std::runtime_error("ADD back: AAHHHHHH");
    }
    

    a->AllocateGradientMem(m_data->Rows(), m_data->Columns());
    b->AllocateGradientMem(m_data->Rows(), m_data->Columns());


    Tensor* a_grad = a->Gradient(); 
    Tensor* b_grad = b->Gradient(); 


    *a_grad = op::Add(*a_grad, *m_gradient);
    *b_grad = op::Add(*b_grad, *m_gradient);

}


// //SUB
// const std::string Sub::Name(){return "Sub";}

// void Sub::Forward()
// {
//     //we can assume these nodes exist
//     Tensor* a = m_input_nodes[0]->Data(); 
//     Tensor* b = m_input_nodes[1]->Data(); 

//     if(a == nullptr || b == nullptr)
//     {
//         throw std::runtime_error("SUB: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }

//     m_data = new Tensor(op::Sub(*a, *b));
// }

// void Sub::Backward()
// {
//     Node* a = m_input_nodes[0];
//     Node* b = m_input_nodes[1];
//     //input nodes gradients

//     a->AllocateGradientMem(m_data->Rows(), m_data->Columns());
//     a->AllocateGradientMem(m_data->Rows(), m_data->Columns());

//     Tensor* a_grad = a->Gradient(); 
//     Tensor* b_grad = b->Gradient(); 

//     *a_grad = op::Add(*a_grad, *m_data);
//     *b_grad = op::Add(*b_grad, *m_data);
// }

// //POW
// const std::string Pow::Name(){return "Pow";}

// void Pow::Forward()
// {
//     Tensor* a = m_input_nodes[0]->Data();
//     if(a == nullptr)
//     {
//         throw std::runtime_error("POW: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
    
//     m_data = new Tensor(op::Pow(*a, m_exponent));
// }

// void Pow::Backward()
// {
    
// }

// //DOT

// const std::string Dot::Name(){return "Dot";}

// void Dot::Forward()
// {
//     //we can assume these exist
//     Tensor* a = m_input_nodes[0]->Data(); 
//     Tensor* b = m_input_nodes[1]->Data(); 

//     if(a == nullptr || b == nullptr)
//     {
//         throw std::runtime_error("Dot: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
//     m_data = new Tensor(op::Dot(*a, *b));
// }

// void Dot::Backward()
// {
    
// }

// //MUL

// const std::string Mul::Name(){return "Mul";}

// void Mul::Forward()
// {
//     Tensor* a = m_input_nodes[0]->Data(); 
//     Tensor* b = m_input_nodes[1]->Data(); 

//     if(a == nullptr || b == nullptr)
//     {
//         throw std::runtime_error("ADD: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
//     m_data = new Tensor(op::Mul(*a, *b));
// }

// void Mul::Backward()
// {
    
// }

// //Sum
// Sum::Sum(Node * node)
// {
//     Connect(node);
// }

// const std::string Sum::Name(){return "Mul";}

// void Sum::Forward()
// {
//     Tensor* a = m_input_nodes[0]->Data(); 

//     if(a == nullptr)
//     {
//         throw std::runtime_error("Sum: Data of input nodes == nullptr. Check nodes are connected properly : ) ");
//     }
//     m_data = new Tensor(op::Sum(*a));
// }

// void Sum::Backward()
// {
    
// }