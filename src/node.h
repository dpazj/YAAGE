#pragma once

#include "matrix.h"

#include <vector>
#include <memory>

enum NodeType {Operation, Variable, Placeholder}; 

//Base Node
class Node  
{
    public:
        void AddOutputNode(std::shared_ptr<Node> node);
        void SetOutput(Matrix* output);
        NodeType GetType();
    protected:
        std::vector<std::shared_ptr<Node>> m_input_nodes;
        std::vector<std::shared_ptr<Node>> m_output_nodes;
        Matrix* m_output;
        NodeType m_node_type;
       
};

//Operation
class Operation : public Node
{
    public:
        Operation(std::vector<std::shared_ptr<Node>> input_nodes);
        //virtual Matrix* Forward(Matrix* x, Matrix* y) = 0;
        //virtual void Backward(Matrix* x, Matrix* y);        
};

//Variable
class Variable : public Node
{
    public:
        Variable(Matrix* val);
        ~Variable();
};

//Placeholder
class Placeholder : public Node
{
    public:
        Placeholder();

};

class Add : public Operation
{
    public:
        Matrix* Forward(Matrix* x, Matrix* y);

};