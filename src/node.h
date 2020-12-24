#pragma once

#include "matrix.h"

#include <vector>
#include <memory>
#include <string>



//Base Node
class Node  
{
    public:
        ~Node();
        virtual void Forward() = 0;
        //virtual void Backward() = 0;     
        Matrix* Data();
        std::vector<Node*> Children();
        const std::string Name();

    protected:
        void Connect(Node* node);
                
        std::vector<Node*> m_input_nodes;
        std::vector<Node*> m_output_nodes;
        
        Matrix* m_data = nullptr;

        std::string m_name = "Node";

    private:
        void AddOutput(Node * node);
};


//Value
class Value : public Node
{
    public:
        Value(Matrix& val) : Value(&val){};
        Value(Matrix* val);
        void Forward();
        
        
};

//OPERATIONS
class Add : public Node
{
    public:
        Add(Node& x, Node& y) : Add(&x, &y){};
        Add(Node * x, Node * y);
        void Forward();
};


class Pow : public Node
{
    public:
        Pow(Node& x, double exponent) : Pow(&x, exponent){};
        Pow(Node * x, double exponent);
        void Forward();
    private:
        double m_exponent;
};

