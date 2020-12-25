#pragma once

#include "tensor.h"

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

        Tensor* Data();
        Tensor* Gradient();

        std::vector<Node*> Children();
        const std::string Name();

    protected:
        void Connect(Node* node);
                
        std::vector<Node*> m_input_nodes;
        std::vector<Node*> m_output_nodes;
        
        Tensor* m_data = nullptr;
        Tensor* m_gradient = nullptr;

        std::string m_name = "Node";
        bool m_owns_memory = true;
    private:
        void AddOutput(Node * node);
        
        //Tensor* 
        
};



//Value
class Value : public Node
{
    public:
        Value(Tensor& val) : Value(&val){};
        Value(std::initializer_list<std::initializer_list<double>> il);
        Value(Tensor* val);
        void Forward();
        void Backward();
};

//OPERATIONS


class Add : public Node
{
    public:
        Add(Node& x, Node& y) : Add(&x, &y){};
        Add(Node * x, Node * y);
        void Forward();
        //void Backward();
};

class Sub : public Node
{
    public:
        Sub(Node& x, Node& y) : Sub(&x, &y){};
        Sub(Node * x, Node * y);
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

class Dot : public Node
{
    public:
        Dot(Node& x, Node& y) : Dot(&x, &y){};
        Dot(Node * x, Node * y); 
        void Forward();
};

class Mul : public Node
{
    public:
        Mul(Node& x, Node& y) : Mul(&x, &y){};
        Mul(Node * x, Node * y); 
        void Forward();
};

