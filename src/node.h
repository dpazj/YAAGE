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

        virtual const std::string Name() = 0;
        virtual void Forward() = 0;
        virtual void Backward() = 0;  

        Matrix* Data();
        Matrix* Gradient();
        std::vector<Node*> Children();

        void AllocateGradientMem(size_t row, size_t col, double val = 0.0f);

    protected:
        void Connect(Node* node);
                
        std::vector<Node*> m_input_nodes;
        std::vector<Node*> m_output_nodes;
        
        Matrix* m_data = nullptr;
        Matrix* m_gradient = nullptr;

        bool m_owns_memory = true;
    private:
        void AddOutput(Node * node);
        
};

class TwoInputNode : public Node
{
    public:
        TwoInputNode(Node& x, Node& y) : TwoInputNode(&x, &y){};
        TwoInputNode(Node * x, Node * y);
};

//Value
class Value : public Node
{
    public:
        Value(Matrix& val) : Value(&val){};
        Value(std::initializer_list<double> il);
        Value(std::initializer_list<std::initializer_list<double>> il);
        Value(Matrix* val);

        void Forward();
        void Backward();


        const std::string Name();
};

//OPERATIONS

class Add : public TwoInputNode
{
    
    public:
        Add(Node& x, Node& y) : Add(&x, &y){};
        Add(Node * x, Node * y) : TwoInputNode(x,y){};
        void Forward();
        void Backward();
        const std::string Name();
};

class Sub : public TwoInputNode
{
    public:
        Sub(Node& x, Node& y) : Sub(&x, &y){};
        Sub(Node * x, Node * y) : TwoInputNode(x,y){};
        void Forward();
        void Backward();
        const std::string Name();
};

class Pow : public Node
{
    public:
        Pow(Node& x, double exponent) : Pow(&x, exponent){};
        Pow(Node* x, double exponent);
        void Forward();
        void Backward();
        const std::string Name();
    private:
        double m_exponent;
};

class Dot : public TwoInputNode
{
    public:
        Dot(Node& x, Node& y) : Dot(&x, &y){};
        Dot(Node * x, Node * y) : TwoInputNode(x,y){};
        void Forward();
        void Backward();
        const std::string Name();
};

class Mul : public TwoInputNode
{
    public:
        Mul(Node& x, Node& y) : Mul(&x, &y){};
        Mul(Node * x, Node * y) : TwoInputNode(x,y){};
        void Forward();
        void Backward();
        const std::string Name();
};

class Sum : public Node
{
    public:
        Sum(Node& x) : Sum(&x){};
        Sum(Node * x); 
        void Forward();
        void Backward();
        const std::string Name();
};

class ReLU : public Node
{
    public:
        ReLU(Node& x) : ReLU(&x){};
        ReLU(Node * x); 
        void Forward();
        void Backward();
        const std::string Name();
};