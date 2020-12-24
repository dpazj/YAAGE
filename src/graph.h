#pragma once

#include "node.h"
#include "tensor.h"

#include <unordered_set>

class Graph
{

    public:
        Graph(Node& input, Node& output) : Graph(&input, &output){};
        Graph(Node * input, Node * output);
        //Graph(std::vector<Node*> inputs, std::vector<Node*> outputs);
        Tensor* Forward();
        //void Backward();

    private:
        Node* m_input_node;
        Node* m_output_node;

        void TraverseBackwards(Node * node);
        std::unordered_set<Node*> m_exec_order;


};