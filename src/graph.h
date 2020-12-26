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
        Tensor* Backward();

    private:
        Node* m_input_node;
        Node* m_output_node;

        void PopulateExecOrder(Node * node);
 
        std::unordered_set<Node*> m_visited;

        void VisitBackwards(Node * node);
        void VisitForwards(Node * node);


};