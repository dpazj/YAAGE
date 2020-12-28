#pragma once

#include "node.h"
#include "matrix.h"

#include <unordered_set>

class Graph
{

    public:
        Graph(Node& input, Node& output) : Graph(&input, &output){};
        Graph(Node * input, Node * output);

        Matrix* Forward();
        Matrix* Backward();

    private:
        Node* m_input_node;
        Node* m_output_node;

        void PopulateExecOrder(Node * node);
 
        std::unordered_set<Node*> m_visited;
        std::vector<Node *> m_exec_order;

};
