#pragma once

#include "node.hpp"

#include <vector>

class node;
//class manages allocated memory and in future gpu stuff 
class Session
{
    public:
        static Session& get_session()
        {
            static Session instance;
            return instance;
        }

        void add_node(node* x)
        {
            m_session_nodes.push_back(x);
        };

        std::vector<node*> get_session_nodes(){ return m_session_nodes;}
    private:
        Session(){}; // Default constructor
        ~Session(){}; // Destructor

        std::vector<node*> m_session_nodes;
};