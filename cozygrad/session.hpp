#pragma once

#include <vector>

namespace czy{

//class manages allocated memory and in future gpu stuff 
class node;

class Session
{
    public:
        static Session& get_session()
        {
            static Session instance;
            return instance;
        }

        void add_node(node* x);       

        std::vector<node*> get_session_nodes(){ return m_session_nodes;}
    private:
        Session(){}; // Default constructor
        ~Session(){}; // Destructor

        std::vector<node*> m_session_nodes;
};


//#include "node.hpp"


void Session::add_node(node* x)
{
    m_session_nodes.push_back(x);       
}

}//namespace czy


