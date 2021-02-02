#pragma once

#include <vector>

namespace czy{

//class manages allocated memory and in future gpu stuff 
template <typename T>
class node;

template <typename T>
class Session
{
    public:

        static Session<T>& get_session()
        {
            static Session<T> instance;
            return instance;
        }

        void add_node(node<T>* x);       

        std::vector<node<T>*>& get_session_nodes(){ return m_session_nodes;}


    private:
        Session(){}; // Default constructor
        ~Session(){}; // Destructor

        std::vector<node<T>*> m_session_nodes;
};

template <typename T>
void Session<T>::add_node(node<T>* x)
{
    m_session_nodes.push_back(x);       
}

}//namespace czy


