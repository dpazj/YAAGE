#include <iostream>
#include <iomanip>

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "cozygrad/node.hpp"
#include "cozygrad/graph.hpp"
#include "cozygrad/tensor.hpp"
#include "cozygrad/utils.hpp"
#include "cozygrad/model.hpp"




void sanity_test()
{
    tensor input = {-4.0};
    
    node x(input);

    auto& z = 2 * x + 2 + x;
    auto& q = z.relu() + z * x;
    auto& h = (z * z).relu();
    auto& y = h + q + q * x; 

    graph g(x,y);
    g.forwards();
    g.backwards();

    std::cout << "y should be -20, y = ";   y.data()->print();
    std::cout << "dy/dx should be 46, dy/dx = "; x.gradient()->print();
}

void test1()
{
    tensor* input = new tensor({-4.0});
    
    node x(input);

    auto& z = 34 - x;
    auto& c = z.pow(2);
    auto& y = c - 34;

    graph g(x,y);
    g.forwards();
    g.backwards();

    std::cout << "y should be 1410, y = ";   y.data()->print();
    std::cout << "dy/dx should be -76, dy/dx = "; x.gradient()->print();
}

void test2()
{
    tensor input = {24.0f};
    node x(input);
    node y = x.sigmoid();
    graph g(x,y);
    g.forwards();
    g.backwards();

    y.data()->print();
    x.gradient()->print();
}

class MoonNet : public model
{
    node& create_model()
    {
        auto& input = create_input_node();

        auto& w1 = create_model_param(2,16);
        auto& w2 = create_model_param(16,16);
        auto& w3 = create_model_param(16,1);

        auto& b1 = create_model_param(1,16);
        auto& b2 = create_model_param(1,16);
        auto& b3 = create_model_param(1,1);

        auto& l1 = (input.dot(w1) + b1).relu(); //layer 1
        auto& l2 = (l1.dot(w2) + b2).relu(); //layer 2
        auto& l3 = (l2.dot(w3) + b3); //layer 3
        
        return l3;
    }
  
};

std::vector<std::string> split(const std::string &s, char delim) {
    std::vector<std::string> elems;
    std::stringstream ss(s);
    std::string item;
    while(std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
    return elems;
}

void do_moon()
{

    std::vector<tensor> X;
    std::vector<tensor> y;

    std::ifstream moon("./datasets/moon_dataset.txt");
    std::string line;
    while (std::getline(moon, line))
    {
        auto vals = split(line, ',');
        tensor tmp = {atof(vals[0].c_str()), atof(vals[1].c_str())};
        tensor tmp1 = {atof(vals[2].c_str()) * 2 -1};
        X.push_back(tmp);
        y.push_back(tmp1); 
    }

    MoonNet model;

    model.train(X,y);

    // process pair (a,b)
}





int main()
{
    //sanity_test();
    //test1();
    //test2();

    do_moon();

    Session& session = Session::get_session();
    for(node* x : session.get_session_nodes())
    {
        delete x;
    }
    
    return 0;

}

