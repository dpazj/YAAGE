#include <iostream>
#include <iomanip>

#include <fstream>
#include <sstream>
#include <string>
#include <stdlib.h>

#include "cozygrad/cozygrad.h"



void sanity_test()
{
    tensor input = {-4.0};
    tensor input1 = {2.0};
    
    node x(input);
    node two(input1);

    node z = two * x + two + x;
    node q = z.relu() + z * x;
    node h = (z * z).relu();
    node y = h + q + q * x; 

    graph g(x,y);
    g.forwards();
    g.backwards();

    std::cout << "y should be -20, y = ";   y.data()->print();
    std::cout << "dy/dx should be 46, dy/dx = "; x.gradient()->print();
}

node& test1()
{
    tensor* input = new tensor({-4.0});
    tensor* input1 = new tensor({34.0});

    node* x = new node(input);
    node* x1 = new node(input1);

    auto& z = *x1 - *x;
    auto& c = z.pow(2);
    auto& y = c - *x1;

    // graph g(*x,y);
    // g.forwards();
    // g.backwards();

    // std::cout << "y should be 1410, y = ";   y.data()->print();
    // std::cout << "dy/dx should be -76, dy/dx = "; x->gradient()->print();
    return y;

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

class MoonNet : model
{
    node create_model()
    {
        node input = create_input_node();

        node w1 = create_model_param(2,16);
        node w2 = create_model_param(16,16);
        node w3 = create_model_param(16,1);

        node b1 = create_model_param(1,16);
        node b2 = create_model_param(1,16);
        node b3 = create_model_param(1,1);

        node l1 = (input.dot(w1) + b1).relu(); //;layer 1
        node l2 = (l1.dot(w2) + b2).relu();
        node l3 = (l2.dot(w3) + b3).sigmoid();

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
        tensor tmp1 = {atof(vals[2].c_str()) * 2 - 1};
        X.push_back(tmp);
        y.push_back(tmp1); 
    }



    // process pair (a,b)
}



int main()
{
    //do_moon();
    // sanity_test();
    auto& y = test1();
    graph g(y,y);
    g.forwards();
    //g.backwards();

    std::cout << "y should be 1410, y = ";   y.data()->print();
    //std::cout << "dy/dx should be -76, dy/dx = "; x.gradient()->print();



    //test2();

    utils::clean_session();
    
    return 0;

}

