A cozy header-only autograd engine written from scratch in c++. Cozygrad evaluates directed acyclic graphs and computes their gradients. Graphs are constructed with a simple and easy to use (pytorch ish?) API. 

## example

```c++
#include "cozygrad/cozygrad.h"

int main()
{
    tensor t = {-8.0};

    node x(t);
    auto& a = 42 - x;
    auto& b = a.pow(2) + a;
    auto& c = b.log() * x;
    auto& y = -c;

    //construct a graph
    graph g(y);
    g.forwards();
    g.backwards();

    y.data()->print(); // 62.7508
    x.gradient()->print(); // dy/dx -8.16071
}
```

## pytorch
```python
import torch

x = torch.Tensor([-8.0]).double()
x.requires_grad = True

a = 42 - x
b = a.pow(2) + a
c = b.log() * x
y = -c
y.backward()

print(y.data) # 62.7508
print(x.grad) # dy/dx -8.1607

```

## constructing a neural network
Simpy extend the model class and implement the create_model function
```c++
class MyNet : public model
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

int main()
{
    //... code to get dataset X and y ...
    double learning_rate = 0.05;
    SDG sdg(learning_rate);
    MyNet model;
    model.train(X,y, sdg);
}


```

## TODO
* Get simple MNIST working & training.
* Make tensor class better - n dimensions, broadcasting etc
* GPU support!


