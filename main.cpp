#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    ndarray<int> x({1, 2, 3});
    ndarray<int> y({2, 3, 1});

    y.fill(1);
    x.random(10, 20);


    std::cout << "x = " << x << std::endl;
    x.transpose();
    std::cout << "x = " << x << std::endl;
    x.flatten();
    std::cout << "x = " << x << std::endl;
    return 0;
}