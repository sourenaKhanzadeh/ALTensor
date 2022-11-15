#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    ndarray<float> x({1, 2, 3});
    ndarray<float> y({1, 2, 3});

    y.fill(1.f);
    x.random(10, 20);


    std::cout << "x = " << x << std::endl;
    std::cout << "square(x)" << square(x) << std::endl;
    std::cout << "deriv(square, x)" << deriv(square, x) << std::endl;
    std::cout << "relu(x)" << relu(x) << std::endl;
    std::cout << "deriv(relu, x)" << deriv(relu, x) << std::endl;
    return 0;
}