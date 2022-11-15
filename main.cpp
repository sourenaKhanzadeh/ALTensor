#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    ndarray<int> x({1, 2, 3});
    ndarray<int> y({1, 2, 3});

    y.fill(1);
    x.random(10, 20);


    std::cout << "x = " << x << std::endl;
    std::cout << "y = " << y << std::endl;
    std::cout << "x + y = " << x + y << std::endl;
    std::cout << "x - y = " << x - y << std::endl;
    std::cout << "x * y = " << x * y << std::endl;
    std::cout << "x / y = " << x / y << std::endl;
    std::cout << "x / 2 = " << x / 2 << std::endl;
    std::cout << "x * 2 = " << x * 2 << std::endl;
    return 0;
}