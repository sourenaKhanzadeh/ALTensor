#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    NDArray<int> x({1, 2, 3});
    NDArray<int> y({2, 3, 1});

    y.fill(1);
    x.random(10, 20);


    std::cout << "x = " << x << std::endl;
    std::cout << "x = " << x << std::endl;
    return 0;
}