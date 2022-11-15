#include <iostream>
#include <deriv.h>
#include <ndarray.h>

int main() {

    NDArray<double> x({2, 3, 4, 2});

    x.random();

    std::cout << x << std::endl;
    std::cout << x[0][0][0] << std::endl;

    return 0;
}