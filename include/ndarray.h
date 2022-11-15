#ifndef NDARRAY_H
#define NDARRAY_H

#include <vector>
#include <sstream>
#include <random>

template <typename T>
class NDArray {
    public:
        NDArray(std::vector<int> shape);
        NDArray(std::vector<int> shape, std::vector<T> data);
        ~NDArray();
        T& operator[](std::vector<int> index);
        NDArray<T> operator[](int index);
        // print the array
        friend std::ostream& operator<<(std::ostream& os, const NDArray<T>& arr) {
            os << arr.str();
            return os;
        }
        int size();
        int size(int dim);
        int rank();
        std::vector<int> shape();
        std::vector<int> strides();
        void reshape(std::vector<int> shape);
        void resize(std::vector<int> shape);
        void resize(int size);
        void resize(int size, int dim);
        void resize(int size, int dim, T value);
        void resize(int size, int dim, T value, bool copy);
        void resize(int size, int dim, bool copy);
        void resize(int size, T value);
        void resize(int size, T value, bool copy);
        void resize(int size, bool copy);
        void resize(std::vector<int> shape, T value);
        void resize(std::vector<int> shape, T value, bool copy);
        void resize(std::vector<int> shape, bool copy);
        void resize(T value);
        void resize(T value, bool copy);
        void resize(bool copy);
        void fill(T value);
        void fill(T value, bool copy);
        void fill(bool copy);
        void copy(NDArray<T> &other);

        // return a string representation of the array
        std::string str();
        std::string str() const;
        void random();

    private:
        std::vector<T> data;
        std::vector<int> shape_;
        std::vector<int> strides_;
        int size_;
        int rank_;
};

// implementation
template <typename T>
NDArray<T>::NDArray(std::vector<int> shape) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    data.resize(size_);
}

template <typename T>
NDArray<T>::NDArray(std::vector<int> shape, std::vector<T> data) {
    shape_ = shape;
    rank_ = shape.size();
    strides_.resize(rank_);
    strides_[rank_ - 1] = 1;
    size_ = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
        size_ *= shape_[i];
    }
    size_ *= shape_[0];
    this->data = data;
}

template <typename T>
NDArray<T>::~NDArray() {
    data.clear();
    shape_.clear();
    strides_.clear();
}

template <typename T>
T& NDArray<T>::operator[](std::vector<int> index) {
    int offset = 0;
    for (int i = 0; i < rank_; i++) {
        offset += index[i] * strides_[i];
    }
    return data[offset];
}

template <typename T>
NDArray<T> NDArray<T>::operator[](int index) {
    std::vector<int> new_shape(shape_.begin() + 1, shape_.end());
    std::vector<T> new_data(data.begin() + index * strides_[0], data.begin() + (index + 1) * strides_[0]);
    return NDArray<T>(new_shape, new_data);
}

template <typename T>
std::string NDArray<T>::str() {
    // recursively print the array
    std::stringstream ss;
    if (rank_ == 1) {
        ss << "[";
        for (int i = 0; i < size_; i++) {
            ss << data[i];
            if (i < size_ - 1) {
                ss << ", ";
            }
        }
        ss << "]";
    } else {
        ss << "[";
        for (int i = 0; i < shape_[0]; i++) {
            ss << NDArray<T>(std::vector<int>(shape_.begin() + 1, shape_.end()), data).str();
            if (i < shape_[0] - 1) {
                ss << ", ";
            }
        }
        ss << "]";
    }
    return ss.str();
}

template <typename T>
std::string NDArray<T>::str() const {
    // recursively print the array
    std::stringstream ss;
    if (rank_ == 1) {
        ss << "[";
        for (int i = 0; i < size_; i++) {
            ss << data[i];
            if (i < size_ - 1) {
                ss << ", ";
            }
        }
        ss << "]";
    } else {
        ss << "[";
        for (int i = 0; i < shape_[0]; i++) {
            ss << NDArray<T>(std::vector<int>(shape_.begin() + 1, shape_.end()), data).str();
            if (i < shape_[0] - 1) {
                ss << ", ";
            }
        }
        ss << "]";
    }
    return ss.str();
}

template <typename T>
int NDArray<T>::size() {
    return size_;
}

template <typename T>
int NDArray<T>::size(int dim) {
    return shape_[dim];
}

template <typename T>
int NDArray<T>::rank() {
    return rank_;
}

template <typename T>
std::vector<int> NDArray<T>::shape() {
    return shape_;
}

template <typename T>
std::vector<int> NDArray<T>::strides() {
    return strides_;
}

template <typename T>
void NDArray<T>::reshape(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    if (size != size_) {
        throw "Shape size does not match array size";
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size) {
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, T value, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, int dim, bool copy) {
    if (dim >= rank_) {
        throw "Dimension out of range";
    }
    shape_[dim] = size;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(T value) {
    data.resize(size_, value);
}

template <typename T>
void NDArray<T>::resize(T value, bool copy) {
    data.resize(size_, value);
}

template <typename T>
void NDArray<T>::resize(bool copy) {
    data.resize(size_);
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, T value, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size, value);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(std::vector<int> shape, bool copy) {
    if (shape.size() != rank_) {
        throw "Shape size does not match rank";
    }
    int size = 1;
    for (int i = 0; i < rank_; i++) {
        size *= shape[i];
    }
    shape_ = shape;
    strides_[rank_ - 1] = 1;
    for (int i = rank_ - 1; i > 0; i--) {
        strides_[i - 1] = strides_[i] * shape_[i];
    }
    data.resize(size);
    size_ = size;
}

template <typename T>
void NDArray<T>::resize(int size, T value) {
    data.resize(size, value);
}

template <typename T>
void NDArray<T>::resize(int size, T value, bool copy) {
    data.resize(size, value);
}

template <typename T>
void NDArray<T>::resize(int size, bool copy) {
    data.resize(size);
}

template <typename T>
void NDArray<T>::fill(T value) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(T value, bool copy) {
    std::fill(data.begin(), data.end(), value);
}

template <typename T>
void NDArray<T>::fill(bool copy) {
    std::fill(data.begin(), data.end(), 0);
}

template <typename T>
void NDArray<T>::copy(NDArray<T> &other) {
    if (other.size_ != size_) {
        throw "Size mismatch";
    }
    data = other.data;
}

template <typename T>
void NDArray<T>::random() {
    // fill with random values
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(0, 1);
    for (int i = 0; i < size_; i++) {
        data[i] = dis(gen);
    }
}

#endif