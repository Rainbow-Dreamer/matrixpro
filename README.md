# matrixpro

This is a python module for handling matrices, including matrix calculation, analysis and algorithms.

All of the basic calculations of matrices and many more high-level numerical methods of matrices are included in this module, which could be used as easy-to-write functions.

This module is easy to be used as a toolkit in your projects (for example, this module will be useful in games and AI), and this module is also easy to extend for new functionality.

## Installation
You can use `pip` to install matrixpro, run the following line in cmd/terminal:
```python
pip install matrixpro
```
When the installation finished, matrixpro is installed on your computer, you can use the following line at the beginning of your code to use matrixpro:
```python
from matrixpro.matrix import *
```
or you can use
```python
import matrixpro.matrix as mp
```
to avoid duplicate function/class/variable names problems that could possibly occur.

## Usage
This will only be a synopsis of the whole functionality of matrixpro, which includes the most basic and important usages of matrixpro, for more detailed introductions of this module, refer to wiki.

### Create a matrix
There are many ways to create a matrix in matrixpro, the most basic way to create a matrix in matrixpro is passing a list of lists to `matrix` class, where lists are rows of the matrix, each list has the elements of the row.

For example, if we want to create a matrix
```
[1, 2]
[3, 4]
```
we can write
```python
matrix_A = matrix([[1,2], [3, 4]])
```
or for more readability,
```python
matrix_A = matrix([[1, 2],
                   [3, 4]])
```
We can print this matrix,
```python
>>> print(matrix_A)
[1, 2]
[3, 4]
```

If we want to quickly create a matrix of m x n size with a default value, we can use `build` function:
```python
build(row_number, column_number=None, element=0)

# row_number: the row number of the matrix

# column_number: the column number of the matrix, if not set, this will be the same as the row number

# element: the default element of all of the entries of the matrix, default value is 0

matrix_B = build(10, 5) # build a matrix of 10 rows and 5 columns with default value 0
>>> matrix_B
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
```

If we want to create a matrix which is filled with integers 1 ~ n, we can use `mrange` function:
```python
mrange(nrow, ncol=None, start=None, stop=None, default=0)

# nrow: the row number of the matrix

# ncol: the column number of the matrix, if not set, this will be the same as the row number

# start: the start of the elements

# stop: the last of the elements

# default: the default value of the elements that are not filled

# if both of start and stop are not set, the start value will be 1, the stop value will be nrow * ncol;

# if start is set and stop is not set, the start value will be 1, the stop value will be start;

# if both of start and stop are set, the start value will be start, the stop value will be stop;

# you canoot make start not set and stop set, which means if you want to set the stop value, you must set the start value as well

matrix_C = mrange(5) # generate a 5 x 5 square matrix with elements from 1 to 25
>>> matrix_C
[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

matrix_D = mrange(5, 5, 0, 24) # generate a 5 x 5 square matrix with elements from 0 to 24
>>> matrix_D
[0, 1, 2, 3, 4]
[5, 6, 7, 8, 9]
[10, 11, 12, 13, 14]
[15, 16, 17, 18, 19]
[20, 21, 22, 23, 24]
```

If we want to turn a list of elements into a matrix, we can use `form` function:
```python
form(val, nrow, ncol=None, default=0)
# the list that contains elements you want to turn into a matrix
# other parameters: refer to functions above

values = [1, 2, 3, 4 ,5, 6]
matrix_E = form(values, 2, 3)
>>> matrix_E
[1, 2, 3]
[4, 5, 6]
```

### Create special matrices
To create an identity matrix, you can use `identity` or `ids` function:
```python
>>> identity(5) # create an 5 x 5 identity matrix
[1, 0, 0, 0, 0]
[0, 1, 0, 0, 0]
[0, 0, 1, 0, 0]
[0, 0, 0, 1, 0]
[0, 0, 0, 0, 1]
```

To create a diagonal matrix, you can use `diagonal` function:
```python
diagonal(element, nrow=None, ncol=None)

# element: the list of elements at the diagonal from upper left corner to bottom right corner

# nrow, ncol: if not set, these will both be the length of element, you can set nrow and ncol separately

>>> diagonal([1, 2, 3]) # create a diagonal matrix with 1, 2, 3 on the diagonal
[1, 0, 0]
[0, 2, 0]
[0, 0, 3]
```

To create a square matrix, you can use `square` function:
```python
# square function is basically build function when the row number is equal to the column number
>>> square(5) # create a 5 x 5 square matrix with default value 0
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
```

### Transpose of a matrix
To get the transpose of a matrix, you can use `transpose` or `T` function of matrix.
```python
matrix_F = matrix([[1,2,3], [4,5,6]])

>>> matrix_F
[1, 2, 3]
[4, 5, 6]

>>> matrix_F.transpose()
[1, 4]
[2, 5]
[3, 6]

>>> matrix_F.T()
[1, 4]
[2, 5]
[3, 6]
```

### Matrix addition, subtraction, multiplication and division
The usages of calculations of matrix in matrixpro is similar as the calculations of integers and floats.

To add 2 matrices A and B, you can write `A + B`.  
To subtract matrix B from matrix A, you can write `A - B`.  
To multiply matrix A and matrix B, you can write `A * B`.  
To divide matrix A by matrix B, you can write `A / B`.
