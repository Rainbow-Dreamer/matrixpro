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

### Get row number and column number of the matrix
You can directly get the attributes `row_number` and `column_number` of the matrix object, these attributes are calculated once the matrix object is build.  
For short, you can use `rown` function of the matrix object to get the row number of the matrix, and `coln` function of the matrix object to get the column 
number of the matrix.  
You can use `dim` function of the matrix object to get the dimension (size) of the matrix in a tuple `(m, n)`, where `m` is row number and `n` is column number.  
You can use `size` function of the matrix object to get the dimension (size) of the matrix in a string `mxn`, where `m` is row number and `n` is column number.
```python
matrix_A = build(5, 7)

>>> matrix_A
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]
[0, 0, 0, 0, 0, 0, 0]

>>> matrix_A.row_number # get the attribute 'row_number' of the matrix
5

>>> matrix_A.column_number # get the attribute 'column_number' of the matrix
7

>>> matrix_A.rown() # get the row number of the matrix
5

>>> matrix_A.coln() # get the column number of the matrix
7

>>> matrix_A.dim() # get the dimension (size) of the matrix in a tuple
(5, 7)

>>> matrix_A.size() # get the dimension (size) of the matrix in a string
'5x7'
```

### Matrix addition, subtraction, multiplication and division
The usages of calculations of matrix in matrixpro is similar as the calculations of integers and floats.

To add 2 matrices A and B, you can write `A + B`.  
To subtract matrix B from matrix A, you can write `A - B`.  
To multiply matrix A and matrix B, you can write `A * B`.  
To divide matrix A by matrix B, you can write `A / B`.

```python
matrix_G = matrix([[1, 2], [3, 4]])
matrix_H = matrix([[5, 6], [7, 8]])

>>> matrix_G
[1, 2]
[3, 4]

>>> matrix_H
[5, 6]
[7, 8]

>>> matrix_G + matrix_H
[6, 8]
[10, 12]

>>> matrix_G - matrix_H
[-4, -4]
[-4, -4]

>>> matrix_G * matrix_H
[19, 22]
[43, 50]

>>> matrix_G / matrix_H
[3.000000000000009, -2.0000000000000067]
[2.0000000000000018, -1.0000000000000018]

# if the division result has too many digits after the decimal point and it is annoying for you,
# you can use 'formated' function of matrix to round the floats to a given precision.
# For more details about 'formated' function of matrix, refer to wiki.

>>> (matrix_G / matrix_H).formated() # using default formated parameters
[3, -2]
[2, -1]
```

### Get a row, column or element of the matrix
The syntax to get an element of the matrix is `matrix[row_number, column_number]`, or you can also get the element by treating the matrix object 
as list of lists, which is `matrix[row_number][column_number]`. The row number and column number are both 0-based (start from 0). 
The indexing of the row number and the column number are the same as list in python. For example:
```python
matrix_A = matrix([[1, 2], [3, 4]])

>>> matrix_A
[1, 2]
[3, 4]

>>> matrix_A[0, 0] # get the element at first row and first column of the matrix
1

>>> matrix_A[1, 1] # get the element at second row and second column of the matrix
4

>>> matrix_A[-1, -1] # get the element at the last row and the last column of the matrix
4
```

To get a row of the matrix, you can write `matrix[row_number]`, which will return a list which is the corresponding row of the matrix with the row number.  
To get a column of the matrix, you can write `matrix[column_number,]`, which will return a list which is the corresponding column of the matrix with the column number.  
For example:
```python
>>> matrix_A[0] # get the first row of the matrix
[1, 2]

>>> matrix_A[0,] # get the first row of the matrix
[1, 3]
```

### Modify rows, columns and elements of the matrix
You can modify the rows, columns and elements by assigning to a list (for rows and columns) or element (for elements) when you are getting them.  
For example:
```python
matrix_A = mrange(3)

>>> matrix_A
[1, 2, 3]
[4, 5, 6]
[7, 8, 9]

matrix_A[0] = [10, 10, 10] # change the first row of the matrix to [10, 10, 10]
>>> matrix_A
[10, 10, 10]
[4, 5, 6]
[7, 8, 9]

matrix_A[0,] = [10, 10, 10] # change the first column of the matrix to [10, 10, 10]
>>> matrix_A
[10, 10, 10]
[10, 5, 6]
[10, 8, 9]

matrix_A[2, 2] = 20 # change the element at the 3rd row and the 3rd column to 20
>>> matrix_A
[10, 10, 10]
[10, 5, 6]
[10, 8, 20]
```

### Calculate power of a matrix
You can use `matrix ** n` or `matrix ^ n` to calculate the nth power of a matrix. Here `n` could be an integer, a float or a fraction.
```python
matrix_A = matrix([[1, 2], [3, 4]])

>>> matrix_A
[1, 2]
[3, 4]

>>> matrix_A ^ 2
[7, 10]
[15, 22]

>>> matrix_A ** 2
[7, 10]
[15, 22]

>>> matrix_A ^ -2
[5.499999999999985, -2.499999999999993]
[-3.74999999999999, 1.7499999999999953]
```

### Convenient syntax for adding/subtracting/multiplying/dividing every element in a matrix
You can simply add/subtract/multiply/divide a number from the matrix, and the calculations will be applied to every element in the matrix.  
For example, if you write `matrix + 1`, then you will get a new matrix object with every element in the matrix increased by 1.  
The same logic applies for subtraction, multiplication and division. Here are some examples:
```python
matrix_A = matrix([[1, 2], [3, 4]])

>>> matrix_A
[1, 2]
[3, 4]

>>> matrix_A + 1
[2, 3]
[4, 5]

>>> matrix_A - 1
[0, 1]
[2, 3]

>>> matrix_A * 2
[2, 4]
[6, 8]

>>> matrix_A / 2
[0.5, 1.0]
[1.5, 2.0]
```

### Calculate the determinant of a matrix
You can use `det` function of matrix object to calculate the determinant of a matrix.
```python
matrix_A = matrix([[1, 2], [3, 4]])

>>> matrix_A
[1, 2]
[3, 4]

>>> matrix_A.det()
-2.0
```

### Calculate the inverse of a matrix
There are a few of algorithms to calculate the inverse of a matrix implemented in matrixpro. The default method is the adjoint method, which is a very 
common method for calculating the inverse of a matrix **by hand**, it is not very efficient when it comes to large matrices. There are some numerical 
methods for calculating the inverse of a matrix implemented in matrixpro, including LU decomposition method and Newton iteration method.  
After some tests of large matrices, it shows that LU decomposition method has the fastest speed for calculating the inverse of a matrix, and also remains 
good stability.

To use default inverse method, you can use `inverse` or `inv` function of the matrix object.  
To use LU decomposition method, you can use `inv_lu` function of the matrix object.  
To use Newton iteration method, you can use `inv_newton` function of the matrix object.

You can also calculate the generalized inverse (pseudoinverse) of a m x n matrix by using `pinv` function of the matrix object.

```python
matrix_A = matrix([[1, 2], [3, 4]])

>>> matrix_A
[1, 2]
[3, 4]

>>> matrix_A.inverse() # using the default adjoint method
[-2.0, 1.0]
[1.5, -0.5]

>>> matrix_A.inv_lu() # using the LU decomposition method
[-2.0, 1.0]
[1.5, -0.5]

>>> matrix_A.inv_newton(tol=1e-3) # using the Newton iteration method
[-1.9996027561324514, 0.9998242712755511]
[1.499720054620951, -0.49987616059455053]

matrix_B = mrange(2, 3)

>>> matrix_B
[1, 2, 3]
[4, 5, 6]

>>> matrix_B.pinv() # calculate pseudoinverse of a non-square matrix
[-0.9444444444444433, 0.444444444444444]
[-0.11111111111111072, 0.11111111111111116]
[0.7222222222222219, -0.22222222222222188]
```

### Calculate the (reduced) row echelon form of a matrix
You can use `ref` function of the matrix object to calculate the row echelon form of a matrix, and use `rref` function of the matrix object to calculate 
the reduced row echelon form of a matrix.
```python
matrix_A = mrange(5)

>>> matrix_A
[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

>>> matrix_A.ref() # calculate the row echelon form of the matrix
[1, 2, 3, 4, 5]
[0, 1, 2, 3, 4]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]

>>> matrix_A.rref() # calculate the reduced row echelon form of the matrix
[1, 0, -1, -2, -3]
[0, 1, 2, 3, 4]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
```

### Calculate the eigenvalues and eigenvectors of a matrix
You can use `eigen` function of the matrix object to calculate the eigenvalues and the eigenvectors of a matrix.  
The return values of `eigen` function are 2 matrices, the first one is a diagonal matrix with eigenvalues on the diagonal, 
the second one is a matrix where each column is one of the eigenvectors.

To get only eigenvalues or eigenvectors of a matrix, you can use `eigval` function of the matrix object for eigenvalues, 
`eigvec` function of the matrix object for eigenvectors. Note that `eigval` function will return a list of eigenvalues.

```python
matrix_A = mrange(5)

>>> matrix_A
[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

eigenvalues, eigenvectors = matrix_A.eigen() # calculate the eigenvalues and the eigenvectors of the matrix

>>> eigenvalues
[68.62358776522458, 0, 0, 0, 0]
[0, -3.62358776522458, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]
[0, 0, 0, 0, 0]

>>> eigenvectors
[0.10797497846502196, 0.6749528825442898, 0.14744195615489714, 0.29488391230979427, 0.4423258684646914]
[0.25277500450721396, 0.3603897570511307, -0.29488391230979427, -0.4423258684646914, -0.5897678246195885]
[0.3975750305494059, 0.045826631557971335, 0.14744195615489714, 0.0, 0.0]
[0.5423750565915979, -0.26873649393518717, 0.0, 0.14744195615489714, 0.0]
[0.6871750826337898, -0.5832996194283467, 0.0, 0.0, 0.14744195615489714]

>>> matrix_A.eigval()
[68.62358776522458, -3.623587765224578, 3.314793091830836e-16, -1.5318106288467553e-16, 3.9624216894511006e-16]

>>> matrix_A.eigvec()
[0.10797497846502196, 0.6749528825442898, 0.14744195615489714, 0.29488391230979427, 0.4423258684646914]
[0.25277500450721396, 0.3603897570511307, -0.29488391230979427, -0.4423258684646914, -0.5897678246195885]
[0.3975750305494059, 0.045826631557971335, 0.14744195615489714, 0.0, 0.0]
[0.5423750565915979, -0.26873649393518717, 0.0, 0.14744195615489714, 0.0]
[0.6871750826337898, -0.5832996194283467, 0.0, 0.0, 0.14744195615489714]
```

### Some other functionalities
You can use `rank` function of the matrix object to calculate the rank of a matrix.
```python
matrix_A = mrange(5)
>>> matrix_A.rank()
2
```

You can use `trace` function of the matrix object to calculate the trace of a matrix.
```python
matrix_A = mrange(5)
>>> matrix_A.trace()
65
```

You can use `flip` function of the matrix object to flip a matrix vertically or horizontally.

The parameter `mode` of `flip` function determines to flip horizontally or vertically. 
If the parameter `mode` is 0, flip the matrix horizontally, otherwise, flip the matrix vertically. 
The parameter `mode` is 0 as default.
```python
matrix_A = mrange(5)

>>> matrix_A
[1, 2, 3, 4, 5]
[6, 7, 8, 9, 10]
[11, 12, 13, 14, 15]
[16, 17, 18, 19, 20]
[21, 22, 23, 24, 25]

>>> matrix_A.flip() # flip the matrix horizontally
[5, 4, 3, 2, 1]
[10, 9, 8, 7, 6]
[15, 14, 13, 12, 11]
[20, 19, 18, 17, 16]
[25, 24, 23, 22, 21]

>>> matrix_A.flip(1)
[21, 22, 23, 24, 25]
[16, 17, 18, 19, 20]
[11, 12, 13, 14, 15]
[6, 7, 8, 9, 10]
[1, 2, 3, 4, 5]
```
