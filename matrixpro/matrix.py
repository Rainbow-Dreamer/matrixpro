import time, random, fractions
from copy import deepcopy
'''    
This is a module handling with matrix attributes, transformations and calculations
written by me. Feel free to use it when you encounter large matrix calculations
and transfomations in your projects or assignments. Moerover, this module
holds for neseted matrices, which means multi-dimensional matrices/arrays.
You can set the attributes of the higher dimensions to be any class you want,
which could help you to deal with a sort of scientific analysis like machine learning.

Here I am writing a help instruction as detailed as I could.

Firstly, for building up a matrix, it is pretty easy, for example,
a = matrix([[1,2,3],[6,7,8]]) will build up a matrix named a with 2 rows and 3 columns,
its first row's elements are 1, 2, 3 in order, second row's elements are 6, 7, 8 in order.

I write many built-in functions in matrix module that you can use to handle with various of
attributes, transformations and calculations of the matrix you have set up.

For detailed instrcuctions of each function,
please look at the help docs of each function below.

Here I will give you some examples of the functions I write to give you
the informations and help you quickly calculate stuffs of matrices.

Here we assume you set up a matrix a = matrix([[1,2,3],[6,7,8]]).

>>> a = matrix([[1,2,3],[6,7,8]])

(note here I assume if this matrix module is imported as "from matrix import *",
which means the namespace of matrix module is ignored, but if you import matrix
as m(for example), you should call the function like m.func(a) instead of
just func(a), and it is recommended to import matrix as something otherwise
you will potentially interrupt with the functions defined in other modules
or python built-in functions)

e.g.(1) print(a) will give you a pretty straightforward matrix representation
        like what you regularly see on textbooks.
>>> print(a)
>>> [1, 2, 3]
    [6, 7, 8]
>>>

e.g.(2) inverse(a) or a.inverse() gives you the inverse matrix of matrix a
        if such inverse exists otherwise it will give you the reason why such inverse
        does not exist.

e.g.(3) det(a) or a.det() gives you the determinant of matrix a if it has
        determinant otherwise gives you the reason why matrix a does not have determinant.

e.g.(4) ref(a) or a.ref() gives you the row echelon form of matrix a, rref(a)
        or a.rref() gives you reduced row echelon form of matrix a.

e,g.(5) you can do calculations between matrices such as multiply: a*b,
        addition: a+b, subtraction: a-b, division: a/b, power: a**n (n is the power),
        check if equal: a == b, etc.

For more details, please see the help documentations of each function I design in this module.
'''

START = 'start'
END = 'end'
alert = 'this matrix cannot be created, please ensure to use a list filling with one or more lists to build this matrix, for example: [[1,2],[5,6]] is a valid construction which builds up a matrix with 2 rows and 2 columns, the first row\'s elements are 1,2, the second row\'s elements are 5,6'
alert2 = 'this matrix cannot be created or is considered as an empty matrix, please ensure to use a list filling with one or more lists to build this matrix, for example: [[1,2],[5,6]] is a valid construction which builds up a matrix with 2 rows and 2 columns, the first row\'s elements are 1,2, the second row\'s elements are 5,6'
alert3 = 'this matrix cannot be created, please ensure every row have the same number of elements'


class matrix:
    '''this class represents a matrix object containing some basic informations of matrix,
       and all of the functions with matrices in this module can be called on this class.
    
    Attributes:
        row: A list contained one or more sublists to store the data of the matrix.
             each sublist in the list represents a single row in a matrix.
        row_number: An integer represents the row number of the matrix.
        column_number: An integer represents the column number of the matrix.
        
    '''
    def __init__(self,
                 row,
                 rownames=[],
                 colnames=[],
                 formated=False,
                 asfloat=False,
                 dr=0):
        '''
        Initialize the matrix with given rows (lists) in a list.
        
        For example, a = matrix([[1,2,3],[6,7,8]]) will build up a matrix
        named a with 2 rows and 3 columns, its first row's elements are 1, 2, 3
        in order, second row's elements are 6, 7, 8 in order.
        
        If the built-up information is valid, i.e. each row should
        have exactly same number of elements; the user is using lists inside
        a list to build up the matrix; the matrix have at least one element
        (otherwise this module will give the user a warning with empty matrix,
        anyway, empty matrix is considered as a valid kind of matrix in Math).
        
        If the information is invalid, this module will give you the reason
        why this matrix is invalid to be exist.
        
        P.S. You can make the elements of matrix be any kind of values,
        integers, floats, fractions, strings and so on, but if the value is not
        a number, this module has nothing to do with its calculations.
        
        Parameters:
          row: Expect a list contained lists to build up a matrix successfully.
               The elements in the sublists have no type restrictions to set up.
        
        Returns:
          A matrix object contained basic informations: row number, column number,
          all of the rows in the matrix.
        
        Raises:
          ValueError: this error will raise in situations with invalid information
          to built up a matrix. This error raises when: the built-up information
          is not a list contained with one or more lists; the matrix does not
          contain any elements; the sublists in list does not have equal number
          of elements. Once this error raises, the users will see the specific
          corresponding reason why the given built-up information is invalid
          for a matrix in every case of invalidity.
        '''
        if row == []:
            self.row = row
            self.row_number = 0
            self.column_number = 0
            self.rownames = rownames
            self.colnames = colnames
        else:
            if any(not isinstance(y, list) for y in row):
                if dr == 0:
                    row = [row]
                else:
                    row = [[x] for x in row]
            if not isinstance(row, list):
                raise ValueError(alert)
            elif all(len(x) == 0 for x in row):
                raise ValueError(alert2)
            elif len(set([len(t) for t in row])) != 1:
                raise ValueError(alert3)
            else:
                self.row = row
                self.row_number = len(row)
                self.column_number = len(row[0])
                self.rownames = rownames
                self.colnames = colnames
                if formated:
                    formating(self, asfloat)

    def setcolname(self, n, x):
        if isls(x):
            self.colnames = x
        else:
            if self.colnames == []:
                self.colnames = [None for i in range(self.column_number)]
            self.colnames[n] = x

    def setrowname(self, n, x):
        if isls(x):
            self.rownames = x
        else:
            if self.rownames == []:
                self.rownames = [None for i in range(self.row_number)]
            self.rownames[n] = x

    def rowname(self, ind=None):
        if ind is None:
            return self.rownames
        return self.rownames[ind]

    def colname(self, ind=None):
        if ind is None:
            return self.colnames
        return self.colnames[ind]

    def getrowname(self, t):
        return self[self.rownames.index(t)]

    def getcolname(self, t):
        return self.getcol(self.colnames.index(t))

    def __mod__(self, name):
        if isls(name):
            result = matrix([self % x for x in name])
            return result
        else:
            if name not in self.rownames:
                return 'no rows have this name'
            else:
                return self[self.rownames.index(name)]

    def __matmul__(self, name):
        if isls(name):
            result = matrix([self @ x for x in name])
            return result
        else:
            if name not in self.colnames:
                return 'no columns have this name'
            else:
                return self.getcol(self.colnames.index(name))

    def clrowname(self):
        self.rownames = None

    def clcolname(self):
        self.colnames = None

    def clallname(self):
        self.rownames = None
        self.colnames = None

    def find(self, element, mode=0, obj=0, each=0):
        if isinstance(element, list):
            if callable(element[0]):
                if mode == 0:
                    for i in range(self.rown()):
                        for j in range(self.coln()):
                            if all(t(self[i, j]) for t in element):
                                return [i, j] if obj == 0 else self[i, j]
                    return -1
                elif mode == 1:
                    if each == 0:
                        for i in range(self.rown()):
                            if all(t(self[i]) for t in element):
                                return i if obj == 0 else self[i]
                    else:
                        for i in range(self.rown()):
                            if all(
                                    all(t(j) for j in self[i])
                                    for t in element):
                                return i if obj == 0 else self[i]
                    return -1
                else:
                    if each == 0:
                        for i in range(self.coln()):
                            if all(t(self.getcol(i)) for t in element):
                                return i if obj == 0 else self.getcol(i)
                    else:
                        for i in range(self.coln()):
                            if all(
                                    all(t(j) for j in self.getcol(i))
                                    for t in element):
                                return i if obj == 0 else self.getcol(i)
                    return -1
            else:
                if mode == 0:
                    row = self.row
                    result = row.index(element) if element in row else -1
                    if result == -1:
                        return result
                    else:
                        return result if obj == 0 else self[result]
                else:
                    row = self.transpose().row
                    result = row.index(element) if element in row else -1
                    if result == -1:
                        return result
                    else:
                        return result if obj == 0 else self.getcol(result)
        else:
            if callable(element):
                if mode == 0:
                    for i in range(self.rown()):
                        for j in range(self.coln()):
                            if element(self[i, j]):
                                return [i, j] if obj == 0 else self[i, j]
                    return -1
                elif mode == 1:
                    if each == 0:
                        for i in range(self.rown()):
                            if element(self[i]):
                                return i if obj == 0 else self[i]
                    else:
                        for i in range(self.rown()):
                            if all(element(j) for j in self[i]):
                                return i if obj == 0 else self[i]
                    return -1
                else:
                    if each == 0:
                        for i in range(self.coln()):
                            if element(self.getcol(i)):
                                return i if obj == 0 else self.getcol(i)
                    else:
                        for i in range(self.coln()):
                            if all(element(j) for j in self.getcol(i)):
                                return i if obj == 0 else self.getcol(i)
                    return -1
            else:
                whole = self.split()
                if element not in whole:
                    return -1
                ind = whole.index(element)
                result = list(divmod(ind, self.coln()))
                return result if obj == 0 else self[result]

    def findall(self, element, mode=0, obj=0, each=0, asmatrix=0):
        isfunc = False
        if isinstance(element, list):
            if callable(element[0]):
                isfunc = True
                if mode == 0:
                    result = [[i, j] for i in range(self.row_number)
                              for j in range(self.column_number) if all(
                                  t(self[i, j]) for t in element)]
                elif mode == 1:
                    result = [
                        i for i in range(self.row_number) if (all(
                            t(self[i]) for t in element) if each == 0 else all(
                                all(t(j) for j in self[i]) for t in element))
                    ]
                else:
                    trans = self.transpose()
                    result = [
                        i for i in range(trans.row_number) if (all(
                            t(trans[i])
                            for t in element) if each == 0 else all(
                                all(t(j) for j in trans[i]) for t in element))
                    ]
            else:
                if mode == 0:
                    result = [
                        i for i in range(self.row_number) if self[i] == element
                    ]
                else:
                    trans = self.transpose()
                    result = [
                        i for i in range(trans.row_number)
                        if trans[i] == element
                    ]
        else:
            if callable(element):
                isfunc = True
                if mode == 0:
                    result = [[i, j] for i in range(self.row_number)
                              for j in range(self.column_number)
                              if element(self[i, j])]
                elif mode == 1:
                    result = [
                        i for i in range(self.row_number)
                        if (element(self[i]) if each == 0 else all(
                            element(j) for j in self[i]))
                    ]
                else:
                    trans = self.transpose()
                    result = [
                        i for i in range(trans.row_number)
                        if (element(trans[i]) if each == 0 else all(
                            element(j) for j in trans[i]))
                    ]
            else:
                result = [[i, j] for i in range(self.row_number)
                          for j in range(self.column_number)
                          if self[i, j] == element]
        if obj == 0:
            return result
        else:
            try:
                if type(result[0]) == list:
                    return [self[i] for i in result]
                else:
                    temp = [self.getrow(i) for i in result] if mode == 1 else [
                        self.getcol(i) for i in result
                    ]
                    if asmatrix == 0:
                        return temp
                    else:
                        return matrix(temp) if mode == 1 else matrix(
                            temp).transpose()
            except:
                return result

    def count(self, element, mode=0):
        # count the number of occurrences of an element in the matrix
        return len(self.findall(element, mode))

    def flipped(self, mode=0):
        if mode != 0:
            self.row.reverse()
        else:
            self.row = matrix(list(reversed(
                self.transpose().row))).transpose().row

    def flip(self, mode=0):
        temp = self.copy()
        temp.flipped(mode)
        return temp

    def reverse(self):
        temp = self.copy()
        temp.row = form(list(reversed(temp.split())), temp.row_number,
                        temp.column_number).row
        return temp

    def reversing(self):
        self.row = form(list(reversed(self.split())), self.row_number,
                        self.column_number).row

    def element(self):
        return [j for i in self for j in i]

    def show(self, interval=2, sign='', get=False):
        rep = ''
        if self.rownames != [] or self.colnames != []:
            nrow, ncol = self.dim()
            for i in range(nrow):
                try:
                    rep += f'{self.rownames[i]} {self.row[i]}\n'
                except:
                    rep += f'[{i},] {self.row[i]}\n'
            cols = '  '
            for j in range(ncol):
                try:
                    cols += f'{self.colnames[j]}  '
                except:
                    cols += f'[{j},] '
            cols += '\n'
            rep = cols + rep
        else:
            for i in self.row:
                q = str([str(j) for j in i]).replace("\'", '')
                rep += q[1:-1].replace(', ', f'{sign}{" "*interval}') + '\n'
        rep = rep[:-1]
        if not get:
            print(rep)
        else:
            return rep

    def ind_to_dim(self, ind):
        rownum, colnum = divmod(ind, self.coln())
        return rownum, colnum

    def getfromind(self, ind):
        return self[self.ind_to_dim(ind)]

    def pickbyind(self, ind):
        return self.split()[ind]

    def dim_to_ind(self, x, y, natural=False):
        if natural:
            x -= 1
            y -= 1
        return self.coln() * x + y

    def getfromls(self, x, y, natural=False):
        return self.split()[self.dim_to_ind(x, y, natural)]

    def __str__(self):
        ''' When call print(a) as a is a matrix object, gives the representation
            of matrix a, here the representation will ensure all of the float
            is represented as fractions which is easier to calculate for users. '''
        rep = ''
        if self.rownames != [] or self.colnames != []:
            nrow, ncol = self.dim()
            if self.rownames == []:
                for i in range(nrow):
                    rep += f'[{i},] {self.row[i]}\n'
            else:
                for i in range(nrow):
                    currow = self.rownames[i]
                    if currow is None:
                        rep += f'[{i},] {self.row[i]}\n'
                    else:
                        rep += f'{currow} {self.row[i]}\n'
            cols = '  '
            if self.colnames == []:
                for j in range(ncol):
                    cols += f'[{j},] '
            else:
                for j in range(ncol):
                    curcol = self.colnames[j]
                    if curcol is None:
                        cols += f'[{j},] '
                    else:
                        cols += f'{curcol}  '
            cols += '\n'
            rep = cols + rep
        else:
            for i in self.row:
                q = str([str(j) for j in i]).replace("\'", '')
                rep += f'{q}\n'
        rep = rep[:-1]
        return rep

    __repr__ = __str__

    def __len__(self):
        return self.row_number * self.column_number

    def __add__(self, another):
        ''' Return the result matrix of addition between matrices if the matrices could be added,
            otherwise return a message showing the reason why they cannot be added. '''
        if type(another) != matrix:
            return form([x + another for x in self.split()], *self.dim())
        if self.row_number != another.row_number or self.column_number != another.column_number:
            return 'matrix addition should take two matrices that have the same size'
        else:
            result = list()
            for i in range(len(self.row)):
                result.append([
                    self.row[i][x] + another.row[i][x]
                    for x in range(len(self.row[i]))
                ])
            return matrix(result)

    def __sub__(self, another):
        ''' Return the result matrix of subtraction between matrices if the matrices could be subtracted,
            otherwise return a message showing the reason why they cannot be subtracted. '''
        if type(another) != matrix:
            return form([x - another for x in self.split()], *self.dim())
        if self.row_number != another.row_number or self.column_number != another.column_number:
            return 'matrix subtraction should take two matrices that have the same size'
        else:
            result = list()
            for i in range(len(self.row)):
                result.append([
                    self.row[i][x] - another.row[i][x]
                    for x in range(len(self.row[i]))
                ])
            return matrix(result)

    def __radd__(self, another):
        return self + another

    def __rsub__(self, another):
        return -(self - another)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __mul__(self, another):
        ''' Return the result matrix of regular multiplication between matrices
            if the matrices could be multiplied, otherwise return a message
            showing the reason why they cannot be multiplied. '''
        if type(another) in [int, float, fractions.Fraction, complex]:
            return matrix([[i * another for i in j] for j in self.row])
        else:
            if self.column_number != another.row_number:
                return 'matrix multiplication should take two matrices that has one\'s column number equals to another one\'s row number'
            else:
                row1 = self.row
                row2 = another.row
                result = [[0 for t in range(another.column_number)]
                          for i in range(self.row_number)]
                for i in range(self.row_number):
                    for j in range(another.column_number):
                        result[i][j] = sums([
                            row1[i][x] * row2[x][j]
                            for x in range(len(row1[i]))
                        ])
                return matrix(result)

    def filldiag(self, element, mode=0):
        N = min(self.row_number, self.column_number)
        if isls(element):
            N = min(N, len(element))
            if mode == 0:
                for i in range(N):
                    self[i, i] = element[i]
            else:
                for i in range(N):
                    self[i, N - 1 - i] = element[i]
        else:
            if mode == 0:
                for i in range(N):
                    self[i, i] = element
            else:
                for i in range(N):
                    self[i, N - 1 - i] = element

    def det(self, formated=False, mode='digit', twod=True):
        ''' Calculate the determinant of the matrix, return a number
            which is the determinant of the matrix.
        '''
        if not self.is_square():
            return 'non-square matrix does not have determinant'
        else:
            if mode == 'poly':
                if any(
                        isinstance(x,
                                   __import__('polynomial').polynomial)
                        for y in self.row for x in y):
                    return det2(self)
                else:
                    return self.det()
            if not twod:
                return det2(self)
            if self.is_upper_triangular() or self.is_lower_triangular():
                result = 1
                for i in range(self.row_number):
                    result *= self.row[i][i]
                if mode == 'poly':
                    if isinstance(result, __import__('polynomial').polynomial):
                        return result
                if formated:
                    result = fractions.Fraction(
                        result).limit_denominator().__float__()
                    return int(result) if result.is_integer() else result
                return result

            else:
                return self.detnew(formated)
                # transform the matrix into a triangular matrix and then
                # we can calculate the determinant by simply product the values
                # of the diagonals, here I choose upper triangular matrix
                # (This method is related to Gauss Elimination)

                #temp = self.copy()
                #row = temp.row
                #row_number = temp.row_number
                #swap_times = 0
                #for i in range(1, row_number):
                #if row[i-1][i-1] == 0:
                #return 0
                #if any(x != 0 for x in row[i][:i]):
                #for t in range(i):
                #if all(x == 0 for x in row[i][:i]):
                #break
                ##print(temp)
                ##os.system('cls')
                #if row[i][t] != 0:
                #found = False
                #found_row = None
                #rprow = None
                #multi = None
                #for h in range(row_number):
                #if h != i:
                #if all(x == 0 for x in row[h][:t]):
                #rprow = h
                #if row[h][t] != 0:
                #found = True
                #found_row = h
                #multi = (row[i][t] / row[h][t]) * (-1)
                #break
                #if found:
                #row[i] = [row[found_row][x]*multi + row[i][x] for x in range(row_number)]
                #row[i][t] = 0
                #else:
                #print('apple')
                #if rprow != None:
                #swaprow(temp, i+1, rprow+1)
                #swap_times += 1
                #else:
                #if row[i][i] == 0:
                #return 0
                #result = 1
                #for k in range(temp.row_number):
                #result *= temp.row[k][k]
                #result *= (-1)**swap_times
                #if showtime:
                #print(swap_times)
            ''' Here is a method calculating determinant by calculating the
            sum of the product of each entry with the determinant of
            the minor matrix generated by the entry (times 1 or -1 alternatively).
            Considering with very large matrices this method is insanely slow,
            here I choose a relatively fast algorithm which is Gauss Elimination,
            but actually there are many other good algorithms much more faster
            than Gauss Elimination exist. '''
            '''
            result = 0
            if self.row_number == 1:
                result = self.row[0][0]
            elif self.row_number == 2:
                row = self.row
                result = row[0][0]*row[1][1] - row[0][1]*row[1][0]
            else:
                row = self.row
                for j in range(len(row[0])):
                    if j%2 == 0:
                        result += row[0][j]*matrix([[y[x] for x in range(len(y)) if x != j] for y in row[1:]]).det()
                    else:
                        result -= row[0][j]*matrix([[y[x] for x in range(len(y)) if x != j] for y in row[1:]]).det()
            '''

    def old_det(self, mode='digit'):
        if not self.is_square():
            return 'non-square matrix does not have determinant'
        else:
            if mode == 'poly':
                if any(
                        isinstance(x,
                                   __import__('polynomial').polynomial)
                        for y in self.row for x in y):
                    return det2(self)
                else:
                    return self.det()
            if self.is_upper_triangular() or self.is_lower_triangular():
                result = 1
                for i in range(self.row_number):
                    result *= self.row[i][i]
                if mode == 'poly':
                    if isinstance(result, __import__('polynomial').polynomial):
                        return result
                return result
            else:
                temp = self.copy()
                row = temp.row
                row_number = temp.row_number
                swap_times = 0
                for i in range(1, row_number):
                    if row[i - 1][i - 1] == 0:
                        return 0
                    if any(x != 0 for x in row[i][:i]):
                        for t in range(i):
                            if all(x == 0 for x in row[i][:i]):
                                break
                            #print(temp)
                            #os.system('cls')
                            if row[i][t] != 0:
                                found = False
                                found_row = None
                                rprow = None
                                multi = None
                                for h in range(row_number):
                                    if h != i:
                                        if all(x == 0 for x in row[h][:t]):
                                            rprow = h
                                            if row[h][t] != 0:
                                                found = True
                                                found_row = h
                                                multi = (row[i][t] /
                                                         row[h][t]) * (-1)
                                                break
                                if found:
                                    row[i] = [
                                        row[found_row][x] * multi + row[i][x]
                                        for x in range(row_number)
                                    ]
                                    row[i][t] = 0
                                else:
                                    if rprow != None:
                                        swaprow(temp, i, rprow)
                                        swap_times += 1
                    else:
                        if row[i][i] == 0:
                            return 0
                result = 1
                for k in range(temp.row_number):
                    result *= temp.row[k][k]
                result *= (-1)**swap_times
                return result

    def detnew(self, formated=False):
        temp = self.copy()
        swaptime = 0
        nrow = temp.row_number
        for i in range(nrow):
            first_col = temp.getcol(i)[i:]
            if any(x != 0 for x in first_col):
                N = nrow - i
                for k in range(N):
                    if first_col[k] != 0:
                        if k != 0:
                            swaprow(temp, i, k + i)
                            swaptime += 1
                        break

                first_row = temp[i]
                for j in range(i + 1, nrow):
                    current = temp[j, i]
                    if current != 0:
                        multi = -(current / temp[i, i])
                        temp[j] = [
                            temp[j, x] + multi * first_row[x]
                            for x in range(nrow)
                        ]
            else:
                return 0
        if not formated:
            diag = [temp[t, t] for t in range(nrow)]
            result = mult(diag) * (-1)**swaptime
            return result
        else:
            diag = [
                fractions.Fraction(temp[t, t]).limit_denominator().__float__()
                for t in range(nrow)
            ]
            result = mult(diag) * (-1)**swaptime
            if result.is_integer():
                result = int(result)
            return result

    def ispinv(self, other):
        return self * other * self == self

    def isreinv(self, other):
        return self.ispinv(other) and other * self * other == other

    def ismpinv(self, other):
        return self.isreinv(other) and (self * other).transpose(
        ) == self * other and (other * self).transpose() == other * self

    def isinv(self, other):
        return self * other == other * self == identity(self.row_number)

    def inv(self, formated=False, asfloat=False, singular=True):
        return self.inverse(formated, asfloat, singular)

    def mean(self):
        whole = self.split()
        return sum(whole) / len(whole)

    def var(self):
        whole = self.split()
        average = sum(whole) / len(whole)
        return (1 / (len(whole) - 1)) * sum([(x - average)**2 for x in whole])

    def se(self):
        return self.var()**0.5

    def transpose(self):
        ''' Return the transpose of the matrix as a new matrix object. '''
        row, nrow, ncol = self.row, self.row_number, self.column_number
        return matrix([[row[i][j] for i in range(nrow)] for j in range(ncol)],
                      self.colnames, self.rownames)

    def ctranspose(self):
        result = self.transpose()
        m, n = result.dim()
        for i in range(m):
            for j in range(n):
                current = result[i][j]
                if isinstance(current, complex):
                    result[i][j] = current.conjugate()
        return result

    def is_tri(self):
        return self.is_uptri() or self.is_lowtri()

    def is_unitary(self):
        return self.is_square() and self * self.cT() == identity(
            self.row_number)

    def is_hermitian(self):
        return self.is_square() and self == self.cT()

    def is_orthogonal(self):
        return self.is_square() and self * self.transpose() == identity(
            self.row_number)

    def is_symmetric(self):
        return self == self.transpose()

    def is_permutation(self):
        return self.is_square() and all(self[x].count(1) == 1 and\
        self[x].count(1) + self[x].count(0) == self.column_number\
        for x in range(self.row_number)) and all(self.getcol(y).count(1) == 1\
        and self.getcol(y).count(1) + self.getcol(y).count(0) == self.row_number\
        for y in range(self.column_number))

    def is_normal(self):
        return (self.cT() * self) == (self * (self.cT()))

    def cT(self):
        return self.ctranspose()

    def formated(self, asfloat=False, tol=1e-3):
        temp = self.copy()
        formating(temp, asfloat, tol)
        return temp

    def commute(self, other):
        return self * other == other * self

    def commutator(self, other):
        return self * other - other * self

    def adj(self):
        row = self.row
        result = [[0 for t in range(self.row_number)]
                  for i in range(self.row_number)]
        for i in range(self.row_number):
            for j in range(self.row_number):
                result[i][j] = ((-1)**((i + j))) * matrix(
                    [[y[t] for t in range(self.row_number) if t != j]
                     for y in row[:i] + row[i + 1:]]).det()
        minor = matrix(result)
        adjoint = minor.transpose()
        return adjoint

    def inverse(self, formated=False, asfloat=False, singular=True):
        ''' Return the inverse of the matrix as a new matrix object if such inverse exists,
            otherwise return a message showing why such inverse does not exist. '''
        if self.row_number != self.column_number:
            return 'non-square matrix does not have inverse matrix, you can use pinv() to calculate its pseudo inverse'
        else:
            det1 = self.det()
            if singular and (abs(det1) < 1e-5):
                return 'this matrix has determinant of 0 which means it does not have inverse matrix, you can use pinv() to calculate its pseudo inverse'
            elif self.row_number == 1:
                if formated:
                    return matrix([[
                        fractions.Fraction(1 / self.det()).limit_denominator()
                    ]])
                return matrix([[1 / self.det()]])
            else:
                result = self.adj() / self.det()
                if formated:
                    formating(result, asfloat)
                return result

    def __truediv__(self, another):
        ''' Return the result matrix of the division between matrices as a new matrix object
            if the division is valid, otherwise return a message showing why the division
            is invalid under the given cases. '''
        if type(another) in [int, float, fractions.Fraction, complex]:
            return matrix([[i / another for i in j] for j in self.row])
        else:
            if another.row_number != another.column_number:
                return 'the divisor matrix is non-square matrix which cannot be divided'
            elif another.det() == 0:
                return 'the divisor matrix has determinant of 0 which means it cannot be divided'
            else:
                return self * another.inv_lu()

    def __rtruediv__(self, another):
        return another * self.inv_lu()

    def cut(self, ind1, ind2=None, mode=0, notasmatrix=0):
        if all(isls(x) for x in [ind1, ind2]):
            ind1 = list(ind1)
            ind2 = list(ind2)
            if len(ind1) == 1:
                ind1 = [ind1[0], ind1[0] + 1]
            if len(ind2) == 1:
                ind2 = [ind2[0], ind2[0] + 1]
            if ind1[0] == START:
                ind1[0] = 0
            if ind1[1] == END:
                ind1[1] = self.rown()
            if ind2[0] == START:
                ind2[0] = 0
            if ind2[1] == END:
                ind2[1] = self.coln()
            return self.cut(*ind1).cut(*ind2, mode=1, notasmatrix=notasmatrix)
        if ind2 is None:
            if type(ind1) == int:
                ind2 = ind1 + 1
            else:
                ind1, ind2 = ind1[0], ind1[1]
        if mode == 0:
            if ind1 == START:
                ind1 = 0
            if ind2 == END:
                ind2 = self.row_number
            result = deepcopy(self.row[ind1:ind2])
            if notasmatrix:
                return result
            else:
                return matrix(result)
        else:
            if ind1 == START:
                ind1 = 0
            if ind2 == END:
                ind2 = self.column_number
            result = [i[ind1:ind2] for i in self.row]
            if notasmatrix:
                return result
            else:
                return matrix(result)

    def __neg__(self):
        return self * (-1)

    def __xor__(self, number):
        return self.__pow__(number)

    def dim(self):
        return self.row_number, self.column_number

    def __pow__(self, number):
        ''' Return the power of the matrix with degree of the given number as
            a new matrix object if the matrix can be multiplied by itself, otherwise
            return the message showing why the power of given matrix is not defined. '''
        c = self
        if type(number) == matrix:
            return self.dot(number)
        if not isinstance(number, int):
            number = float(number)
            result = self.root(1 / abs(number))
            return result if number > 0 else result.inv_lu()
        elif number == 0:
            if self.row_number != self.column_number:
                return 'non-square matrix has no definition of 0 power as well as any other power, because you cannot multiply a non-square matrix by itself'
            return matrix(
                [[0 if x != i else 1 for x in range(self.row_number)]
                 for i in range(self.row_number)])
        else:
            for i in range(abs(number) - 1):
                c *= self
            return c if number > 0 else c.inv_lu()

    def __rpow__(self, other):
        return form([other**i for i in self.split()], *self.dim())

    def __rxor__(self, other):
        return other**self

    def __contains__(self, value):
        return value in self.split()

    def hasrow(self, r):
        return r in self.row

    def hascol(self, c):
        return c in self.transpose().row

    def __eq__(self, another):
        ''' Check if matrices are equal, return True if matrices are equal,
            otherwise return False. If two matrix are equal, then every element
            of both matrices are equal. '''
        if not isinstance(another, matrix):
            return False
        if self.row_number != another.row_number or self.column_number != another.column_number:
            return False
        elif any(self.row[i] != another.row[i]
                 for i in range(self.row_number)):
            return False
        else:
            return True

    def rowat(self, m):
        if m not in range(self.row_number):
            return 'the required row number is out of range'
        return self[m]

    def colat(self, n, show_as_column=False):
        if n not in range(self.column_number):
            return 'the required column number is out of range'
        result = [i[n] for i in self.row]
        if show_as_column:
            return matrix(result, dr=1)
        else:
            return result

    def select(self, alist, mode=0, asmatrix=1, datatype=list):
        # get rows or columns from a list of number
        # or a list of functions with conditions
        # could be used in __getitem__
        if callable(alist):
            alist = [alist]
            if mode == 0:
                return self.select([
                    x for x in range(self.row_number) if all(
                        cond(x) for cond in alist)
                ])
            else:
                return self.select([
                    x for x in range(self.column_number) if all(
                        cond(x) for cond in alist)
                ], 1)
        else:
            if not isls(alist):
                alist = [alist]
            if callable(alist[0]):
                if mode == 0:
                    return self.select([
                        x for x in range(self.row_number) if all(
                            cond(x) for cond in alist)
                    ])
                else:
                    return self.select([
                        x for x in range(self.column_number) if all(
                            cond(x) for cond in alist)
                    ], 1)
            if mode == 0:
                if asmatrix == 0:
                    return [datatype.__call__(self[x]) for x in alist]
                else:
                    return matrix([self[x] for x in alist])
            else:
                if asmatrix == 0:
                    return [datatype.__call__(self.getcol(x)) for x in alist]
                else:
                    return matrix([self.getcol(x) for x in alist]).transpose()

    def choose(self, t, mode=0):
        if mode == 0:
            return matrix(self[t])
        else:
            return matrix(self.getcol(t), dr=1)

    def gete(self, m, n):
        if m not in range(self.row_number) or n not in range(
                self.column_number):
            return 'required row or column number is out of range'
        return self[m][n]

    def __and__(self, cond):
        if callable(cond):
            whole = self.split()
            return [x for x in whole if cond(x)]
        if isls(cond):
            if all(callable(x) for x in cond):
                return self.findall(cond)
            N = len(cond)
            if N == 1:
                return self & cond[0]
            elif N == 2:
                if type(cond[0]) == int:
                    condition = [cond[1]] if callable(cond[1]) else cond[1]
                    current = self[cond[0]]
                    return [
                        t for t in range(self.coln()) if all(
                            j(current[t]) for j in condition)
                    ]
                else:
                    if callable(cond[0]):
                        condition = [cond[0]]
                    else:
                        condition = cond[0]
                    chooselist = self[cond[1]]
                    return [
                        x for x in chooselist if all(t(x) for t in condition)
                    ]
            elif N > 2:
                if type(cond[0]) == int and type(cond[2]) == int:
                    condition = [cond[1]] if callable(cond[1]) else cond[1]
                    if cond[2] == 0:
                        current = self[cond[0]]
                        result = [
                            t for t in range(self.coln()) if all(
                                j(current[t]) for j in condition)
                        ]
                    else:
                        current = self.getcol(cond[0])
                        result = [
                            t for t in range(self.rown()) if all(
                                j(current[t]) for j in condition)
                        ]

                else:
                    if callable(cond[0]):
                        condition = [cond[0]]
                    else:
                        condition = cond[0]
                    chooselist = self[
                        cond[1]] if cond[2] == 0 else self.getcol(cond[1])
                    result = [
                        x for x in chooselist if all(t(x) for t in condition)
                    ]
                if N > 3:
                    if cond[3] == 0:
                        return matrix(result)
                    else:
                        return matrix(result, dr=1)
                else:
                    return result

    def __getitem__(self, ind):
        row = self.row
        if isls(ind):
            indlen = len(ind)
            if indlen == 2:
                if isls(ind[0]) or callable(ind[0]):
                    return self.select(ind[0], ind[1])
                return row[ind[0]][ind[1]]
            elif indlen == 1:
                if type(ind) == tuple:
                    if isls(ind[0]) or callable(ind[0]):
                        return self.select(ind[0])
                    return [i[ind[0]] for i in row]
                return row[ind]
            elif indlen > 2:
                if ind[-1] == 'col':
                    return self.select(ind[:-1], 1)
                else:
                    return self.select(ind)
        if callable(ind):
            return self.select(ind)
        if isinstance(ind, cell):
            return row[ind[0]][ind[1]]
        return row[ind]

    def __setitem__(self, ind, item):
        row = self.row
        if isls(ind):
            if len(ind) == 2:
                row[ind[0]][ind[1]] = item
            else:
                self.rpcol(ind[0], item)
        else:
            if isinstance(ind, cell):
                row[ind[0]][ind[1]] = item
            else:
                row[ind] = item

    def __delitem__(self, ind):
        row = self.row
        if not isinstance(ind, tuple):
            if self.row_number == 1:
                print(
                    'no more rows can be deleted since after deleting this matrix will be empty'
                )
            else:
                del row[ind]
                self.row_number -= 1
        else:
            if self.column_number == 1:
                print(
                    'no more columns can be deleted since after deleting this matrix will be empty'
                )
            else:
                for i in row:
                    del i[ind[0]]
                self.row_number -= 1

    def append(self, m, mode=0):
        if mode == 0:
            if len(m) != self.column_number:
                return 'the length of the new row is not equal to f this matrix\'s column number'
            self.row.append(m)
            self.row_number += 1
        else:
            if len(m) != self.row_number:
                return 'the length of the new column is not equal to f this matrix\'s row number'
            for i in range(self.row_number):
                self[i].append(m[i])
            self.column_number += 1

    def insert(self, ind, m, mode=0):
        if mode == 0:
            if len(m) != self.column_number:
                return 'the length of the new row is not equal to f this matrix\'s column number'
            self.row.insert(ind, m)
            self.row_number += 1
        elif mode == 1:
            if len(m) != self.row_number:
                return 'the length of the new column is not equal to f this matrix\'s row number'
            for i in range(self.row_number):
                self[i].insert(ind, m[i])
            self.column_number += 1

    def __floordiv__(self, other):
        if type(other) != matrix:
            return self / other
        return self.dot(other.recipro())

    def __rfloordiv__(self, other):
        return other * self.recipro()

    def recipro(self):
        return form([(1 / x) for x in self.split()], *self.dim())

    def range_fill(self, rows, cols, elements):
        e = deepcopy(elements)
        if type(e) == matrix:
            e = e.element()
        row1, row2 = rows
        col1, col2 = cols
        for i in range(row1, row2):
            for j in range(col1, col2):
                self[i, j] = e.pop(0)

    def fillin(self, element):
        nrow, ncol = self.dim()
        if isls(element):
            N = min(len(self), len(element))
            count = 0
            for i in range(nrow):
                for j in range(ncol):
                    self[i, j] = element[count]
                    count += 1
                    if count >= N:
                        break
                if count >= N:
                    break
        else:
            for i in range(nrow):
                for j in range(ncol):
                    self[i, j] = element

    def put(self, ind, element, mode=0, new=False):
        if new:
            temp = self.copy()
            temp.put(ind, element, mode)
            return temp
        poly = not isls(element)
        if mode == 0:
            if poly:
                self[ind] = [element for i in range(self.coln())]
            else:
                self[ind] = element + self[ind][len(element):]
        else:
            if poly:
                self.setcol(ind, [element for i in range(self.rown())])
            else:
                self.setcol(ind, element + self.getcol(ind)[len(element):])

    def fill(self, start=None, stop=None, step=1):
        if stop is None:
            start, stop = 1, start
        fillin = range(start, stop + 1, step)
        k = 0
        nrow = self.row_number
        ncol = self.column_number
        size = len(fillin)
        for i in range(nrow):
            if k < size:
                for j in range(ncol):
                    if k < size:
                        self[i, j] = fillin[k]
                        k += 1
                    else:
                        break
            else:
                break

    def clear(self, default=0):
        nrow = self.row_number
        ncol = self.column_number
        row = self.row
        for i in range(nrow):
            for j in range(ncol):
                row[i][j] = default

    def sums(self, row=None, col=None):
        if row == col == None:
            return sum(sums(self.row))
        else:
            if col is None:
                if row == 'all':
                    return sum([self.sums(x) for x in range(self.row_number)])
                if not isinstance(row, int):
                    return sum([self.sums(x) for x in row])
                return sum(self[row])
            else:
                if col == 'all':
                    return sum(
                        [self.sums(col=x) for x in range(self.column_number)])
                if not isinstance(col, int):
                    return sum([self.sums(col=x) for x in col])
                return sum(self.getcol(col))

    def sumrange(self, ind1, ind2):
        x = cell(ind1[0], ind1[1], [0, self.row_number - 1],
                 [0, self.column_number - 1])
        result = 0
        while True:
            result += self[x]
            x = x.move()
            if x == ind2:
                result += self[x]
                break
        return result

    def dot(self, other):
        ''' Return the result matrix of matrices with multiplying each other's
            elements one by one directly. '''
        if self.row_number == other.column_number == 1:
            return sum([
                self.row[0][i] * other.row[i][0]
                for i in range(self.column_number)
            ])
        elif other.row_number == self.column_number == 1:
            return sum([
                other.row[0][i] * self.row[i][0]
                for i in range(self.row_number)
            ])
        else:
            if self.row_number != other.row_number or self.column_number != other.column_number:
                return 'dot product between two n*n matrices (n>1) should have the same size'
            else:
                result = []
                for i in range(self.row_number):
                    result.append([
                        self.row[i][x] * other.row[i][x]
                        for x in range(len(self.row[i]))
                    ])
                return matrix(result)

    def outer(self, other):
        pass

    def cross(self, other, tomatrix=True):
        ''' Return the cross product of given matrices. Only defined between
            1*2 and 1*3 matrices. When taking matrices to do cross product
            in this function, you should make sure both of them
            are 1*2 matrix or 1*3 matrix, for example, if you want to calculate
            the cross product of matrix([[1,2]]) and matrix([[5],[6]])
            (one is 1*2 matrix and another one is 2*1 matrix), you should
            use this function as cross(matrix([[1,2]]),matrix([[5,6]])).
            The parameter tomatrix is set to True as default, when you use
            this function without declaring anything about tomatrix,
            this function will return a matrix object as a result of cross
            product, if you set tomatrix to False by cross(a, b, tomarix = False),
            or more simplified: cross(a, b, False), this function will return
            the exact value (for 1*2 case) or the polynomial (for 1*3 case)
            of the result of cross product. '''
        if self.row_number == other.row_number == 1 and self.column_number == other.column_number == 2:
            if tomatrix:
                return matrix([self.row[0], other.row[0]])
            else:
                return matrix([self.row[0], other.row[0]]).det()
        elif self.row_number == other.row_number == 1 and self.column_number == other.column_number == 3:
            if tomatrix:
                return matrix([['i', 'j', 'k'], self.row[0], other.row[0]])
            else:
                result = ''
                rows = [['i', 'j', 'k'], self.row[0], other.row[0]]
                for j in range(len(rows[0])):
                    if j % 2 == 0:
                        result += '+' + '(' + str(
                            matrix([[y[x] for x in range(len(y)) if x != j]
                                    for y in rows[1:]
                                    ]).det()) + ')' + rows[0][j]
                    else:
                        result += '+' + '(' + str((-1) * matrix(
                            [[y[x] for x in range(len(y)) if x != j]
                             for y in rows[1:]]).det()) + ')' + rows[0][j]
                return result[1:]
        else:
            return 'cross product only defined between two 1*2 matrices or 1*3 matrices here'

    def __call__(self, cond, mode=0, obj=0, each=0, asmatrix=0):
        if callable(cond):
            cond = [cond]
        if mode == 0:
            if each == 0:
                result = [
                    x for x in range(self.rown()) if all(
                        t(self[x]) for t in cond)
                ]
            else:
                result = [
                    x for x in range(self.rown()) if all(
                        all(t(y) for y in self[x]) for t in cond)
                ]
            result = result if obj == 0 else [self[i] for i in result]
            return result if asmatrix == 0 else (matrix(
                result) if asmatrix == 1 else matrix(result).transpose())
        elif mode == 1:
            if each == 0:
                result = [
                    x for x in range(self.coln()) if all(
                        t(self.getcol(x)) for t in cond)
                ]
            else:
                result = [
                    x for x in range(self.coln()) if all(
                        all(t(y) for y in self.getcol(x)) for t in cond)
                ]
            result = result if obj == 0 else [self.getcol(i) for i in result]
            return result if asmatrix == 0 else (matrix(
                result).transpose() if asmatrix == 1 else matrix(result))
        else:
            result = self.findall(cond)
            return result if obj == 0 else [self[i] for i in result]

    def singular(self, formated=False):
        if not self.is_square():
            return 'this matrix is not a square matrix, \
which does not satisfy the very first condition of whether \
a matrix is singular or non-singular.'

        return self.det(formated) == 0

    def nonsingular(self, formated=False):
        if not self.is_square():
            return 'this matrix is not a square matrix, \
which does not satisfy the very first condition of whether \
a matrix is singular or non-singular.'

        return not self.singular()

    def do(self, action, some=None, mode=0):
        if some is None:
            return form([action(x) for x in self.split()], *self.dim())
        else:
            temp = self.copy()
            if not isls(some):
                some = [some]
            if mode == 0:
                for i in some:
                    temp[i] = [action(x) for x in temp[i]]
            else:
                for k in some:
                    col1 = temp.getcol(k)
                    temp.setcol(k, [action(x) for x in col1])
            return temp

    def conv(self, other):
        pass

    def wedge(self, other):
        pass

    def trace(self):
        ''' Return the trace of the matrix if the matrix has trace, otherwise
            return a message showing why the matrix does not have trace. '''
        if self.row_number != self.column_number:
            return 'non-square matrix has no trace'
        else:
            return sum([self.row[i][i] for i in range(self.row_number)])

    def info(self):
        ''' This is a very useful function for the matrices you are dealing with.
            This function returns a detailed information about this matrix,
            including: matrix representation, size, row number, column number,
            determinant, inverse, transpose, trace, row echelon form,
            reduced row echelon form, rank, etc. (if any of above does not
            exist for this matrix, there will be reasons why it does not
            exist following up) '''
        determinant = self.det()
        return f'matrix:\n{self.__str__()} \n\nsize: {self.row_number} x {self.column_number}\n\nrow number: {self.row_number} \n\ncolumn number: {self.column_number} \n\ndeterminant: {determinant} \n\ninverse:\n{self.inv_lu() if type(determinant) != str and determinant != 0 else "this matrix has determinant of 0 which means it does not have inverse matrix"} \n\ntranspose: \n{self.transpose()}\n\ntrace: {self.trace()}\n\nrow echelon form:\n{self.ref()}\n\nreduced row echelon form:\n{self.rref()}\n\nrank: {self.rank()}' + (
            f'\n{self.is_fullrank()}'
            if self.is_fullrank() == 'this matrix is in full rank' else '')

    def size(self):
        return f'{self.row_number}x{self.column_number}'

    def rowspace(self, formated=False):
        x = self.ref(formated)
        nonzero_rows = x(lambda t: anynot(t, 0))
        return self.getcol(nonzero_rows)

    def colspace(self, formated=False):
        x = self.rref(formated)
        pivot_col = []
        nrow, ncol = x.dim()
        for i in range(nrow):
            for j in range(ncol):
                if x[i][j] != 0:
                    pivot_col.append(j)
                    break

        return self[pivot_col, 1]

    def nullity(self, formated=False):
        return self.coln() - self.rank(formated)

    def nullspace(self, formated=False):
        x = self.rref(formated)
        nrow, ncol = x.dim()
        pivot_column = []
        for i in range(nrow):
            for j in range(ncol):
                if x[i][j] != 0:
                    pivot_column.append(j)
                    break
        not_allzeros = [y for y in range(nrow) if set(x[y]) != {0}]
        free_column = [x for x in range(ncol) if x not in pivot_column]
        if len(free_column) == 0:
            return zeros(nrow, 1)
        x = matrix([x.getcol(i) for i in free_column]).transpose()
        if len(not_allzeros) == 0:
            return identity(ncol)
        x = matrix([x[k] for k in not_allzeros])
        x *= -1
        N = ncol - x.row_number
        iderow = identity(N).row
        xrow = x.row
        T = len(xrow)
        free = len(free_column)
        for j in range(free):
            ind = free_column[j]
            xrow.insert(ind, iderow[j])
        return matrix(xrow)

    def null(self, formated=False):
        return self.nullspace(formated)

    def cokernel(self):
        return self.transpose().kernel()

    def leftnullspace(self):
        return self.cokernel()

    def kernel(self, formated=False):
        return self.nullspace(formated)

    def image(self):
        return self.colspace()

    def preimage(self):
        pass

    def coimage(self):
        return self.rowspace()

    def copy(self):
        row = self.row
        return matrix([[x for x in i] for i in row])

    def is_square(self):
        if self.row_number == self.column_number:
            return True
        else:
            return False

    def isdiag(self):
        nrow, ncol = self.dim()
        return all(self[i, j] == 0 for i in range(nrow) for j in range(ncol)
                   if i != j)

    def root(self, number):
        if self.isdiag():
            temp = self.copy()
            diagonals = temp.diag()
            diagonals = [x**(1 / number) for x in diagonals]
            temp.filldiag(diagonals)
            return temp
        else:
            D, V = self.eigen(formated=True)
            return V * D.root(number) * V.inv_lu()

    def is_upper_triangular(self):
        if not self.is_square():
            return 'non-square matrix cannot be any kind of triangular matrix, \
since triangular matrix is a special kind of square matrix.'

        elif all(self.row[i][x] == 0 for i in range(self.row_number)
                 for x in range(i)):
            return True
        else:
            return False

    def is_lower_triangular(self):
        if not self.is_square():
            return 'non-square matrix cannot be any kind of triangular matrix, \
since triangular matrix is a special kind of square matrix.'

        elif all(self.row[i][x] == 0 for i in range(self.row_number)
                 for x in range(i + 1, self.column_number)):
            return True
        else:
            return False

    def is_uptri(self):
        return self.is_upper_triangular()

    def is_lowtri(self):
        return self.is_lower_triangular()

    def is_diagonal(self):
        if self.row_number != self.column_number:
            return 'non-square matrix cannot be diagonal'
        else:
            if self.is_upper_triangular() and self.is_lower_triangular():
                return True
            else:
                return False

    def diagonalizable(self, tol=0.1, error=1e-2, times=None):
        if self.row_number != self.column_number:
            return 'non-square matrix is not diagonalizable'
        else:
            A, Q = self.eigen(tol, times=times)
            return Q.invertible()
            #diagonals = self.eigen(times, formated = True)[0].diag()
            #eigenlist = [diagonals[0]]
            #for k in diagonals[1:]:
            #if all(abs(x-k) > error for x in eigenlist):
            #eigenlist.append(k)
            #return len(eigenlist) == len(diagonals)

    def __lt__(self, another):
        a = self.split()
        b = another.split()
        N = min(len(self), len(another))
        return all(a[x] < b[x] for x in range(N))

    def __gt__(self, another):
        a = self.split()
        b = another.split()
        N = min(len(self), len(another))
        return all(a[x] > b[x] for x in range(N))

    def __le__(self, another):
        return not (self > another)

    def __ge__(self, another):
        return not (self < another)

    def per(self, func, n=None, ind=None, mode=0):
        temp = self.copy()
        if n is not None:
            if mode == 0:
                temp[n] = [func(i) for i in temp[n]]
            else:
                col1 = temp.getcol(n)
                temp.setcol(n, [func(i) for i in col1])
            return temp
        if ind is not None:
            for j in ind:
                temp[j] = func(temp[j])
            return temp
        temp.fillin([func(i) for i in temp.split()])
        return temp

    def inv_lu(self):
        try:
            L, U = self.lu()
            return U.inv_tri(1) * L.inv_tri()
        except:
            P, L, U = self.plu()
            return U.inv_tri(1) * L.inv_tri() * P

    def inv_newton(self, tol=0.1):
        if abs(self.det()) < 1e-5:
            return 'this matrix is singular, so it does not has an inverse, you can compute its pseudo inverse by pinv()'
        nrow = self.rown()
        Astar = self.cT()
        alpha = random.uniform(0, 2 / (self.norm()**2))
        Xk = alpha * Astar
        unit = identity(nrow)
        while True:
            Xk = Xk * (2 * unit - self * Xk)
            testmat = (self * Xk - unit).split()
            if all(abs(i) < tol for i in testmat):
                break

        return Xk

    def norm(self):
        return sum([x**2 for y in self.row for x in y])**0.5

    def householder(self, x, m):
        # assume self is a square matrix
        e1 = build(m, 1, 0)
        e1[0][0] = 1
        alpha = x.norm()
        if x[0][0] > 0:
            alpha = -alpha
        u = x - alpha * e1
        v = u / (u.norm())
        H = identity(m) - 2 * v * v.transpose()
        nrow = self.row_number
        if m < nrow:
            H = H.addto(identity(nrow), (nrow - m, nrow - m))
        return H

    def qr(self, formated=False, asfloat=False):
        A = self.copy()
        m, n = self.dim()
        N = min(m - 1, n)
        Q = identity(m)
        R = zeros(A.row_number, A.column_number)
        #R = self.copy()
        for i in range(N):
            x = matrix([A.getcol(i)[i:]]).transpose()
            if x.anynot(0):
                H = A.householder(x, m - i)
                A1 = H * A
                R = A1
                Q *= H
                A = A1
        if formated:
            formating(Q, asfloat)
            formating(R, asfloat)
        return Q, R

    def rowm(self, m):
        return self.cut(m)

    def colm(self, m):
        return self.cut(m, mode=1)

    def get(self, m, n):
        return self.row[m][n]

    def getrow(self, m):
        return self.row[m]

    def getcol(self, n):
        row = self.row
        return [i[n] for i in row]

    def setrow(self, m, x):
        self.row[m] = x

    def setcol(self, n, x):
        row, nrow = self.row, self.row_number
        for i in range(nrow):
            row[i][n] = x[i]

    def setrowall(self, m, x):
        self.row[m] = [x for x in range(self.column_number)]

    def setcolall(self, n, x):
        row = self.row
        for i in row:
            i[n] = x

    def sete(self, m, n, value=None):
        if isls(m):
            self.row[m[0]][m[1]] = n
        else:
            self.row[m][n] = value

    def setall(self, value):
        nrow, ncol = self.dim()
        row = self.row
        for i in range(nrow):
            for j in range(ncol):
                row[i][j] = value

    def setas(self, alist):
        obj = [i for i in alist]
        nrow, ncol = self.dim()
        row = self.row
        for i in range(nrow):
            for j in range(ncol):
                row[i][j] = alist.pop(0)

    def clrow(self, m, default=0):
        self.row[m] = [default for i in range(self.coln())]

    def clcol(self, n, default=0):
        for j in range(self.rown()):
            self.row[j][n] = default

    def cle(self, m, n, default=0):
        if isls(m):
            self.row[m[0]][m[1]] = default
        else:
            self.row[m][n] = default

    def getrowi(self, alist):
        # get rows from a list of indexes
        row = self.row
        return [row[i] for i in alist]

    def getcoli(self, alist, ver=0):
        # get columns from a list of indexes
        row = self.row
        result = [[i[j] for j in alist] for i in row]
        if ver != 0:
            return result
        N = len(result[0])
        return [[i[j] for i in result] for j in range(N)]

    def getei(self, alist):
        # get elements from a list of indexes
        row = self.row
        return [row[i[0]][i[1]] for i in alist]

    def getrowr(self, m, n, step=1):
        # get rows from a range
        row = self.row
        return [row[i] for i in range(m, n, step)]

    def getcolr(self, m, n, step=1, ver=0):
        # get columns from a range
        row = self.row
        result = [[i[j] for j in range(m, n, step)] for i in row]
        if ver != 0:
            return result
        N = len(result[0])
        return [[i[j] for i in result] for j in range(N)]

    def geter(self, m, n):
        # get elements between 2 ranges
        row = self.row
        r1, r2 = range(m), range(n)
        return [row[i][j] for i in r1 for j in r2]

    def getrowsl(self, m, n, step=1):
        # get rows from a slice
        s1 = slice(m, n, step)
        return self.row[s1]

    def getcolsl(self, m, n, step=1, ver=0):
        # get columns from a slice
        s2 = slice(m, n, step)
        row = self.row
        result = [i[s2] for i in row]
        if ver != 0:
            return result
        N = len(result[0])
        return [[i[j] for i in result] for j in range(N)]

    def getesl(self, m, n):
        # get elements between 2 slices
        s1 = slice(m)
        s2 = slice(n)
        return row[s1][s2]

    def allis(self, element):
        whole = self.split()
        return allis(whole, element)

    def anyis(self, element):
        whole = self.split()
        return anyis(whole, element)

    def allnot(self, element):
        return not self.anyis(element)

    def anynot(self, element):
        return not self.allis(element)

    def exist(self, element, n, mode=0):
        # mode = 0: rows
        # mode = 1 : columns
        # mode = else : whole matrix
        if mode == 0:
            part = self[n]
        elif mode == 1:
            part = self.getcol(n)
        else:
            part = self.split()
        return anyis(part, element)

    def same(self, element, n, mode=0):
        if mode == 0:
            part = self[n]
        elif mode == 1:
            part = self.getcol(n)
        else:
            part = self.split()
        return allis(part, element)

    def nexist(self, element, n, mode=0):
        return not self.exist(element, n, mode)

    def nsame(self, element, n, mode=0):
        return not self.same(element, n, mode)

    def eigen(self,
              tol=0.1,
              formated=False,
              asfloat=False,
              eigvec_times=2,
              times=None,
              merge_eigval=True,
              tols=1e-3,
              method='lu'):
        if not self.is_square():
            return 'non-square matrices do not have eigenvalues and eigenvectors, \
maybe you want to find singular values? Then you can use svd method to get them.'

        # Use QR algorithm (householder)
        # return a tuple (eigenvalues, eigenvectors)

        # We can get the eigenvectors by multiplying new Q each time
        # from a QR decomposition, more precise as the iteration time goes up.
        # Eigenvalues are given by the multiplication of the swap order
        # of Q and R, also more precise as the increase of iteration time.
        # Return a tuple of two matrices, the eigenvalues are given on
        # the first matrix's diagonals, and eigenvectors are the columns
        # of the second matrix.
        temp = self.copy()
        nrow, ncol = dim(temp)
        ide = identity(nrow)
        if times is None:
            if method == 'lu':
                while True:
                    try:
                        Q, R = temp.lu()
                    except:
                        P, L, U = temp.plu()
                        Q = P.transpose() * L
                        R = U
                    temp1 = R * Q
                    diag1 = temp.diag()
                    diag2 = temp1.diag()
                    if all(
                            abs(diag1[i] - diag2[i]) < tol
                            for i in range(nrow)):
                        break
                    temp = temp1
            elif method == 'qr':
                while True:
                    Q, R = temp.qr()
                    temp1 = R * Q
                    diag1 = temp.diag()
                    diag2 = temp1.diag()
                    if all(
                            abs(diag1[i] - diag2[i]) < tol
                            for i in range(nrow)):
                        break
                    temp = temp1
        else:
            if method == 'lu':
                for i in range(times):
                    try:
                        Q, R = temp.lu()
                    except:
                        P, L, U = temp.plu()
                        Q = P.transpose() * L
                        R = U
                    temp = R * Q
            elif method == 'qr':
                for i in range(times):
                    Q, R = temp.qr()
                    temp = R * Q
        if merge_eigval:
            diagonals = temp.diag()
            diagonals = formatlist(diagonals, tofloat=True, tol=tols)
            eigval = []
            for t in diagonals:
                if t not in eigval:
                    eigval.append(t)
        else:
            eigval = temp.diag()
        eigvalmat = zeros(nrow)
        eigvalmat.filldiag(eigval)
        N = len(eigval)
        unit = identity(nrow)
        eigvecm = matrix([])
        for i in range(N):
            bk = build(nrow, 1, 1)
            current = eigval[i]
            if current == 0:
                bk = self.nullspace(True)
                norms = bk.norm()
                if norms != 0:
                    bk /= norms
                eigvecm.add(bk)
            else:
                for k in range(eigvec_times):
                    new = self - current * unit
                    try:
                        bk = new.inv_lu() * bk
                    except:
                        bk = new.nullspace(True)
                        norms = bk.norm()
                        if norms != 0:
                            bk /= norms
                        break
                    norms = bk.norm()
                    if norms != 0:
                        bk /= norms
                eigvecm.add(bk)
        #eigvec = [(self-ide*x).nullspace(True) for x in eigval]
        #eigvec = [x/x.norm() if (not x.isid() and x.norm() != 0) else x for x in eigvec]
        #if eigvecm.coln() != self.coln():
        #return self.eigen(tol = 0.1, formated = False, asfloat = True, eigvec_times = 2, times = None, merge_eigval = False)
        if formated:
            try:
                formating(temp)
                formating(eigvecm)
            except:
                pass
        if asfloat:
            tofloat(eigvalmat)
            tofloat(eigvecm)

        return eigvalmat, eigvecm

    def eigval(self,
               tol=0.1,
               formated=False,
               asfloat=True,
               asmatrix=False,
               times=None,
               tols=1e-3,
               method='lu'):
        if not self.is_square():
            return 'non-square matrices do not have eigenvalues and eigenvectors, \
maybe you want to find singular values? Then you can use svd method to get them.'

        temp = self.copy()
        nrow, ncol = dim(temp)
        ide = identity(nrow)
        if times is None:
            if method == 'lu':
                while True:
                    try:
                        Q, R = temp.lu()
                    except:
                        P, L, U = temp.plu()
                        Q = P.transpose() * L
                        R = U
                    temp1 = R * Q
                    diag1 = temp.diag()
                    diag2 = temp1.diag()
                    if all(
                            abs(diag1[i] - diag2[i]) < tol
                            for i in range(nrow)):
                        break
                    temp = temp1
            elif method == 'qr':
                while True:
                    Q, R = temp.qr()
                    temp1 = R * Q
                    diag1 = temp.diag()
                    diag2 = temp1.diag()
                    if all(
                            abs(diag1[i] - diag2[i]) < tol
                            for i in range(nrow)):
                        break
                    temp = temp1
        else:
            if method == 'lu':
                for i in range(times):
                    try:
                        Q, R = temp.lu()
                    except:
                        P, L, U = temp.plu()
                        Q = P.transpose() * L
                        R = U
                    temp = R * Q
            elif method == 'qr':
                for i in range(times):
                    Q, R = temp.qr()
                    temp = R * Q

        if formated:
            eigv = []
            diagonals = temp.diag()
            diagonals = formatlist(diagonals, asfloat, tols)
            for t in diagonals:
                if t not in eigv:
                    eigv.append(t)
        else:
            eigv = temp.diag()
        if asmatrix:
            eigvalmat = zeros(nrow)
            eigvalmat.filldiag(eigv)
            return eigvalmat
        return eigv

    def eigvec(self,
               tol=0.1,
               formated=False,
               eigvec_times=2,
               eigvals=None,
               times=None,
               formated2=True,
               tols=1e-3):
        if eigvals is None:
            eigval = self.eigval(tol,
                                 times=times,
                                 formated=formated2,
                                 tols=tols)
        else:
            eigval = eigvals
        nrow, ncol = self.dim()
        N = len(eigval)
        unit = identity(nrow)
        eigvecm = matrix([])
        for i in range(N):
            bk = build(nrow, 1, 1)
            current = eigval[i]
            if current == 0:
                bk = self.nullspace(True)
                norms = bk.norm()
                if norms != 0:
                    bk /= norms
                eigvecm.add(bk)
            else:
                for k in range(eigvec_times):
                    new = self - current * unit
                    try:
                        bk = new.inv_lu() * bk
                    except:
                        bk = new.nullspace(True)
                        norms = bk.norm()
                        if norms != 0:
                            bk /= norms
                        break
                    norms = bk.norm()
                    if norms != 0:
                        bk /= norms
                eigvecm.add(bk)
        if formated:
            try:
                formating(eigvecm, tol=tols)
            except:
                pass
        return eigvecm

    def addto(self, other, startind):
        temp, temp2 = other.copy(), self.copy()
        try:
            nrow, ncol = temp2.dim()
            rstart, cstart = startind
            for i in range(nrow):
                temp[rstart +
                     i] = temp[rstart +
                               i][:cstart] + temp2[i] + temp[rstart +
                                                             i][cstart + ncol:]
            return temp
        except:
            print('adding error')

    def inv_tri(self, mode=0):
        if mode == 0:
            nrow = self.rown()
            invrow = []
            unit = identity(nrow)
            # lower triangular matrix
            for i in range(nrow):
                current = [0 for x in range(i)]
                bcol = unit.getcol(i)
                for j in range(i, nrow):
                    b = bcol[j]
                    minus = sum([self[j, x] * current[x] for x in range(i, j)])
                    X = (b - minus) / self[j, j]
                    current.append(X)
                invrow.append(current)
            return matrix(invrow).transpose()
        else:
            # upper triangular matrix
            return self.transpose().inv_tri().transpose()

    def rown(self):
        return self.row_number

    def coln(self):
        return self.column_number

    def rows(self):
        return deepcopy(self.row)

    def cols(self):
        return self.transpose().row

    def split(self):
        # add all elements in the matrix to a list
        row = self.row
        return [x for y in row for x in y]

    def max(self):
        return max(self.split())

    def min(self):
        return min(self.split())

    def sort(self, mode=0):
        # mode = 0 : sort from smallest to biggest
        # mode = 1 : sort from biggest to smallest
        whole = self.split()
        if mode == 0:
            whole.sort()
        else:
            whole.sort(reverse=True)
        self.fillin(whole)

    def sorted(self, mode=0):
        temp = self.copy()
        temp.sort(mode)
        return temp

    def rounds(self, dec=None, new=False, formated=False):
        if new:
            temp = self.copy()
            rounds(temp, dec, formated)
            return temp
        else:
            rounds(self, dec, formated)

    def rotate(self, times=1):
        # times = n (n > 0): rotate right (clockwise) for n unit
        # times = -n (n > 0): rotate left (counterclockwise) for n unit
        # note that n could only be set to integers and has a period of 4,
        # which means rotate 90 degrees for each unit.
        temp = self.copy()
        nrow, ncol = temp.dim()
        if times == 0:
            return self
        elif times < 0:
            unit = abs(times) % 4
            if unit == 1:
                putlist = [
                    self[i, j] for j in range(ncol - 1, -1, -1)
                    for i in range(0, nrow)
                ]
                return form(putlist, ncol, nrow)
            else:
                for i in range(unit):
                    temp = temp.rotate(-1)
                return temp
        else:
            unit = times % 4
            if unit == 1:
                putlist = [
                    self[i, j] for j in range(0, ncol)
                    for i in range(nrow - 1, -1, -1)
                ]
                return form(putlist, ncol, nrow)
            else:
                for i in range(unit):
                    temp = temp.rotate()
                return temp

    def swap(self, m, n, mode=0):
        if isls(m) and isls(n):
            self[m], self[n] = self[n], self[m]
        if mode == 0:
            self.swaprow(m, n)
        elif mode == 1:
            self.swapcol(m, n)

    def swaprow(self, row1, row2):
        if any(x not in range(self.row_number) for x in [row1, row2]):
            return 'input row number is out of range'
        self.row[row1], self.row[row2] = self.row[row2], self.row[row1]

    def swapcol(self, column1, column2):
        if any(x not in range(self.column_number) for x in [column1, column2]):
            return 'input column number is out of range'
        nrow = self.rown()
        for i in self.row:
            i[column1], i[column2] = i[column2], i[column1]

    def uptri(self, formated=False, show_swaptime=False, asfloat=False):
        if not self.is_square():
            return 'non-square matrix cannot transform to triangular form'
        temp = self.copy()
        swaptime = 0
        nrow = temp.row_number
        for i in range(nrow):
            #print()
            #print(temp)
            first_col = temp.getcol(i)[i:]
            if any(x != 0 for x in first_col):
                N = nrow - i
                for k in range(N):
                    if first_col[k] != 0:
                        if k != 0:
                            swaprow(temp, i, k + i)
                            swaptime += 1
                        break

                first_row = temp[i]
                for j in range(i + 1, nrow):
                    current = temp[j, i]
                    if current != 0:
                        multi = -(current / temp[i, i])
                        temp[j] = [
                            temp[j, x] + multi * first_row[x]
                            for x in range(nrow)
                        ]
        if formated:
            formating(temp, asfloat)
        if show_swaptime:
            print(swaptime)
        return temp

    def mult_plu(self):
        result = self.plu()
        return result[0].transpose() * result[1] * result[2]

    def lu_full(self, formated=False, asfloat=False):
        temp = self.copy()
        nrow = temp.row_number
        ncol = temp.column_number
        L = zeros(nrow, nrow)
        N = min(nrow, ncol)
        for i in range(N - 1):
            diag1 = temp[i, i]
            if abs(diag1) < 1e-3:
                current_col = temp.getcol(i)[i:]
                abscol = [abs(h) for h in current_col]
                ind = abscol.index(max(abscol)) + i
                temp.swaprow(i, ind)
                L.swaprow(i, ind)
                diag1 = temp[i, i]
            #if diag1 != 0:
            #scalar = 1/diag1
            #L[i,i] = diag1
            #temp[i] = [j*scalar for j in temp[i]]
            #diag1 = 1
            first_row = temp[i]
            for j in range(i + 1, N):
                current = temp[j, i]
                if current != 0:
                    multi = -(current / diag1)
                    temp[j] = [
                        temp[j, x] + multi * first_row[x] for x in range(ncol)
                    ]
                    L[j, i] = -multi
        L.filldiag(1)
        U = temp
        #U = zeros(nrow, ncol)
        #for j in range(ncol):
        #for k in range(ncol):
        #U[j,k] = temp[j,k]
        if formated:
            formating(U, asfloat)
            formating(L, asfloat)
        #print(swaptime)
        return L, U

    def lu(self, formated=False, asfloat=False):
        # LU decomposition, return L, U in order (if the matrix does not have
        # a LU decomposition, then return a warning message)
        temp = self.copy()
        nrow = temp.row_number
        ncol = temp.column_number
        L = identity(nrow)
        N = min(nrow, ncol)
        for i in range(N - 1):
            diag1 = temp[i, i]
            if abs(diag1) < 1e-3:
                return 'this matrix does not has an LU decomposition, you can try its PLU decomposition'

            #if diag1 != 0:
            #scalar = 1/diag1
            #L[i,i] = diag1
            #temp[i] = [j*scalar for j in temp[i]]
            #diag1 = 1
            first_row = temp[i]
            for j in range(i + 1, N):
                current = temp[j, i]
                if current != 0:
                    multi = -(current / diag1)
                    temp[j] = [
                        temp[j, x] + multi * first_row[x] for x in range(ncol)
                    ]
                    L[j, i] = -multi
        U = temp
        #U = zeros(nrow, ncol)
        #for j in range(ncol):
        #for k in range(ncol):
        #U[j,k] = temp[j,k]
        if formated:
            formating(U, asfloat)
            formating(L, asfloat)
        return L, U

    def mult_lu(self):
        return mult(self.lu())

    def mult_lufull(self):
        return mult(self.lu_full())

    def leftm(self):
        return self * self.transpose()

    def rightm(self):
        return self.transpose() * self

    def plu(self, formated=False, asfloat=False):
        # PLU decomposition, return P, L, U in order
        temp = self.copy()
        nrow = temp.row_number
        ncol = temp.column_number
        P = identity(nrow)
        N = min(nrow, ncol)
        L = zeros(nrow, nrow)
        for i in range(N - 1):
            current_col = temp.getcol(i)[i:]
            abscol = [abs(h) for h in current_col]
            ind = abscol.index(max(abscol)) + i
            if ind != i:
                temp.swaprow(i, ind)
                P.swaprow(i, ind)
                L.swaprow(i, ind)
            diag1 = temp[i, i]

            #if diag1 != 0:
            #scalar = 1/diag1
            #L[i,i] = diag1
            #temp[i] = [j*scalar for j in temp[i]]
            #diag1 = 1
            first_row = temp[i]

            for j in range(i + 1, N):
                current = temp[j, i]
                if current != 0:
                    multi = -(current / diag1)
                    temp[j] = [
                        temp[j, x] + multi * first_row[x] for x in range(ncol)
                    ]
                    L[j, i] = -multi
        L.filldiag(1)
        U = temp
        return P, L, U

    def evd(self, tol=0.1, formated=False, times=None):
        # eigendecomposition
        if not self.is_square():
            return 'an eigendecomposition requires an nxn matrix, but this matrix is not a square matrix'
        A, Q = self.eigen(tol, formated, times=times)
        if not Q.invertible():
            return 'this matrix is not diagonalizable, so it has no eigendecomposition'
        return Q, A, Q.inv_lu()

    def rf(self):
        # rank factorization
        B = self.rref()
        nrow, ncol = B.dim()
        pivot_column = []
        for i in range(nrow):
            for j in range(ncol):
                if B[i][j] != 0:
                    pivot_column.append(j)
                    break
        not_allzeros = [y for y in range(nrow) if set(B[y]) != {0}]
        C = bind([self.getcol(x) for x in pivot_column], 1)
        F = bind([B[y] for y in not_allzeros])
        return C, F

    def inv_diag(self):
        temp = self.copy()
        temp.filldiag([1 / x if x != 0 else x for x in temp.diag()])
        return temp

    def diag(self, mode=0):
        N = min(self.row_number, self.column_number)
        if mode == 0:
            return [self[i][i] for i in range(N)]
        else:
            return [self[i][N - 1 - i] for i in range(N)]

    def svd(self, tol=0.1, times=None, formated=True):
        '''
        if the multiple of the result U, D, VT is not very close to the
        original matrix, maybe the iterative times of the svd is not enough,
        just set the times to be larger, larger times ensures more precision
        of the svd.
        '''
        trans = self.transpose()
        left = self * trans
        right = trans * self
        X, V = right.eigen(tol, times=times, merge_eigval=formated)

        D = build(self.row_number, self.column_number, 0)
        diagonal = X.diag()
        putlist = [y**0.5 for y in diagonal if formatnumber(y) > 0]
        for i in range(len(putlist)):
            D[i][i] = putlist[i]
        try:
            U = self * V.transpose().inv_lu() * D.inv_diag()
            if type(U) == str:
                U = left.eigen(tol, times=times, merge_eigval=formated)[1]
                # test for signs of columns in U and V
                # test for AV = UD
                t1 = self * V
                t2 = U * D
                nrow, ncol = t1.dim()
                G = min(nrow, ncol)
                for j in range(G):
                    if any(
                            sign(t1[x, j], t2[x, j]) is False
                            for x in range(nrow)):
                        for k in range(U.row_number):
                            U[k, j] *= -1
        except:
            U = left.eigen(tol, times=times, merge_eigval=formated)[1]
            # test for signs of columns in U and V
            # test for AV = UD
            t1 = self * V
            t2 = U * D
            nrow, ncol = t1.dim()
            G = min(nrow, ncol)
            for j in range(G):
                if any(sign(t1[x, j], t2[x, j]) is False for x in range(nrow)):
                    for k in range(U.row_number):
                        U[k, j] *= -1

        return U, D, V.transpose()

        # calculate the eigenvalues

    def invertible(self):
        return self.is_square() and self.det() != 0

    def appro(self, error=1e-3):
        # round the elements of the matrix to the nearest number within the error
        pass

    def aeq(self, other, error=1e-3):
        # compare 2 matrices each other within an error, if the difference
        # of every two corresponding elements are within the error then
        # return True, else return False
        if self.dim() != other.dim():
            return False
        r = self - other
        element = r.split()
        return all(abs(x) < error for x in element)

    def extract(self, n, mode=0, notasmatrix=0):
        # extract a row/column or a range of rows/columns or some of the
        # chosen rows/columns from the matrix and form as a new matrix,
        # return the new matrix
        # mode 0 : extract rows
        # mode != 0 : extract columns
        # if n is an integer: extract nth row/column
        # if n is a 2-element tuple (or 3-element): extract range of n (the
        # third element, if exists, is step), START is from the beginning,
        # END is extract until the end of the matrix
        # if n is a list: extract the rows/columns by indexes in the list
        temp = self.copy()
        if mode == 0:
            if isinstance(n, int):
                result = temp.getrow(n)
                if notasmatrix:
                    return result
                else:
                    return matrix(result)
            elif isinstance(n, tuple):
                if len(n) == 2:
                    n0, n1 = n
                    if n[0] == START:
                        n0 = 0
                    if n[1] == END:
                        n1 = temp.row_number
                    exrange = range(n0, n1)
                elif len(n) == 3:
                    n0, n1, n2 = n
                    if n[0] == START:
                        n0 = 0
                    if n[1] == END:
                        n1 = temp.row_number
                    exrange = range(n0, n1, n2)
                else:
                    return 'range tuple should be either 2 or 3 elements'
                result = [temp.getrow(x) for x in exrange]
            elif isinstance(n, list):
                result = [temp.getrow(x) for x in n]
            if notasmatrix:
                return result
            else:
                return matrix(result)
        else:
            if isinstance(n, int):
                result = temp.getcol(n)
                if notasmatrix:
                    return result
                else:
                    return matrix(result, dr=1)
            elif isinstance(n, tuple):
                if len(n) == 2:
                    n0, n1 = n
                    if n[0] == START:
                        n0 = 0
                    if n[1] == END:
                        n1 = temp.column_number
                    exrange = range(n0, n1)
                elif len(n) == 3:
                    n0, n1, n2 = n
                    if n[0] == START:
                        n0 = 0
                    if n[1] == END:
                        n1 = temp.column_number
                    exrange = range(n0, n1, n2)
                else:
                    return 'range tuple should be either 2 or 3 elements'
                result = [[i[x] for x in exrange] for i in temp.row]
            elif isinstance(n, list):
                result = [[i[x] for x in n] for i in temp.row]
            if notasmatrix:
                return result
            else:
                return matrix(result)

    def reshape(self, m, n, default=0):
        # return a reshape matrix with the elements of the original matrix
        # with row number m and column number n
        elements = self.split()
        new = form(elements, m, n, default)
        return new

    def __invert__(self):
        return self.transpose()

    def pinv(self, tol=0.1, times=None):

        rank = self.rank(True)
        if rank == self.row_number == self.column_number:
            return self.inv_lu()
        else:
            trans = ~self
            if rank == self.row_number:
                return trans * ((self * trans).inv_lu())
            elif rank == self.column_number:
                return (trans * self).inv_lu() * trans
            else:
                U, D, VT = self.svd(tol, times=times)
                diag = D.diag()
                diag = [1 / x if x != 0 else x for x in diag]
                D.filldiag(diag)
                D = D.transpose()
                return VT.transpose() * D * U.transpose()

    def to_upper_triangular(self, formated=False):
        return self.uptri(formated)

    def to_lower_triangular(self, formated=False):
        return self.uptri(formated).transpose()

    def lowtri(self, formated=False):
        return self.uptri(formated).transpose()

    def is_semiorthogonal(self):
        pass

    def formating(self, asfloat=False, tol=1e-3):
        # here we ensure all of the values in the matrix and integers and
        # floats without disrupting data like 1.0 which is actually 1
        self.row = formatrow(self.row, asfloat, tol)

    def T(self):
        return self.transpose()

    def add(self, other, direction='r', new=False):
        # if direction == 'r'/'right' then combine matrix other with self to the right,
        # if direction == 'd'/'down' then combine matrix other with self to the bottom,
        # 'l'/'left' to the left, 'u'/'up' to the top
        if other == 'self':
            other = self.copy()

        if new:
            temp = deepcopy(self)
            temp.add(other, direction)
            return temp
        if self.row == []:
            self.row = other.row
            self.row_number, self.column_number = other.row_number, other.column_number
        else:
            if direction in ['r', 'right']:
                if other.row_number != self.row_number:
                    return 'Error: to combine a matrix to the left or right, the matrix should have same row number as the original matrix'
                for i in range(self.row_number):
                    self[i] += other[i]
                self.column_number += other.column_number
            elif direction in ['l', 'left']:
                if other.row_number != self.row_number:
                    return 'Error: to combine a matrix to the left or right, the matrix should have same row number as the original matrix'
                for i in range(self.row_number):
                    self[i] = other[i] + self[i]
                self.column_number += other.column_number
            elif direction in ['d', 'down']:
                if other.column_number != self.column_number:
                    return 'Error: to combine a matrix to the top or bottom, the matrix should have same column number as the original matrix'
                self.row += other.copy().row
                self.row_number += other.row_number
            elif direction in ['u', 'up']:
                if other.column_number != self.column_number:
                    return 'Error: to combine a matrix to the top or bottom, the matrix should have same column number as the original matrix'
                self.row = other.copy().row + self.row
                self.row_number += other.row_number

    def addn(self, other, direction='r', new=False):
        if other == 'self':
            other = self.copy()
        if self.row == []:
            self.row = other.row
        temp = self.copy()
        temp.add(other, direction, new)
        return temp

    def save(self, name, file='txt'):
        with open(name + '.' + file, "w") as f:
            f.write(self.__str__())

    def addrow(self, new_row):
        if not isinstance(new_row, list):
            return 'the row you want to add should be a list'
        if len(new_row) != self.column_number:
            return 'Error: the input row\'s length does not match the column number of the matrix'
        self.row.append(new_row)
        self.row_number += 1

    def addcol(self, new_column):
        if not isinstance(new_column, list):
            return 'the column you want to add should be a list'
        if len(new_column) != self.row_number:
            return 'Error: the input column\'s length does not match the row number of the matrix'
        row = self.row
        for i in range(self.row_number):
            row[i].append(new_column[i])

    def rprow(self, old_row, new_row):
        if old_row not in range(self.row_number):
            return 'input row number is out of range'
        if not isinstance(new_row, list):
            return 'the row you want to replace should be a list'
        self.row[old_row] = new_row

    def rpcol(self, old_column, new_column):
        if old_column == -1:
            old_column = self.coln() - 1
        if old_column not in range(self.column_number):
            return 'input column number is out of range'
        if not isinstance(new_column, list):
            return 'the column you want to replace should be a list'
        if len(new_column) != self.row_number:
            raise ValueError(
                'the new column does not have right length to fit in this matrix'
            )
        row = self.row
        for i in range(self.row_number):
            row[i][old_column] = new_column[i]

    def change(self, row1, column1, element):
        if row1 not in range(self.row_number) or column1 not in range(
                self.column_number):
            return 'row number or column number is out of range'
        self.row[row1][column1] = element

    def swape(self, row1, column1, row2, column2):
        if any(x not in range(self.row_number)
               for x in [row1, row2]) or any(x not in range(self.column_number)
                                             for x in [column1, column2]):
            return 'please ensure both of the indexes you want to exchange are in the range of matrix size'
        self.row[row1][column1], self.row[row2][column2] = self.row[row2][
            column2], self.row[row1][column1]

    def rprowall(self, row1, element):
        if row1 not in range(self.row_number):
            return 'row number is out of range'
        self.row[row1] = [element for i in range(self.column_number)]

    def rpcolall(self, column1, element):
        if column1 == -1:
            column1 = self.coln() - 1
        if column1 not in range(self.column_number):
            return 'column number is out of range'
        row = self.row
        for i in range(self.row_number):
            row[i][old_column] = element

    def rank(self, formated=False):
        ''' Return the rank of the matrix. '''
        temp = self.ref(formated)
        return [any(x != 0 for x in j) for j in temp.row].count(True)

    def is_fullrank(self):
        ''' Return a message showing whether the matrix is in full rank or not. '''
        if self.rank() == self.row_number:
            return 'this matrix is in full rank'
        else:
            return 'this matrix is not in full rank'

    def is_aug(self):
        return self.column_number - self.row_number == 1

    def isid(self):
        nrow = self.row_number
        return self.is_square() and all(
            self[i, i] == 1 and all(self[i, x] == 0
                                    for x in range(nrow) if x != i)
            for i in range(nrow))

    def solve(self, b):
        temp = self.copy()
        temp.add(b)
        return temp.solve_lineq()

    def solve_lineq(self, varname=None):
        if not self.is_aug():
            return 'this matrix is not in an augmented matrix'
        result = self.rref()
        nrow = result.row_number
        ncol = result.column_number - 1
        if result.cut(0, nrow, 1).isid():
            if varname is None:
                return [fractcheck(result[i, ncol]) for i in range(nrow)]
            else:
                return {
                    varname[i]: fractcheck(result[i, ncol])
                    for i in range(nrow)
                }
        else:
            if varname is None:
                # return the solution basis in a list
                pass
            else:
                equations = []
                for k in range(nrow):
                    if any(x != 0 for x in result[k][:-1]):
                        neweq = ' + '.join([
                            f'''{fractcheck(result[k][j])
                                            if result[k][j] not in [-1,1]
                                            else ("-" if result[k][j]
                                            == -1 else "")}{varname[j]}'''
                            for j in range(nrow) if result[k][j] != 0
                        ])
                        neweq += f' = {result[k][ncol]}'
                        equations.append(neweq)
                return equations

    def fullrank(self):
        return self.is_square() and self.rank() == self.row_number

    def rfullrank(self):
        return self.rank() == self.row_number

    def cfullrank(self):
        return self.rank() == self.column_number

    def ref(self, formated=False):
        ''' Calculate the row echelon form of the matrix, return as a new
            matrix object. '''
        pivot_row = None
        pivot_column = None
        pivot = None
        counter = 0
        temp = self.copy()
        trans = temp.transpose()
        while counter < temp.row_number:
            for i in range(trans.row_number):
                for j in range(counter, len(trans.row[i])):
                    if trans.row[i][j] != 0:
                        pivot_row = j
                        pivot_column = i
                        pivot = trans.row[i][j]
                        break
                if pivot != None:
                    break
            if pivot == None:
                break
            new_row = [temp.row[x]
                       for x in range(counter)] + [temp.row[pivot_row]] + [
                           temp.row[k] for k in range(counter, temp.row_number)
                           if k != pivot_row
                       ]
            for x in range(len(new_row[counter])):
                new_row[counter][x] /= pivot
            for t in range(counter + 1, temp.row_number):
                multi = new_row[t][pivot_column] * (-1)
                new_row[t] = [
                    new_row[t][i] + temp.row[pivot_row][i] * multi
                    for i in range(len(new_row[t]))
                ]
            if formated:
                new_row = formatrow(new_row)
            pivot = None
            temp = matrix(new_row)
            trans = temp.transpose()
            counter += 1
        temp.formating()
        return temp

    def rref(self, formated=False):
        ''' Calculate the reduced row echelon form of the matrix, return
            as a new matrix object. '''
        temp = self.ref(formated)
        counter = temp.row_number - 1
        pivot_row = 0
        pivot_column = 0
        while counter > 0:
            for i in range(len(temp.row[counter])):
                if temp.row[counter][i] == 1:
                    pivot_row = counter
                    pivot_column = i
                    break
            new_row = temp.row
            for t in range(pivot_row):
                multi = new_row[t][pivot_column] * (-1)
                new_row[t] = [
                    new_row[t][i] + temp.row[pivot_row][i] * multi
                    for i in range(len(new_row[t]))
                ]
            temp = matrix(new_row)
            counter -= 1
        return temp

    def apply(self, func):
        temp = self.copy()
        temp.row = [[func(i) for i in each] for each in temp.row]
        return temp

    def invert(self):
        return self.apply(lambda s: 1 - s)

    def vector(self, ind, mode=0):
        if mode == 0:
            return self.cut(ind, ind + 1, 1)
        else:
            return self.cut(ind, ind + 1)


class cell:
    def __init__(self, x, y, xrange=None, yrange=None):
        self.x = x
        self.y = y
        self.xrange = xrange
        self.yrange = yrange

    def up(self, step=1):
        xrange = self.xrange
        yrange = self.yrange
        if xrange is None:
            return cell(self.x - step, self.y)
        newx = self.x
        newy = self.y
        for i in range(step):
            newx -= 1
            if newx < xrange[0]:
                newx = xrange[1]
        return cell(newx, newy, xrange, yrange)

    def down(self, step=1):
        xrange = self.xrange
        yrange = self.yrange
        if xrange is None:
            return cell(self.x + step, self.y)
        newx = self.x
        newy = self.y
        for i in range(step):
            newx += 1
            if newx > xrange[1]:
                newx = xrange[0]
        return cell(newx, newy, xrange, yrange)

    def left(self, step=1):
        xrange = self.xrange
        yrange = self.yrange
        if yrange is None:
            return cell(self.x, self.y - step)
        newx = self.x
        newy = self.y
        for i in range(step):
            newy -= 1
            if newy < yrange[0]:
                newy = yrange[1]
        return cell(newx, newy, xrange, yrange)

    def right(self, step=1):
        xrange = self.xrange
        yrange = self.yrange
        if yrange is None:
            return cell(self.x, self.y + step)
        newx = self.x
        newy = self.y
        for i in range(step):
            newy += 1
            if newy > yrange[1]:
                newy = yrange[0]
        return cell(newx, newy, xrange, yrange)

    def move(self, step=1, mode='right'):
        if mode == 'right':
            xrange = self.xrange
            yrange = self.yrange
            if yrange is None:
                return cell(self.x, self.y + step)
            newx = self.x
            newy = self.y
            for i in range(step):
                newy += 1
                if newy > yrange[1]:
                    newy = yrange[0]
                    newx += 1
                    if newx > xrange[1]:
                        newx = xrange[0]
            return cell(newx, newy, xrange, yrange)
        elif mode == 'left':
            xrange = self.xrange
            yrange = self.yrange
            if yrange is None:
                return cell(self.x, self.y - step)
            newx = self.x
            newy = self.y
            for i in range(step):
                newy -= 1
                if newy < yrange[0]:
                    newy = yrange[1]
                    newx -= 1
                    if newx < xrange[0]:
                        newx = xrange[1]
            return cell(newx, newy, xrange, yrange)

    def __call__(self, mode='list'):
        if mode == 'list':
            return [self.x, self.y]
        elif mode == 'tuple':
            return self.x, self.y

    def __repr__(self):
        return str((self.x, self.y))

    def swap(self):
        return cell(self.y, self.x)

    def __getitem__(self, ind):
        return [self.x, self.y][ind]

    def __setitem__(self, ind, value):
        if ind == 0:
            self.x = value
        elif ind == 1:
            self.y = value

    def __eq__(self, other):
        if not isinstance(other, cell):
            return self.x == other[0] and self.y == other[1]
        return self.x == other.x and self.y == other.y

    def moves(self, mode='right', step=1):
        if mode == 'right':
            xrange = self.xrange
            yrange = self.yrange
            if yrange is None:
                self.y += step
            else:
                newx = self.x
                newy = self.y
                for i in range(step):
                    newy += 1
                    if newy > yrange[1]:
                        newy = yrange[0]
                        newx += 1
                        if newx > xrange[1]:
                            newx = xrange[0]
                self.x, self.y = newx, newy
        elif mode == 'left':
            xrange = self.xrange
            yrange = self.yrange
            if yrange is None:
                self.y -= step
            else:
                newx = self.x
                newy = self.y
                for i in range(step):
                    newy -= 1
                    if newy < yrange[0]:
                        newy = yrange[1]
                        newx -= 1
                        if newx < xrange[0]:
                            newx = xrange[1]
                self.x, self.y = newx, newy


def det(self, showtime=False):
    return self.det(showtime)


def transpose(self):
    return self.transpose()


def inverse(self):
    return self.inverse()


def dot(self, other):
    return self.dot(other)


def outer(self, other):
    pass


def cross(self, other, tomatrix=True):
    return self.cross(other, tomatrix)


def conv(self, other):
    pass


def wedge(self, other):
    pass


def trace(self):
    return self.trace()


def info(self):
    return self.info()


def rowspace(self):
    pass


def colspace(self):
    pass


def nullspace(self):
    pass


def kernel(self):
    pass


def rank(self):
    return self.rank()


def is_fullrank(self):
    return self.is_fullrank()


def ref(self):
    return self.ref()


def rref(self):
    return self.rref()


def copy(self):
    return self.copy()


def root(a, number):
    return a.root(number)


def diag(self):
    return self.diag()


def diagonalizable(self):
    return self.diagonalizable()


def eigenvalues(self):
    return self.eigenvalues()


def eigenvectors(self):
    return self.eigenvectors()


def recid(m, n=None):
    if n is None:
        n = m
    return diagonal(1, m, n)


def identity(size):
    return matrix([[0 if x != i else 1 for x in range(size)]
                   for i in range(size)])


def is_square(a):
    return a.is_square()


def size(a):
    return a.size()


def build(row_number, column_number=None, element=0):
    ''' Build a matrix with given size filling with same element. '''
    if column_number is None:
        column_number = row_number
    return matrix([[element for i in range(column_number)]
                   for j in range(row_number)])


def square(m, element=0):
    return build(m, m, element)


def mrange(nrow, ncol=None, start=None, stop=None, default=0):
    if ncol is None:
        ncol = nrow
    result = build(nrow, ncol, default)
    if start is None and stop is None:
        start = 1
        stop = nrow * ncol
    if stop is None:
        result.fill(1, start)
    else:
        result.fill(start, stop)
    return result


def swaprow(a, row1, row2):
    a.swaprow(row1, row2)


def swapcol(a, column1, column2):
    a.swapcol(column1, column2)


def rprow(a, old_row, new_row):
    a.rprow(old_row, new_row)


def rpcol(a, old_column, new_column):
    a.rpcol(old_column, new_column)


def change(a, row1, column1, element):
    a.change(row1, column1, element)


def swape(a, row1, column1, row2, column2):
    a.swape(row1, column1, row2, column2)


def to_upper_triangular(a, formated=False):
    return a.to_upper_triangular(formated)


def to_lower_triangular(a):
    pass


def is_upper_triangular(a):
    return a.is_upper_triangular()


def rprow_all(a, row1, element):
    pass


def rpcol_all(a, column1, element):
    pass


def sets(ind, c, x, mode=0):
    if mode == 1:
        b = list(x)
        sets(ind, c, b, 2)
        x = ''.join(b)
        return x
    else:
        if mode == 2:
            if not isinstance(ind, list):
                ind = [ind]
        if mode == 0:
            if not isinstance(ind[0], list):
                ind = [ind]
        if not isls(c):
            for i in ind:
                x[i] = c
        else:
            N = min(len(ind), len(c))
            for i in range(N):
                x[ind[i]] = c[i]


def up(ind, dis=1):
    return [ind[0] - dis, ind[1]]


def down(ind, dis=1):
    return [ind[0] + dis, ind[1]]


def left(ind, dis=1):
    return [ind[0], ind[1] - dis]


def right(ind, dis=1):
    return [ind[0], ind[1] + dis]


def showall(mat, linebreak=None, cat=''):
    result = ''
    if linebreak is None:
        for i in mat.row:
            result += cat.join([str(x) for x in i])
    else:
        N = len(mat.row)
        for i in range(N):
            result += cat.join([str(x) for x in mat.row[i]])
            if i != N - 1:
                result += linebreak
    return result


def modstr(a, ind, value):
    a = sets(ind, value, a, 1)
    return a


def rounding(num, dec=None, tostr=False):
    if dec is None or isinstance(num, int):
        return num
    if isinstance(num, fractions.Fraction):
        num = float(num)
    numstr = str(num)
    ind = numstr.index('.')
    if dec == 0:
        intpart = eval(numstr[:ind])
        return intpart + 1 if eval(numstr[ind + 1]) >= 5 else intpart
    tol = len(numstr) - ind - 1
    if tol < dec:
        return f'{num:.{dec}f}' if tostr else num
    elif tol == dec:
        return num
    else:
        if eval(numstr[ind + dec + 1]) >= 5:
            temp = str(num + 10**-dec)[:ind + dec + 1]
            result = eval(temp)
            resultstr = str(result)
            if len(resultstr) - resultstr.index('.') - 1 == dec:
                return result
            else:
                return f'{result:.{dec}f}' if tostr else result

        else:
            return eval(numstr[:ind + dec + 1])


fractcheck = lambda x: x if x != 0 else 0


def formating(t, asfloat=False, tol=1e-3):
    t.formating(asfloat, tol)


def test(program, digit=3, output=True, printed=False):
    start = time.time()
    result = eval(program)
    stop = time.time()
    spend = stop - start
    second = rounding(spend % 60, digit)
    minute = int(spend // 60)
    hour = int(spend // 3600)
    if output:
        if not printed:
            return f'cost time: {hour}h {minute}m {second}s, result is {result}'
        else:
            print(f'cost time: {hour}h {minute}m {second}s')
            print('result is')
            print(result)
            return
    return f'cost time: {hour}h {minute}m {second}s'


#a = build(100, 100, 3)
#for f in range(100):a.row[f][f] = 1
#for f in range(30):a.row[f][f-1] = f - 3
#b = to_upper_triangular2(a)
#print(test('b = to_upper_triangular2(a)'))
#print(b)
#if isinstance(b, matrix): print(b.is_upper_triangular())
#input()
def dim(a):
    return a.dim()


def det2(self, mode='digit'):
    if not isinstance(self[0][0], matrix):
        if mode == 'poly':
            result = __import__('polynomial').poly('0')
        else:
            result = 0
    else:
        result = build(self[0][0].row_number, self[0][0].row_number, 0)
    if self.row_number == 1:
        result = self.row[0][0]
    elif self.row_number == 2:
        row = self.row
        result = row[0][0] * row[1][1] - row[0][1] * row[1][0]
    else:
        row = self.row
        for j in range(len(row[0])):
            temp = det2(
                matrix([[y[x] for x in range(len(y)) if x != j]
                        for y in row[1:]]))
            temp2 = row[0][j]
            if j % 2 == 0:
                result += temp2 * temp
            else:
                result -= temp2 * temp

    if isinstance(result, matrix):
        return result
    if isinstance(result, __import__('polynomial').polynomial):
        return result
    result = fractions.Fraction(result).limit_denominator().__float__()
    return int(result) if result.is_integer() else result


def sums(*obj):
    if isinstance(obj[0], list):
        obj = obj[0]
        if type(obj[0]) in [int, float, fractions.Fraction, complex]:
            return sum(obj)
        temp = obj[0].copy()
        for i in obj[1:]:
            temp += i
        return temp
    else:
        return sums(list(obj))

    return temp


def mult(*obj):
    if isls(obj[0]):
        obj = obj[0]
        if type(obj[0]) in [int, float, fractions.Fraction, complex]:
            first = 1
            for i in obj:
                first *= i
            return first
        temp = obj[0].copy()
        for k in obj[1:]:
            temp *= k
        return temp
    else:
        return mult(list(obj))


def form(val, nrow, ncol=None, default=0):
    if ncol is None:
        ncol = nrow
    result = build(nrow, ncol, default)
    for i in range(nrow):
        for j in range(ncol):
            if len(val) > 0:
                result[i, j] = val.pop(0)
            else:
                break
    return result


def norm(a):
    if isinstance(a, matrix):
        return a.norm()
    else:
        return sum([x**2 for x in a])**0.5


def rounds(a, dec=None, formated=False):
    if formated:
        formating(a)
    nrow, ncol = dim(a)
    for i in range(nrow):
        for j in range(ncol):
            a[i][j] = rounding(a[i][j], dec)


def diagonal(element, nrow=None, ncol=None):
    if nrow is None and ncol is None:
        if not isls(element):
            element = [element]
        result = identity(len(element))
    else:
        if ncol is None:
            ncol = nrow
        result = build(nrow, ncol, 0)
    result.filldiag(element)
    return result


I = ids = identity


def bind(n, mode=0):
    n = [matrix([x]) if not isinstance(x, matrix) else x for x in n]
    first = n[0]
    for i in n[1:]:
        first.add(i, 'd')
    if mode == 0:
        return first
    elif mode == 1:
        return first.transpose()


E = 2.718281828459045


def exp(num, dec=None):
    if type(num) == matrix:
        return form([exp(i, dec) for i in num.split()], *num.dim())
    result = E**num
    if dec is None:
        return result
    return rounding(result, dec)


def isls(a):
    return type(a) in [list, tuple, set]


def ones(m, n=None):
    if n is None:
        n = m
    return build(m, n, 1)


def zeros(m, n=None):
    if n is None:
        n = m
    return build(m, n, 0)


def allis(a, element):
    if type(a) == matrix:
        return a.allis(element)
    elif callable(element):
        return all(element(x) for x in a)
    elif isls(element):
        return all(all(t(x) for t in element) for x in a)
    else:
        return all(x == element for x in a)


def anyis(a, element):
    if type(a) == matrix:
        return a.anyis(element)
    elif callable(element):
        return any(element(x) for x in a)
    elif isls(element):
        return any(all(t(x) for t in element) for x in a)
    else:
        return any(x == element for x in a)


def allnot(a, element):
    if type(a) == matrix:
        return a.allnot(element)
    return not anyis(a, element)


def anynot(a, element):
    if type(a) == matrix:
        return a.anynot(element)
    return not allis(a, element)


def exist(a, element, n, mode=0):
    return a.exist(element, n, mode)


def nexist(a, element, n, mode=0):
    return a.nexist(element, n, mode)


def same(a, element, n, mode=0):
    return a.same(element, n, mode)


def nsame(a, element, n, mode=0):
    return a.nsame(element, n, mode)


def formatnumber(num, tofloat=False, tol=1e-3):
    if abs(num) < tol:
        return 0
    types = type(num)
    if types != int:
        if types == complex:
            num = num.real
        y = fractions.Fraction(num).limit_denominator()
        return int(y) if y.__float__().is_integer() else (
            y if not tofloat else float(y))
    else:
        return num


def formatlist(x, tofloat=False, tol=1e-3):
    result = [formatnumber(i, tofloat, tol) for i in x]
    return result


def formatrow(x, tofloat=False, tol=1e-3):
    x = [formatlist(i, tofloat, tol) for i in x]
    return x


def tofloat(x):
    nrow, ncol = x.dim()
    for i in range(nrow):
        for j in range(ncol):
            x[i][j] = float(x[i][j])


def nullspace(a, formated=False):
    return a.nullspace(formated)


def null(a, formated=True):
    return a.null(formated)


def mag(t):
    return (sum([x**2 for x in t]))**0.5


def normal(t):
    x = mag(t)
    return [j / x for j in t]


def sign(a, b):
    if (a > 0 and b > 0) or (a < 0 and b < 0):
        return True
    else:
        return False


def sg(num):
    if num < 0:
        return -1
    elif num > 0:
        return 1
    else:
        return 0


def rown(a):
    return a.rown()


def coln(a):
    return a.coln()


def rowname(a, ind=None):
    return a.rowname(ind)


def colname(a, ind=None):
    return a.colname(ind)


def vector(t, mode=0):
    result = matrix([t])
    return result if mode == 0 else result.transpose()


def vecform(element, length, mode=0):
    result = matrix([element for i in range(length)])
    return result if mode == 0 else result.transpose()


def tomatrix(a, tr=0):
    if not isls(a[0]):
        a = [a]
    result = matrix([list(i) for i in a])
    if tr == 0:
        return result
    return result.transpose()


def dolist(a, b, func):
    N = min(len(a), len(b))
    return [func(a[i], b[i]) for i in range(N)]


def save(obj, filename):
    with open(filename, "w") as f:
        f.write(str(obj))


def ind_to_dim(ind, width, backspace_num=0, natural=False):
    rownum, colnum = divmod(ind, width + backspace_num)
    if natural:
        rownum += 1
        colnum += 1
    return rownum, colnum


def get_from_ind(obj, ind, width=None, backspace_num=0, natural=False):
    if type(obj) == matrix:
        width = obj.coln()
    return obj[ind_to_dim(ind, width, backspace_num, natural)]


def dim_to_ind(x, y, width, backspace_num=0, natural=False):
    if natural:
        x -= 1
        y -= 1
    width += backspace_num
    return width * x + y


def get_from_dim(obj, x, y, width, backspace_num=0, natural=False):
    return obj[dim_to_ind(x, y, width, backspace_num, natural)]