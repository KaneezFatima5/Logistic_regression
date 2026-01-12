import unittest
from src.logistic_regression import *

class MyTestCase(unittest.TestCase):
    def test_add_intercept(self):
        x1 = np.array([[1, 3], [3, 4]])
        expected = np.array([[1,1, 3], [1,3, 4]])
        result = add_intercept(x1)
        self.assertTrue((expected == result).all())

    def test_standardize(self):
        x1 = np.array([[1, 3], [3, 4]])
        x2 = np.array([[2, 5], [2, 1]])
        expected1 = [[-1., -1.],[ 1.,  1.]]
        result1 = standardize(x1, x1)
        self.assertTrue((expected1 == result1).all())
        expected2 = [[ 0.,  3.],[ 0., -5.]]
        result2 = standardize(x1, x2)
        self.assertTrue((expected2 == result2).all())




if __name__ == '__main__':
    unittest.main()
