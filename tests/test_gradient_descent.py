import unittest
from src.gradient_descent import *


class MyTestCase(unittest.TestCase):
    def test_calc_logit(self):
        x1 = np.array([[1, 3], [3, 4]])
        w1 = np.array([[1, 0], [0, 1]])
        b1 = np.array([1, 2])
        expected1 = np.array([[2, 5], [4, 6]])
        result1 = calc_logit(x1, w1, b1)
        self.assertTrue((expected1 == result1).all())

        x2 = np.array([[1, 3, 5], [3, 4, 6]])
        w2 = np.array([[1, 0], [0, 1], [1, 2]])
        b2 = np.array([1, 2])
        expected2 = np.array([[7, 15], [10, 18]])
        result2 = calc_logit(x2, w2, b2)
        self.assertTrue((expected2 == result2).all())

    def test_calc_probabilities(self):
        z1 = np.array([[2, 5], [4, 6]])
        expected = np.array([[0.04742587, 0.95257413], [0.11920292, 0.88079708]])
        result = calc_probabilities(z1)
        self.assertTrue((expected == result).all())

    def test_calc_weight_gradient(self):
        x1 = np.array([[1, 2], [3, 4]])
        y1 = np.array([[1, 0], [0, 1]])
        p1 = np.array([[0, 1], [1, 0]])
        expected = np.array([[1.0, -1.0], [1.0, -1.0]])
        result = calc_weight_gradient(x1, y1, p1, 2)
        self.assertTrue((expected == result).all())

    def test_calc_bias_gradient(self):
        y1 = np.array([[1, 0, 3], [0, 1, 3]])
        p1 = np.array([[0, 1, 2], [2, 0, 1]])
        expected = [0.5, 0.0, -1.5]
        result = calc_bias_gradient(y1, p1, 2)
        self.assertTrue((expected == result).all())

    def test_calc_loss(self):
        y1 = np.array([[1, 0], [1, 0]])
        p1 = np.array([[1, 0], [1, 0]])
        expected1 = 0
        result1 = calc_loss(y1, p1, 2)
        self.assertAlmostEqual(expected1, result1, 7)

        y2 = np.array([[1, 0], [1, 0]])
        p2 = np.array([[0.7, 0.3], [0.8, 0.2]])
        expected2 = 0.289909247626
        result2 = calc_loss(y2, p2, 2)
        self.assertAlmostEqual(expected2, result2, 7)

    def test_gradient_descent(self):
        x = np.array([[5, 0], [0, 4]])
        y = np.array([[1, 0], [0, 1]])
        w = np.array([[0.8, 0.2], [0.7, 0.3]])
        b = np.array([0.6, 0.3])
        result_w, result_b = gradient_descent(x, y, 0.001, w, b, 1e-4)
        print(result_w, result_b)
        pass


if __name__ == "__main__":
    unittest.main()
