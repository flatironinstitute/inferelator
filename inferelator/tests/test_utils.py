import unittest
from inferelator import utils
from inferelator.utils import Validator as check
import pandas as pd
import numpy as np
import tempfile
import shutil
import os


class TestUtils(unittest.TestCase):

    def test_dataframe_set_diag(self):

        def check_diag(df, val):
            for match in df.index.intersection(df.columns):
                if df.loc[match, match] == val:
                    pass
                else:
                    return False
            return True

        data_frame = pd.DataFrame(np.ones((20, 10)), index=list(range(20)), columns=list(range(10)))
        data_frame2 = utils.df_set_diag(data_frame, 0, copy=True)

        self.assertTrue(check_diag(data_frame2, 0))
        self.assertEqual(data_frame2.sum().sum(), 190)
        self.assertTrue(check_diag(data_frame, 1))
        self.assertEqual(data_frame.sum().sum(), 200)

        utils.df_set_diag(data_frame, 0, copy=False)
        self.assertTrue(check_diag(data_frame, 0))
        self.assertEqual(data_frame.sum().sum(), 190)


class TestValidator(unittest.TestCase):

    def setUp(self):
        self.frame1 = pd.DataFrame([[0, 1, 2]] * 5,
                                   index=["A", "B", "C", "D", "E"],
                                   columns=["RED", "BLUE", "GREEN"])
        self.frame2 = pd.DataFrame(index=["A", "B", "C", "D", "E"],
                                   columns=["CYAN", "BLUE", "MAUVE"])
        self.frame3 = pd.DataFrame([["zero", 1, "two"]] * 5,
                                   index=["A", "B", "C", "E", "D"],
                                   columns=["RED", "BLUE", "GREEN"])

    def test_frame_alignment(self):
        self.assertTrue(check.dataframes_align([self.frame1, self.frame1, self.frame1]))
        self.assertTrue(check.dataframes_align([self.frame1, self.frame1, self.frame3], check_order=False))

        with self.assertRaises(ValueError):
            check.dataframes_align([self.frame1, self.frame2, self.frame1])

        with self.assertRaises(ValueError):
            check.dataframes_align([self.frame1, self.frame3, self.frame1])

        self.assertTrue(check.dataframes_align([None, None, None], allow_none=True))
        with self.assertRaises(ValueError):
            check.dataframes_align([None, None, None], allow_none=False)

        self.assertTrue(check.dataframes_align([self.frame1, self.frame1, None], allow_none=True))
        with self.assertRaises(ValueError):
            check.dataframes_align([self.frame1, self.frame1, None], allow_none=False)

    def test_index_alignment(self):
        index1 = self.frame1.index
        index2 = self.frame3.index
        index3 = self.frame1.columns

        self.assertTrue(check.indexes_align([index1, index1, index1]))
        self.assertTrue(check.indexes_align([index1, index1, index2], check_order=False))

        with self.assertRaises(ValueError):
            check.indexes_align([index1, index2, index1])

        with self.assertRaises(ValueError):
            check.indexes_align([index1, index3, index1])

        self.assertTrue(check.indexes_align([None, None, None], allow_none=True))
        with self.assertRaises(ValueError):
            check.indexes_align([None, None, None], allow_none=False)

        self.assertTrue(check.indexes_align([index1, index1, None], allow_none=True))
        with self.assertRaises(ValueError):
            check.indexes_align([index1, index1, None], allow_none=False)

    def test_frame_numeric(self):
        self.assertTrue(check.dataframe_is_numeric(None, allow_none=True))
        self.assertTrue(check.dataframe_is_numeric(self.frame1))

        with self.assertRaises(ValueError):
            check.dataframe_is_numeric(self.frame3)

    def test_frame_finite(self):
        self.assertTrue(check.dataframe_is_finite(None, allow_none=True))
        self.assertTrue(check.dataframe_is_finite(self.frame1))
        self.assertTrue(check.dataframe_is_finite(self.frame3))

        with self.assertRaises(ValueError):
            na_frame = self.frame1.copy()
            na_frame['RED'] = np.nan
            check.dataframe_is_finite(na_frame)

        with self.assertRaises(ValueError):
            inf_frame = self.frame1.copy()
            inf_frame['RED'] = np.inf
            check.dataframe_is_finite(inf_frame)

        with self.assertRaises(ValueError):
            na_frame = self.frame1.copy()
            na_frame.index = pd.Index([np.nan] + [na_frame.index[1:].tolist()])
            check.dataframe_is_finite(na_frame)

        with self.assertRaises(ValueError):
            na_frame = self.frame1.copy()
            na_frame.columns = pd.Index([np.nan] + [na_frame.columns[1:].tolist()])
            check.dataframe_is_finite(na_frame)

    def test_numeric(self):
        self.assertTrue(check.argument_numeric(0))
        self.assertTrue(check.argument_numeric(0.0))

        with self.assertRaises(ValueError):
            check.argument_numeric("0")

        self.assertTrue(check.argument_numeric(1, 0, 2))

        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_numeric(2, 0, 1))

        self.assertTrue(check.argument_numeric(None, allow_none=True))

    def test_type(self):
        self.assertTrue(check.argument_type(self, unittest.TestCase))
        self.assertTrue(check.argument_type(None, unittest.TestCase, allow_none=True))

        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_type("0", unittest.TestCase))

    def test_enum(self):
        self.assertTrue(check.argument_enum("A", ("A", "B")))
        self.assertTrue(check.argument_enum(["A", "B", "A"], ("A", "B")))

        with self.assertRaises(ValueError):
            check.argument_enum(["A", "B", "C"], ("A", "B"))

    def test_none(self):
        self.assertTrue(check.arguments_not_none(("A", "B")))
        self.assertTrue(check.arguments_not_none(("A", None), num_none=1))
        with self.assertRaises(ValueError):
            self.assertTrue(check.arguments_not_none((None, None, "A")))
        with self.assertRaises(ValueError):
            self.assertTrue(check.arguments_not_none((None, None, "A"), num_none=0))

    def test_path(self):
        temp_dir = tempfile.mkdtemp()
        temp_test = os.path.join(temp_dir, "test_path")
        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_path(temp_test, create_if_needed=False))
        with self.assertRaises(ValueError):
            self.assertTrue(check.argument_path(os.path.join("/dev/null", "super_test"), create_if_needed=True))
        self.assertTrue(check.argument_path(temp_test, create_if_needed=True))
        self.assertTrue(check.argument_path(temp_test, create_if_needed=False))
        self.assertTrue(check.argument_path(temp_test, create_if_needed=False, access=os.W_OK))
        self.assertTrue(check.argument_path(None, allow_none=True))
        shutil.rmtree(temp_dir)

    def test_is_subpath(self):
        temp_dir = tempfile.gettempdir()
        temp_test = os.path.join(temp_dir, "test_path")

        self.assertTrue(check.argument_subpath(temp_test, temp_dir))
        self.assertTrue(check.argument_subpath(temp_test, "/"))
        self.assertTrue(check.argument_subpath("~" + temp_test, "~"))
        self.assertTrue(check.argument_subpath(None, None, allow_none=True))

        with self.assertRaises(ValueError):
            check.argument_subpath(None, "~")
        with self.assertRaises(ValueError):
            check.argument_subpath("~", None)
        with self.assertRaises(ValueError):
            check.argument_subpath(temp_dir, temp_test)
        with self.assertRaises(ValueError):
            check.argument_subpath("..", ".")

    def test_callable(self):
        def callable_function(x):
            return x

        class NonCallableClass(object):
            pass

        self.assertTrue(check.argument_callable(None, allow_none=True))
        with self.assertRaises(ValueError):
            check.argument_callable(None, allow_none=False)

        self.assertTrue(check.argument_callable(int))
        self.assertTrue(check.argument_callable(callable_function))
        with self.assertRaises(ValueError):
            check.argument_callable(NonCallableClass())
        with self.assertRaises(ValueError):
            check.argument_callable(1)
        with self.assertRaises(ValueError):
            check.argument_callable("string")

    def test_subclass(self):

        class ClassA(object):
            pass

        class ClassB(object):
            pass

        class ClassC(ClassA):
            pass

        self.assertTrue(check.argument_is_subclass(None, ClassA, allow_none=True))
        with self.assertRaises(ValueError):
            check.argument_is_subclass(None, ClassA)

        self.assertTrue(check.argument_is_subclass(ClassC, ClassA))
        self.assertTrue(check.argument_is_subclass(ClassC(), ClassA))

        with self.assertRaises(ValueError):
            check.argument_is_subclass(ClassA, ClassC)

        with self.assertRaises(ValueError):
            check.argument_is_subclass(ClassC, ClassB)

        with self.assertRaises(ValueError):
            check.argument_is_subclass("arg", ClassA)

        with self.assertRaises(ValueError):
            check.argument_is_subclass(ClassC, "arg")


if __name__ == '__main__':
    unittest.main()
