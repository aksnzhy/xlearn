# coding: utf-8
import unittest
import ctypes
from ctypes import cdll, c_bool, c_int, c_float

XLearnHandle = ctypes.c_void_p

def c_str(string):
	return ctypes.c_char_p(string)

class TestCAPI(unittest.TestCase):

	def test_all_c_api(self):
		xl = XLearnHandle()
		xl_handle = ctypes.byref(xl)
		myLib = cdll.LoadLibrary("../../lib/libxlearn.dylib")
		self.assertEqual(myLib.XLearnHello(), 0)
		self.assertEqual(myLib.XLearnCreate(c_str("linear"), xl_handle), 0)
		self.assertEqual(myLib.XLearnSetTrain(xl_handle, c_str("./data.txt")), 0)
		self.assertEqual(myLib.XLearnSetTest(xl_handle, c_str("./data.txt")), 0)
		self.assertEqual(myLib.XLearnSetValidate(xl_handle, c_str("./data.txt")), 0)
		self.assertEqual(myLib.XLearnSetBool(xl_handle, c_str("on_disk"), c_bool(True)), 0)
		self.assertEqual(myLib.XLearnSetBool(xl_handle, c_str("quiet"), c_bool(True)), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("loss"), c_str("squared")), 0)
		self.assertEqual(myLib.XLearnSetInt(xl_handle, c_str("fold"), c_int(10)), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("opt"), c_str("ftrl")), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("alpha"), c_float(1)), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("beta"), c_float(1)), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("lambda_1"), c_float(1)), 0)
		self.assertEqual(myLib.XLearnSetStr(xl_handle, c_str("lambda_2"), c_float(1)), 0)

if __name__ == '__main__':
	unittest.main()