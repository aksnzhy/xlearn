import unittest
import ctypes

class TestCAPI(unittest.TestCase):

	def test_all_c_api(self):
		xl = ctypes.c_void_p
		xl_handle = ctypes.byref(xl)
		myLib = cdll.LoadLibrary("./libxlearn.dylib")
		self.assertEqual(myLib.XLearnHello(), 0)
		self.assertEqual(myLib.XLearnCreate(c_str("linear"), xl_handle), 0)

if __name__ == '__main__':
	unittest.main()