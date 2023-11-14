import unittest
from src import utils


class Object:
    label = ""
    x = 0
    y = 0
    z = 0
    countDown = 10
    conf = 0.5

    def __init__(self, label, x, y, z, conf):
        self.label = label
        self.x = x
        self.y = y
        self.z = z
        self.countDown = Object.countDown
        self.conf = conf

# tests have to start with "test"


class RemoveDuplicates(unittest.TestCase):

    def test1(self):
        objects = []
        objects.append(Object("red-buoy", 1, 1, 1, 0.9))
        objects.append(Object("blue-buoy", 1, 1, 1, 0.8))
        assert len(objects) == 2
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 1
        assert objects[0].label == "red-buoy"

    def test2(self):
        objects = []
        objects.append(Object("red-buoy", 1, 1, 1, 0.8))
        objects.append(Object("blue-buoy", 1, 1, 1, 0.9))
        assert len(objects) == 2
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 1
        assert objects[0].label == "blue-buoy"

    def testTriple(self):
        objects = []
        objects.append(Object("red-buoy", 1, 1, 1, 0.8))
        objects.append(Object("blue-buoy", 1, 1, 1, 0.85))
        objects.append(Object("green-buoy", 1, 1, 1, 0.75))
        assert len(objects) == 3
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 1
        assert objects[0].label == "blue-buoy"

    def testTripleSameConf(self):
        objects = []
        objects.append(Object("red-buoy", 1, 1, 1, 0.8))
        objects.append(Object("blue-buoy", 1, 1, 1, 0.8))
        objects.append(Object("green-buoy", 1, 1, 1, 0.8))
        assert len(objects) == 3
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 1
        assert objects[0].label == "red-buoy"

    def testNoDuplicates(self):
        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.8))
        objects.append(Object("blue-buoy", 0, 0, 1, 0.8))
        objects.append(Object("green-buoy", -1, -1, -1, 0.8))
        assert len(objects) == 3
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 3

    def testRemoveOne(self):
        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        objects.append(Object("blue-buoy", 0, 0, 1, 0.8))
        objects.append(Object("green-buoy", 0.99, 2.01, 3, 0.9))
        assert len(objects) == 3
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 2
        assert objects[0].label == "green-buoy"
        assert objects[1].label == "blue-buoy"

    def testNoRemove(self):
        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        objects.append(Object("red-buoy", 2, 3, 4, 0.8))
        assert len(objects) == 2
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 2


class PersistentMemory(unittest.TestCase):
    def testTry(self):
        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.8))
        objects.append(Object("blue-buoy", 0, 0, 1, 0.8))
        objects.append(Object("green-buoy", 0.99, 2.01, 3, 0.9))
        objects = utils.removeDuplicates(objects)
        assert len(objects) == 2


if __name__ == '__main__':
    unittest.main()
