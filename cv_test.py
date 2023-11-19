import unittest
from src import utils


class Message:
    # tx,y,z = position
    tx = 0
    ty = 0
    tz = 0

    # ox,y,z,w = orientation
    ox = 0
    oy = 0
    oz = 0
    ow = 0

    # acceleration and velocity statistics
    lin_a = 0
    ang_vx = 0
    ang_vy = 0
    ang_vz = 0

    objects = []

    def __init__(self, objects):
        # self.tx = tx
        # self.ty = ty
        # self.tz = tz
        # self.ox = ox
        # self.oy = oy
        # self.oz = oz
        # self.ow = ow
        # self.lin_a = lin_a
        # self.ang_vx = ang_vx
        # self.ang_vy = ang_vy
        # self.ang_vz = ang_vz
        self.objects = objects


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

    def __eq__(self, other):
        if isinstance(other, Object):
            return self.label == other.label and self.x == other.x and self.y == other.y and self.z == other.z and self.conf == other.conf
        return False

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

    # case 1a: we see a buoy in the same location with a different label
    # jolly
    def test_case1a(self):
        currObj = []
        currObj.append(Object("red-buoy", 1, 1, 1, 0.8))
        prevObj = []
        prevObj.append(Object("blue-buoy", 1, 1, 1, 0.85))
        curr = Message(currObj)
        prev = Message(prevObj)

        returnObj = utils.persistent_memory(curr, prev).objects

        assert len(returnObj) == 1

        assert returnObj[0].__eq__(Object("blue-buoy", 1, 1, 1, 0.85))
        assert returnObj[0].countDown == prevObj[0].countDown

    # case 1b: we see a buoy in the same location with the same label
    # kaitlyn

    def test_case1b(self):
        curr_objects = []
        curr_objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        curr_objects.append(Object("blue-buoy", 2, 3, 4, 0.8))

        prev_objects = []
        curr_objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        curr_objects.append(Object("blue-buoy", 2, 3, 4, 0.8))

        curr = Message(curr_objects)
        prev = Message(prev_objects)

        utils_curr = utils.persistent_memory(curr, prev).objects
        assert len(utils_curr) == 2
        assert utils_curr[0].label == "red-buoy"
        assert utils_curr[0].countDown == 10
        assert utils_curr[1].label == "blue-buoy"
        assert utils_curr[1].countDown == 10

    # case 2: buoy in previous frame is not seen again in current frame
    # alex

    def test_case2(self):
        objects1 = []
        objects1.append(Object("red-buoy", 1, 2, 3, 0.5))
        objects1.append(Object("red-buoy", 2, 3, 4, 0.8))
        objects2 = []
        # objects3 = []
        # objects3.append(Object("blue-buoy", 2, 3, 4, 0.8))
        # objects3.append(Object("black-buoy", 5, 5, 5, 0.8))
        curr = Message(objects2)
        prev = Message(objects1)
        returnmessage = utils.persistent_memory(curr, prev).objects

        assert returnmessage[0].__eq__(Object("red-buoy", 1, 2, 3, 0.5))
        assert returnmessage[1].__eq__(Object("red-buoy", 2, 3, 4, 0.8))

        assert len(returnmessage) == 2
        assert returnmessage[0].countDown == 9

       # assert returnObj[0].countDown = prevObj[0].countDown

    # test case for countdown @ 0

    def test_countDown0(self):

        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        objects[0].countDown = 0
        curr = Message([])
        prev = Message(objects)
        ret = utils.persistent_memory(curr, prev)
        assert len(ret.objects) == 0

    def test_repeatCountDown(self):
        objects = []
        objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        curr = Message([])
        prev = Message(objects)
        for i in range(0, 10):
            ret = utils.persistent_memory(curr, prev)
            if (i >= 9):
                assert len(ret.objects) == 0
            else:
                assert len(ret.objects) == 1
                assert ret.objects[0].countDown == 9-i
            prev = ret
            curr = Message([])

    # duplicates

    def duplicate_objects(self):
        curr_objects = []
        curr_objects.append(Object("red-buoy", 1, 2, 3, 0.5))
        curr_objects.append(Object("red-buoy", 1, 2, 3, 0.5))

        prev_objects = []
        curr_objects.append(Object("blue-buoy", 2, 3, 4, 0.5))
        curr_objects.append(Object("blue-buoy", 2, 3, 4, 0.8))

        curr = Message(curr_objects)
        prev = Message(prev_objects)

        return_objects = utils.persistent_memory(curr, prev).objects
        assert len(return_objects) == 2
        assert return_objects[0].label == "red-buoy"
        assert return_objects[0].countDown == 10
        assert return_objects[1].label == "blue-buoy"
        assert return_objects[1].countDown == 9


if __name__ == '__main__':
    unittest.main()
