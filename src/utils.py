CLOSENESS_RADIUS = 1.0  # in meters (larger than necessary)


def persistent_memory(curr, prev):

    # case 0: cleaning up - two or more buoys in the same location

    # prev.objects = removeDuplicates(prev.objects)
    # curr.objects = removeDuplicates(curr.objects)

    prev.objects = removeSameDuplicates(prev.objects)
    curr.objects = removeSameDuplicates(curr.objects)

    # list to keep track of if buoy in previous frame is seen in new frame
    # flags for if we've seen it before
    prev_seen = [False] * len(prev.objects)

    for curr_index in range(len(curr.objects)):
        for prev_index in range(len(prev.objects)):
            curr_obj = curr.objects[curr_index]
            prev_obj = prev.objects[prev_index]

            # case 1: same location buoy
            if isNear(curr_obj, prev_obj, curr, prev, CLOSENESS_RADIUS):

                prev_seen[prev_index] = True

                # case 1a: different label

                if curr_obj.label != prev_obj.label:
                    # if confidence of previous is higher, add to current and decrement
                    # countDown and remove current
                    if prev_obj.conf > curr_obj.conf and prev_obj.countDown > 1:

                        # if prev_seen[prev_index]:
                        prev_obj.countDown = curr_obj.countDown
                        curr.objects[curr_index] = prev_obj  # replace m with p

                # case 1b: same label
                # do nothing

    # case 2: buoy in previous frame is not seen again in current frame
    # decrement countDown
    for index, used in enumerate(prev_seen):
        if not used:
            prev.objects[index].countDown -= 1
            if (prev.objects[index].countDown > 0):
                curr.objects.append(prev.objects[index])

    # print(len(curr.objects))

    # remove duplicates again (just in case we double added)
    curr.objects = removeSameDuplicates(curr.objects)

    return curr


def isNear(o1, o2, curr, prev, radius):
    # o1 is object 1
    # o2 is object 2
    # both objects have position
    # currently we just check x,y to be within a radius,
    # we can check z coord and also do math to calculate different orientation/velocity
    return abs(o1.x-o2.x) < radius and abs(o1.y-o2.y) < radius


# returns a list of buoys with no two buoys in the same location
def removeDuplicates(object_list, curr=None, prev=None):
    return_list = []
    for obj in object_list:
        seen = False
        for i in range(0, len(return_list)):
            obj2 = return_list[i]
            if isNear(obj, obj2, None, None, CLOSENESS_RADIUS):
                seen = True
                if (obj.conf > obj2.conf):
                    return_list[i] = obj
        if not seen:
            return_list.append(obj)

    return return_list


# returns a list of buoys with no two buoys in the same location
def removeSameDuplicates(object_list, curr=None, prev=None):
    return_list = []
    for obj in object_list:
        seen = False
        for i in range(0, len(return_list)):
            obj2 = return_list[i]
            # compared to removeDuplicates, this checks for same label as well
            if isNear(obj, obj2, None, None, CLOSENESS_RADIUS) and (obj2.label == obj.label):
                seen = True
                if (obj.conf > obj2.conf):
                    return_list[i] = obj
        if not seen:
            return_list.append(obj)

    return return_list
