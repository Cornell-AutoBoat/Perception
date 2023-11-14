CLOSENESS_RADIUS = 0.05  # in meters


def persistent_memory(curr, prev):

    prev.objects = removeDuplicates(prev.objects)
    curr.objects = removeDuplicates(curr.objects)

    return_list = []

    # list to keep track of if buoy in previous frame is seen in new frame
    # flags for if we've seen it before
    prev_seen = [False] * len(prev.objects)
    curr_seen = [False] * len(prev.objects)  # flags for if we've seen before

    for curr_index in range(len(curr.objects)):
        for prev_index in range(len(prev.objects)):
            curr_obj = curr.objects[curr_index]
            prev_obj = prev.objects[prev_index]

            # case 1: same location buoy
            if isNear(curr_obj, prev_obj, curr, prev, CLOSENESS_RADIUS):

                # case 1a: seen before (bad heuristic) //to prevent double counting
                if prev_seen[prev_index]:
                    curr_seen[curr_index] = True

                # case 1b: different label

                elif curr_obj.label != prev_obj.label:
                    # if confidence of previous is higher, add to current and decrement
                    # countDown and remove current
                    if prev_obj.conf > curr_obj.conf and prev_obj.countDown > 1:
                        if prev_seen[prev_index]:
                            prev_obj.countDown -= 1
                        curr.objects[curr_index] = prev_obj  # replace m with p
                    prev_seen[prev_index] = True

                # case 1c: same label
                else:
                    # print(curr.objects[curr_index].label + "seen " +
                    #       str(curr.objects[curr_index].countDown))
                    prev_seen[prev_index] = True

                # break out if seen before so as not to double decrement
                if prev_seen[prev_index]:
                    break

        if not curr_seen[curr_index]:
            return_list.append(curr[curr_index])

    # case 2: buoy in previous frame is not seen again in current frame
    # decrement countDown
    for index, used in enumerate(prev_seen):
        if not used:
            prev.objects[index].countDown -= 1
            if (prev.objects[index].countDown > 0):
                return_list.objects.append(prev.objects[index])

    print(len(return_list.objects))

    # case 3 is covered already (when adding objects seen in current frame to m)
    return return_list


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
