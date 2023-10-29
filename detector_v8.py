#!/usr/bin/env python3

from test.msg import ZEDdata
from test.msg import Object
import cv_viewer.tracking_viewer as cv_viewer
import ogl_viewer.viewer as gl
from time import sleep
from threading import Lock, Thread
import sys
import numpy as np

import argparse
import torch
import pyzed.sl as sl
import cv2
from ultralytics import YOLO

import yaml
import rospy
import rospkg

rospack = rospkg.RosPack()

sys.path.insert(0, rospack.get_path('test') + '/src/custom-CV-node/yolov5')
sys.path.insert(0, rospack.get_path('test') + '/src/custom-CV-node')

# global variables
IMG_SIZE = 416  # in pixels
CONF_THRESH = 0.3  # proportion
# filename of model, should be in weights directory
FILENAME = 'yolov8-10-22-2023.pt'
COUNTDOWN_DEFAULT = 10

prev = None


class TimestampHandler:
    def __init__(self):
        self.t_imu = sl.Timestamp()
        self.t_baro = sl.Timestamp()
        self.t_mag = sl.Timestamp()

    ##
    # check if the new timestamp is higher than the reference one, and if yes, save the current as reference
    def is_new(self, sensor):
        if (isinstance(sensor, sl.IMUData)):
            new_ = (sensor.timestamp.get_microseconds()
                    > self.t_imu.get_microseconds())
            if new_:
                self.t_imu = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.MagnetometerData)):
            new_ = (sensor.timestamp.get_microseconds()
                    > self.t_mag.get_microseconds())
            if new_:
                self.t_mag = sensor.timestamp
            return new_
        elif (isinstance(sensor, sl.BarometerData)):
            new_ = (sensor.timestamp.get_microseconds()
                    > self.t_baro.get_microseconds())
            if new_:
                self.t_baro = sensor.timestamp
            return new_


with open(sys.path[0] + '/data_v8.yaml', 'r') as file:
    object_map = yaml.safe_load(file)

lock = Lock()
run_signal = False
exit_signal = False


def xywh2abcd(xywh, im_shape):
    output = np.zeros((4, 2))

    # Center / Width / Height -> BBox corners coordinates
    x_min = (xywh[0] - 0.5*xywh[2])  # * im_shape[1]
    x_max = (xywh[0] + 0.5*xywh[2])  # * im_shape[1]
    y_min = (xywh[1] - 0.5*xywh[3])  # * im_shape[0]
    y_max = (xywh[1] + 0.5*xywh[3])  # * im_shape[0]

    # A ------ B
    # | Object |
    # D ------ C

    output[0][0] = x_min
    output[0][1] = y_min

    output[1][0] = x_max
    output[1][1] = y_min

    output[2][0] = x_max
    output[2][1] = y_max

    output[3][0] = x_min
    output[3][1] = y_max
    return output


def detections_to_custom_box(detections, im, im0):
    output = []
    for i, det in enumerate(detections):
        xywh = det.xywh[0]

        # Creating ingestable objects for the ZED SDK
        obj = sl.CustomBoxObjectData()
        obj.bounding_box_2d = xywh2abcd(xywh, im0.shape)
        obj.label = det.cls
        obj.probability = det.conf
        obj.is_grounded = False
        output.append(obj)
    return output


def torch_thread(weights, img_size, conf_thres=0.2, iou_thres=0.45):
    global image_net, exit_signal, run_signal, detections

    print("Intializing Network...")

    model = YOLO(weights)

    while not exit_signal:
        if run_signal:
            lock.acquire()

            img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2RGB)
            # https://docs.ultralytics.com/modes/predict/#video-suffixes
            det = model.predict(img, save=False, imgsz=img_size, conf=conf_thres, iou=iou_thres)[
                0].cpu().numpy().boxes

            # ZED CustomBox format (with inverse letterboxing tf applied)
            detections = detections_to_custom_box(det, image_net)
            lock.release()
            run_signal = False
        sleep(0.01)


def main():
    pub = rospy.Publisher('chatter', ZEDdata, queue_size=10)
    rospy.init_node('talker', anonymous=True)
    global image_net, exit_signal, run_signal, detections

    capture_thread = Thread(target=torch_thread,
                            kwargs={'weights': sys.path[0] + '/weights/' + FILENAME, 'img_size': IMG_SIZE, "conf_thres": CONF_THRESH})
    capture_thread.start()

    print("Initializing Camera...")

    zed = sl.Camera()

    input_type = sl.InputType()
    # if opt.svo is not None:
    #     input_type.set_from_svo_file(opt.svo)

    # Create a InitParameters object and set configuration parameters
    init_params = sl.InitParameters(
        input_t=input_type, svo_real_time_mode=True)
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.METER
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # QUALITY
    init_params.coordinate_system = sl.COORDINATE_SYSTEM.LEFT_HANDED_Y_UP
    init_params.depth_maximum_distance = 40

    runtime_params = sl.RuntimeParameters()
    status = zed.open(init_params)

    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    image_left_tmp = sl.Mat()

    print("Initialized Camera")

    positional_tracking_parameters = sl.PositionalTrackingParameters()
    # If the camera is static, uncomment the following line to have better performances and boxes sticked to the ground.
    # positional_tracking_parameters.set_as_static = True
    zed.enable_positional_tracking(positional_tracking_parameters)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.detection_model = sl.DETECTION_MODEL.CUSTOM_BOX_OBJECTS
    obj_param.enable_tracking = True
    zed.enable_object_detection(obj_param)

    objects = sl.Objects()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()

    # Display
    camera_infos = zed.get_camera_information()
    # Create OpenGL viewer
    # viewer = gl.GLViewer()
    # point_cloud_res = sl.Resolution(min(camera_infos.camera_resolution.width, 720),
    #                                 min(camera_infos.camera_resolution.height, 404))
    # point_cloud_render = sl.Mat()
    # viewer.init(camera_infos.camera_model, point_cloud_res, obj_param.enable_tracking)
    # point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)
    image_left = sl.Mat()
    # Utilities for 2D display
    display_resolution = sl.Resolution(min(camera_infos.camera_resolution.width, 1280),
                                       min(camera_infos.camera_resolution.height, 720))
    # image_scale = [display_resolution.width / camera_infos.camera_resolution.width, display_resolution.height / camera_infos.camera_resolution.height]
    # image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255], np.uint8)

    # Utilities for tracks view
    camera_config = zed.get_camera_information().camera_configuration
    tracks_resolution = sl.Resolution(400, display_resolution.height)
    track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.camera_fps,
                                                    init_params.depth_maximum_distance)
    track_view_generator.set_camera_calibration(
        camera_config.calibration_parameters)
    image_track_ocv = np.zeros(
        (tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
    # Camera pose
    cam_w_pose = sl.Pose()

    sensors_data = sl.SensorsData()
    ts_handler = TimestampHandler()

    while not rospy.is_shutdown():  # viewer.is_available() and not exit_signal
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            # -- Get the image
            lock.acquire()
            zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
            image_net = image_left_tmp.get_data()
            lock.release()
            run_signal = True

            # -- Detection running on the other thread
            while run_signal:
                sleep(0.001)

            # Wait for detections
            lock.acquire()
            # -- Ingest detections
            zed.ingest_custom_box_objects(detections)
            lock.release()
            zed.retrieve_objects(objects, obj_runtime_param)

            msg = ZEDdata()
            py_translation = sl.Translation()
            msg.tx = round(cam_w_pose.get_translation(
                py_translation).get()[0], 3)
            msg.ty = round(cam_w_pose.get_translation(
                py_translation).get()[1], 3)
            msg.tz = round(cam_w_pose.get_translation(
                py_translation).get()[2], 3)
            # print("Translation: tx: {0}, ty:  {1}, tz:  {2}, timestamp: {3}\n".format(tx, ty, tz, cam_w_pose.timestamp))
            # Display orientation quaternion
            py_orientation = sl.Orientation()
            msg.ox = round(cam_w_pose.get_orientation(
                py_orientation).get()[0], 3)
            msg.oy = round(cam_w_pose.get_orientation(
                py_orientation).get()[1], 3)
            msg.oz = round(cam_w_pose.get_orientation(
                py_orientation).get()[2], 3)
            msg.ow = round(cam_w_pose.get_orientation(
                py_orientation).get()[3], 3)

            if zed.get_sensors_data(sensors_data, sl.TIME_REFERENCE.CURRENT) == sl.ERROR_CODE.SUCCESS:
                if ts_handler.is_new(sensors_data.get_imu_data()):
                    # Get linear and angular velocity
                    msg.lin_a = sensors_data.get_imu_data().get_linear_acceleration()[
                        2]
                    msg.ang_vx = sensors_data.get_imu_data().get_angular_velocity()[
                        0] * np.pi / 180
                    msg.ang_vy = sensors_data.get_imu_data().get_angular_velocity()[
                        1] * np.pi / 180
                    msg.ang_vz = sensors_data.get_imu_data().get_angular_velocity()[
                        2] * np.pi / 180
                    angular_velocity = sensors_data.get_imu_data().get_angular_velocity()
                    # print(" \t Angular Velocities: [ {0} {1} {2} ] [deg/sec]".format(angular_velocity[0], angular_velocity[1], angular_velocity[2]))
            for object in objects.object_list:
                label = object_map['names'][object.raw_label]
                # Display translation and timestamp
                obj = Object()
                obj.label = label
                obj.x = object.position[0]
                obj.y = object.position[1]
                obj.z = object.position[2]
                obj.countDown = COUNTDOWN_DEFAULT
                obj.conf = object.confidence
                msg.objects.append(obj)

            if prev != None:
                msg = persistent_memory(msg, prev)
            prev = msg

            pub.publish(msg)
            # print("{} {} {}".format(object.id, object.position, label))

            # -- Display
            # Retrieve display data
            # zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)
            # point_cloud.copy_to(point_cloud_render)
            zed.retrieve_image(image_left, sl.VIEW.LEFT,
                               sl.MEM.CPU, display_resolution)
            zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

            # 3D rendering
            # viewer.updateData(point_cloud_render, objects)
            # # 2D rendering
            # np.copyto(image_left_ocv, image_left.get_data())
            # cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
            # global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
            # Tracking view
            track_view_generator.generate_view(
                objects, cam_w_pose, image_track_ocv, objects.is_tracked)

            # cv2.imshow("ZED | 2D View and Birds View", global_image)
            key = cv2.waitKey(10)
            if key == 27:
                exit_signal = True
        else:
            exit_signal = True

    # viewer.exit()
    exit_signal = True
    zed.close()


# persistent memory skeleton code
def persistent_memory(m, pm):

    # list to keep track of if buoy in previous frame is seen in new frame
    pm_seen = [False] * pm.length

    for m_index in range(len(m.objects)):
        for pm_index in range(len(pm.objects)):
            m_obj = m.objects[m_index]
            pm_obj = pm.objects[pm_index]

            # case 4: same location with different label
            if m_obj.x == pm_obj.x and m_obj.y == pm_obj.y:
                if m_obj.label != pm_obj.label:
                    # if confidence of previous is higher, add to current and decrement
                    # countDown and remove current
                    if pm_obj.conf > m.objects[m_index].conf:
                        pm_obj.countDown -= 1
                        m[m_index] = pm_obj  # replace m with p
                        pm_seen[pm_index] = True

            # case 1 buoy in previous frame is seen again in current frame
            # use the speed and multiply by 1/15 (around 15 frames per second)
            # to check that the buoy is the same
            elif m_obj.x == pm_obj.x + m.tx / 15 and m_obj.y == pm_obj.y + m.ty / 15:
                if m.objects[index].label == pm_obj.label:
                    pm_seen[pm_index] = True

    # case 2: buoy in previous frame is not seen again in current frame
    # decrement countDown
    for index, used in enumerate(pm_seen):
        if used == False:
            pm.objects[index].countDown -= 1
            m.objects.append(pm.objects[index])

    # case 3 is covered already (when adding objects seen in current frame to m)
    return m


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--weights', nargs='+', type=str, default='./weights/best.pt', help='model.pt path(s)')
    # parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    # parser.add_argument('--img_size', type=int, default=416, help='inference size (pixels)')
    # parser.add_argument('--conf_thres', type=float, default=0.4, help='object confidence threshold')
    # opt = parser.parse_args()

    with torch.no_grad():
        main()
