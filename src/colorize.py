#!/usr/bin/env python2

import time
import numpy as np
import std_msgs.msg as std_msgs
import cv2
import struct
import ctypes
import rospkg
import rospy
import sensor_msgs.msg as sensor_msgs
from sensor_msgs.msg import Image, CameraInfo
from sensor_msgs import point_cloud2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
# import pcl
# import pcl_helper

def cameraInfo_callback(data):
    global sensortf
    sensortf.camera_info = data

def image_callback(data):
    global sensortf
    sensortf.img = sensortf.img2numpy(data)

def pcd_callback(data):
    print("PCD!!!")
    sensortf.pcd=data
    np_pcd = sensortf.pcdmsg2numpy(data)
    colored_pcd = sensortf.pcdPROJ2image(sensortf.img, np_pcd, None)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    # print(colored_pcd.shape)
    msmsg = sensortf.numpy2PCDmsg2(colored_pcd)

    sensortf.pub_points.publish(msmsg)

    # cloud = pcl_helper.ros_to_pcl(data)
    # print(cloud)

class SensorTF():
    def __init__(self):
        self.img_sub = rospy.Subscriber("/cv_camera/image_raw", Image, image_callback)
        self.pointcloud_sub = rospy.Subscriber("/velodyne_points", PointCloud2, pcd_callback)
        self.cameraInfo_sub = rospy.Subscriber("/cv_camera/camera_info", CameraInfo, cameraInfo_callback)

        self.pub_points = rospy.Publisher('/dragon_curve', sensor_msgs.PointCloud2, queue_size=1)
        self.bridge = CvBridge()
        self.pcd = PointCloud2
        self.camera_info = CameraInfo()
        self.img = np.zeros((1,1))
        self.camera_intrinsic = np.array([[615.67,0,328.0], [0, 615.96, 241.3], [0,0,1]])
        
    def img2numpy(self, img_msg):
        cv_img = self.bridge.imgmsg_to_cv2(img_msg, "bgr8")
        return cv_img
        # cv2.imshow("frame" , cv_img)
        # cv2.waitKey(1)
# [  0.0000000,  0.0000000, -1.0000000;
#   -1.0000000,  0.0000000, -0.0000000;
#    0.0000000,  1.0000000,  0.0000000 ]
    # pcd = [index, 7(xyzrgba)]
    def pcdPROJ2image(self, img, pcd, tf):
        pcd = np.transpose(pcd)
        R = [[ 0, -1,  0],
            [ 0,  0, -1],
            [ 1,  0,  0]]

        T = np.zeros(3)
        T[0] = -0.5
        T[1] = -0.2
        T[2] = 0.2



        newnew = []
        # all_rgb_coord = np.matmul(self.camera_intrinsic,np.matmul(R, pcd[0:3])+T)
        # print("ALLRGB {}".format(all_rgb_coord.shape))
        for idx in range(pcd.shape[1]):

            inst = pcd[0:3, idx]
            rgb_coord = np.matmul(self.camera_intrinsic,np.matmul(R, inst)+T)/pcd[0,idx]

            if 0 < rgb_coord[0] and rgb_coord[0] < img.shape[1] and 0< rgb_coord[1] and rgb_coord[1] < img.shape[0] and pcd[0,idx]>0:


                rgb = img[int(rgb_coord[1]), int(rgb_coord[0]),:]  
                ne_rgb = struct.unpack('I', struct.pack('BBBB', int(rgb[0]), int(rgb[1]), int(rgb[2]), 255))[0]
            else:
                ne_rgb = struct.unpack('I', struct.pack('BBBB', int(255), int(0), int(0), 255))[0]
                pcd[3:6,idx] = [255,0,0]
            
            new_inst = [pcd[0,idx], pcd[1,idx], pcd[2,idx], ne_rgb]
            newnew.append(new_inst)
                
                
                # pcd[3:6, idx]  = rgb
                # pcd[6, idx]  = 0
        pcd = np.transpose(pcd)
        return newnew
    
    def pcdmsg2numpy(self, pcd_msg):
        gen = pc2.read_points(pcd_msg, skip_nans=True)
        int_data = list(gen)
        xyz = np.array([[0,0,0]])
        rgb = np.array([[0,0,0]])
        for x in int_data:
            test = x[3] 
            # cast float32 to int so that bitwise operations are possible
            s = struct.pack('>f' ,test)
            i = struct.unpack('>l',s)[0]
            # you can get back the float value by the inverse operations
            pack = ctypes.c_uint32(i).value
            r = (pack & 0x00FF0000)>> 16
            g = (pack & 0x0000FF00)>> 8
            b = (pack & 0x000000FF)
            # prints r,g,b values in the 0-255 range
                        # x,y,z can be retrieved from the x[0],x[1],x[2]
            xyz = np.append(xyz,[[x[0],x[1],x[2]]], axis = 0)
            rgb = np.append(rgb,[[r,g,b]], axis = 0)
        np_pcd = np.concatenate((xyz, rgb, np.ones((rgb.shape[0], 1))), axis=1)
        if np_pcd.shape[0]>0:
            np_pcd = np_pcd[1:]

        msmsg = self.numpy2PCDmsg(np_pcd)
        return np_pcd
    def numpy2PCDmsg(self, np_pcd):
        ros_dtype = sensor_msgs.PointField.FLOAT32
        dtype = np.float32
        itemsize = np.dtype(dtype).itemsize

        data = np_pcd.astype(dtype).tobytes()

        fields = [sensor_msgs.PointField(
            name=n, offset=i*itemsize, datatype=ros_dtype, count=1)
            for i, n in enumerate('xyzrgba')]

        header = std_msgs.Header(frame_id="/velodyne", stamp=rospy.Time.now())

        return sensor_msgs.PointCloud2(
            header=header,
            height=1,
            width=np_pcd.shape[0],
            is_dense=False,
            is_bigendian=False,
            fields=fields,
            point_step=(itemsize * 7),
            row_step=(itemsize * 7 * np_pcd.shape[0]),
            data=data)
    def numpy2PCDmsg2(self, np_pcd):
        # points = []
        # lim = 8
        # for i in range(lim):
        #     for j in range(lim):
        #         for k in range(lim):
        #             x = float(i) / lim
        #             y = float(j) / lim
        #             z = float(k) / lim
        #             pt = [x, y, z, 0]
        #             r = int(x * 255.0)
        #             g = int(y * 255.0)
        #             b = int(z * 255.0)
        #             a = 255
        #             rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, a))[0]
        #             pt[3] = rgb
        #             points.append(pt)
        header = std_msgs.Header(frame_id="/velodyne", stamp=rospy.Time.now())
        fields = [PointField('x', 0, PointField.FLOAT32, 1),
          PointField('y', 4, PointField.FLOAT32, 1),
          PointField('z', 8, PointField.FLOAT32, 1),
          PointField('rgb', 16, PointField.UINT32, 1),
          ]
        pc2 = point_cloud2.create_cloud(header, fields, np_pcd)
        return pc2
def main():
    rospy.init_node("colorize")
    print("SIBAL~~~~~")
    global sensortf

    sensortf = SensorTF()
    while False:

        pcd = np.transpose(np.array([[0,0,1],[1,0,1], [1,1,1], [-1,0,1], [0,0,1], [1,0,0]]))
        R = np.zeros((3,3))
        # R[0,0] = R[1,1] = R[2,2] = 1
        R[0,0] = R[2,1] = 1
        R[1,2] = -1
        R1 = R
        R2 = [  [0.0000000, -1.0000000,  0.0000000],
[   1.0000000,  0.0000000,  0.0000000],
   [0.0000000,  0.0000000,  1.0000000 ]]
        R = np.matmul(R1,R2)
        print(R)
        # R[0,1] = R[1,2] = -1
        # R[2,0] = 1
#         R = [[  0.5000000, -0.7071068,  0.5000000],
#    [0.7071068,  0.0000000, -0.7071068],
#    [0.5000000,  0.7071068,  0.5000000 ]]
#         R = [[  1.0000000,  0.0000000,  0.0000000],
#    [0.0000000,  0.0000000, -1.0000000],
#    [0.0000000,  1.0000000,  0.0000000 ]]
#         R = [[  0.1971501, -0.8028499, -0.5626401],
#   -0.8028499,  0.1971501, -0.5626401],
#    0.5626401,  0.5626401, -0.6056999 ]]
        T = np.zeros(3)

        newnew = []
        temp=0
        for idx in range(pcd.shape[1]):

            inst = pcd[0:3, idx]
            rgb_coord = np.matmul(np.array([[615.67,0,328.0], [0, 615.96, 241.3], [0,0,1]]),np.matmul(R, inst)+T)/pcd[0,idx]
            # rgb_coord = (np.matmul(R, inst)+T)/pcd[2,idx]
            print("INST {}".format(inst))
            print("TFED {}".format(rgb_coord))
        print("----------------------------")
        return
    rospy.spin()

if __name__ == "__main__":
    main()
