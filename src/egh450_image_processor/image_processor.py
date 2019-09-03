#!/usr/bin/env python

import sys
import rospy
import cv2
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
import tf2_ros

class ImageProcessor():
	def __init__(self):
		# Get the path to the cascade XML training file using ROS parameters
		# Then load in our cascade classifier
		sign_cascade_file = str(rospy.get_param("~cascade_file"))
		self.sign_cascade = cv2.CascadeClassifier(sign_cascade_file)

		# Set up the CV Bridge
		self.bridge = CvBridge()

		# Load in parameters from ROS
		self.param_use_compressed = rospy.get_param("~use_compressed", False)
		self.param_circle_radius = rospy.get_param("~circle_radius", 1.0)
		self.param_hue_center = rospy.get_param("~hue_center", 170)
		self.param_hue_range = rospy.get_param("~hue_range", 20) / 2
		self.param_sat_min = rospy.get_param("~sat_min", 50)
		self.param_sat_max = rospy.get_param("~sat_max", 255)
		self.param_val_min = rospy.get_param("~val_min", 50)
		self.param_val_max = rospy.get_param("~val_max", 255)

		# Set additional camera parameters
		self.got_camera_info = True
		self.camera_matrix = None
		self.dist_coeffs = None

		# Set up the publishers, subscribers, and tf2
		#self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)

		if self.param_use_compressed:
			self.sub_img = rospy.Subscriber("~image_raw/compressed", CompressedImage, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image/compressed", CompressedImage, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image/compressed", CompressedImage, queue_size=1)
			#self.pub_img = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)
		else:
			self.sub_img = rospy.Subscriber("~image_raw", Image, self.callback_img)
			self.pub_mask = rospy.Publisher("~debug/image_raw", Image, queue_size=1)
			self.pub_overlay = rospy.Publisher("~overlay/image_raw", Image, queue_size=1)

		self.tfbr = tf2_ros.TransformBroadcaster()

	def shutdown(self):
		# Unregister anything that needs it here
		self.sub_img.unregister()
		self.sub_info.unregister()

	# Collect in the camera characteristics
	def callback_info(self, msg_in):
		self.dist_coeffs = np.array([[msg_in.D[0], msg_in.D[1], msg_in.D[2], msg_in.D[3], msg_in.D[4]]], dtype="double")

		self.camera_matrix = np.array([
                 (msg_in.P[0], msg_in.P[1], msg_in.P[2]),
                 (msg_in.P[4], msg_in.P[5], msg_in.P[6]),
                 (msg_in.P[8], msg_in.P[9], msg_in.P[10])],
				 dtype="double")

		if not self.got_camera_info:
			rospy.loginfo("Got camera info")
			self.got_camera_info = True

	def callback_img(self, msg_in):
		# Don't bother to process image if we don't have the camera calibration
		cv_image = None		
		if self.got_camera_info:
			#Convert ROS image to CV image
			try:
				if self.param_use_compressed:
					cv_image = self.bridge.compressed_imgmsg_to_cv2( msg_in, "bgr8" )
				else:
					cv_image = self.bridge.imgmsg_to_cv2( msg_in, "bgr8" )
			except CvBridgeError as e:
				rospy.loginfo(e)
				return

		if cv_image is not None:
			# ===================
			# Do processing here!
			# ===================
			gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

			sign = self.sign_cascade.detectMultiScale(gray, 1.01, 1, minSize=(100,100))

			for (x,y,w,h) in sign:
				cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,0,0),2)
			# ===================

			# Convert CV image to ROS image and publish
			try:
				self.pub_overlay.publish( self.bridge.cv2_to_compressed_imgmsg( cv_image ) )
				self.pub_mask.publish( self.bridge.cv2_to_compressed_imgmsg( gray ) )
			except CvBridgeError as e:
				print(e)



















