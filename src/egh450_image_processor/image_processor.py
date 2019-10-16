#!/usr/bin/env python

import sys
import rospy
import cv2
import numpy as np
from std_msgs.msg import String
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import CameraInfo
import tf2_ros
from geometry_msgs.msg import TransformStamped

class ImageProcessor():
	def __init__(self):
		self.time_finished_processing = rospy.Time(0)
		# Get the path to the cascade XML training file using ROS parameters
		# Then load in our cascade classifier
		sign_cascade_file = str(rospy.get_param("~cascade_file"))
		self.sign_cascade = cv2.CascadeClassifier(sign_cascade_file)

		# Set up the CV Bridge
		self.bridge = CvBridge()

		# Load in parameters from ROS
		self.param_use_compressed = rospy.get_param("~use_compressed", False)
		self.param_triangle_radius = rospy.get_param("~triangle_radius", 1.0)
		self.param_square_radius = rospy.get_param("~square_radius", 1.0)

		# Orange Square
		self.param_hue_center2 = rospy.get_param("~hue_center", 30)
		self.param_hue_range2 = rospy.get_param("~hue_range", 60) / 2
		self.param_sat_min2 = rospy.get_param("~sat_min", 100)
		self.param_sat_max2 = rospy.get_param("~sat_max", 255)
		self.param_val_min2 = rospy.get_param("~val_min", 100)
		self.param_val_max2 = rospy.get_param("~val_max", 255)

		# Blue Triangle
		self.param_hue_center = rospy.get_param("~hue_center2", 170)
		self.param_hue_range = rospy.get_param("~hue_range2", 250) / 2
		self.param_sat_min = rospy.get_param("~sat_min2", 50)
		self.param_sat_max = rospy.get_param("~sat_max2", 255)
		self.param_val_min = rospy.get_param("~val_min2", 50)
		self.param_val_max = rospy.get_param("~val_max2", 255)

		# Set additional camera parameters
		self.got_camera_info = False
		self.camera_matrix = None
		self.dist_coeffs = None

		# Set up the publishers, subscribers, and tf2
		self.sub_info = rospy.Subscriber("~camera_info", CameraInfo, self.callback_info)

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

		# Generate the model for the pose solver
		# For this example, draw a square around where the circle should be
		# There are 5 points, one in the center, and one in each corner
		r = self.param_triangle_radius
		self.model_object = np.array([(0.0, 0.0, 0.0),
										(r, r, 0.0),
										(r, -r, 0.0),
										(-r, r, 0.0),
										(-r, -r, 0.0)])

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
		if msg_in.header.stamp > self.time_finished_processing:
			# Don't bother to process image if we don't have the camera calibration
			cv_image = None	
			self.model_image = None	
			success = False;
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

				# Image mask for colour filtering
				mask_image = self.process_image(cv_image)

			if cv_image is not None:
				# ===================
				# Do processing here!
				# ===================
				#gray = cv2.cvtColor(mask_image, cv2.COLOR_HSV2GRAY)
				gray = mask_image

				sign = self.sign_cascade.detectMultiScale(gray, 1.01, 1, minSize=(100,100))

				for (x,y,w,h) in sign:
					cv2.rectangle(cv_image,(x,y),(x+w,y+h),(255,0,0),2)
				# ===================

					self.model_image = np.array([
											(x, y),
											(x+w, y+h),
											(x+w, y-h),
											(x-w, y+h),
											(x-w, y-h)])

				# Do the SolvePnP method
				#(success, rvec, tvec) = cv2.solvePnP(self.model_object, self.model_image, self.camera_matrix, self.dist_coeffs)

				# If a result was found, send to TF2
				if success:
					msg_out = TransformStamped()
					msg_out.header = msg_in.header
					msg_out.child_frame_id = "triangle"
					msg_out.transform.translation.x = tvec[0]
					msg_out.transform.translation.y = tvec[1]
					msg_out.transform.translation.z = tvec[2]
					msg_out.transform.rotation.w = 1.0	# Could use rvec, but need to convert from DCM to quaternion first
					msg_out.transform.rotation.x = 0.0
					msg_out.transform.rotation.y = 0.0
					msg_out.transform.rotation.z = 0.0

					self.tfbr.sendTransform(msg_out)

				self.time_finished_processing = rospy.Time.now()
				# Convert CV image to ROS image and publish
				try:
					self.pub_overlay.publish( self.bridge.cv2_to_compressed_imgmsg( cv_image ) )
					self.pub_mask.publish( self.bridge.cv2_to_compressed_imgmsg( gray ) )
				except CvBridgeError as e:
					print(e)
		

	def process_image(self, cv_image):
		#Convert the image to HSV and prepare the mask
		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		mask_image = None

		hue_lower = (self.param_hue_center - self.param_hue_range) % 180
		hue_upper = (self.param_hue_center + self.param_hue_range) % 180

		thresh_lower = np.array([hue_lower, self.param_val_min, self.param_val_min])
		thresh_upper = np.array([hue_upper, self.param_val_max, self.param_val_max])


		if hue_lower > hue_upper:
			# We need to do a wrap around HSV 180 to 0 if the user wants to mask this color
			thresh_lower_wrap = np.array([180, self.param_sat_max, self.param_val_max])
			thresh_upper_wrap = np.array([0, self.param_sat_min, self.param_val_min])

			mask_lower = cv2.inRange(hsv_image, thresh_lower, thresh_lower_wrap)
			mask_upper = cv2.inRange(hsv_image, thresh_upper_wrap, thresh_upper)

			mask_image = cv2.bitwise_or(mask_lower, mask_upper)
		else:
			# Otherwise do a simple mask
			mask_image = cv2.inRange(hsv_image, thresh_lower, thresh_upper)

		# Refine image to get better results
		kernel = np.ones((5,5),np.uint8)
		mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

		return mask_image
