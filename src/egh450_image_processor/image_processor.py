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
		sign_cascade_file_triangle = str(rospy.get_param("~cascade_file_triangle"))
		self.sign_cascade_triangle = cv2.CascadeClassifier(sign_cascade_file_triangle)

		sign_cascade_file_square = str(rospy.get_param("~cascade_file_square"))
		self.sign_cascade_square = cv2.CascadeClassifier(sign_cascade_file_square)

		# Set up the CV Bridge
		self.bridge = CvBridge()

		# Load in parameters from ROS
		self.param_use_compressed = rospy.get_param("~use_compressed", False)
		self.param_triangle_radius = rospy.get_param("~triangle_radius", 1.0)
		self.param_square_radius = rospy.get_param("~square_radius", 1.0)

		# Orange Square
		self.param_hue_center_square = rospy.get_param("~hue_center_square", 30)
		self.param_hue_range_square = rospy.get_param("~hue_range_square", 90) / 2
		self.param_sat_min_square = rospy.get_param("~sat_min_square", 75)
		self.param_sat_max_square = rospy.get_param("~sat_max_square", 255)
		self.param_val_min_square = rospy.get_param("~val_min_square", 100)
		self.param_val_max_square = rospy.get_param("~val_max_square", 255)

		# Blue Triangle
		self.param_hue_center_triangle = rospy.get_param("~hue_center_triangle", 170)
		self.param_hue_range_triangle = rospy.get_param("~hue_range_triangle", 250) / 2
		self.param_sat_min_triangle = rospy.get_param("~sat_min_triangle", 50)
		self.param_sat_max_triangle = rospy.get_param("~sat_max_triangle", 255)
		self.param_val_min_triangle = rospy.get_param("~val_min_triangle", 50)
		self.param_val_max_triangle = rospy.get_param("~val_max_triangle", 255)


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
		r = self.param_square_radius
		r2 = self.param_triangle_radius
		self.model_object_square = np.array([(0.0, 0.0, 0.0),
					      (r, r, 0.0),
					      (r, -r, 0.0),
					      (-r, r, 0.0),
					      (-r, -r, 0.0)], dtype=np.float64)

		self.model_object_triangle = np.array([(0.0, 0.0, 0.0),
					      (r2, r2, 0.0),
					      (r2, -r2, 0.0),
					      (-r2, r2, 0.0),
					      (-r2, -r2, 0.0)], dtype=np.float64)

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
			self.model_image_square = None	
			self.model_image_triangle = None	
			success_square = False;
			success_triangle = False;
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
				mask_image_square = self.process_image(cv_image, 1)
				#mask_image_triangle = self.process_image(cv_image, 2)

			if cv_image is not None:
				# ===================
				# Do processing here!
				# ===================
				#gray = cv2.cvtColor(mask_image, cv2.COLOR_HSV2GRAY)

				sign_square = self.sign_cascade_square.detectMultiScale(mask_image_square, 1.01, 1, minSize=(25,25))

				for (x,y,w,h) in sign_square:
					cv2.rectangle(cv_image,(x,y),(x+w,y+h),(0,165,255),2)
					self.model_image_square = np.array([
											(x, y),
											(x+w, y+h),
											(x+w, y-h),
											(x-w, y+h),
											(x-w, y-h)], dtype=np.float64)

				# Do the SolvePnP method
				if self.model_image_square is not None:
					(success_square, rvec_square, tvec_square) = cv2.solvePnP(self.model_object_square, self.model_image_square, self.camera_matrix, self.dist_coeffs)
					#print('x transform [cm]')
					#print(tvec[0]*100)
					#print('\n')
					#print('y transform [cm]')
					#print(tvec[1]*100)
					#print('\n')
					#print('z transform [cm]')
					#print(tvec[2]*100)
					#print('\n')

				# If a result was found, send to TF2

				if success_square:
					msg_out = TransformStamped()
					msg_out.header = msg_in.header
					msg_out.child_frame_id = "Square"
					msg_out.transform.translation.x = tvec_square[0]
					msg_out.transform.translation.y = tvec_square[1]
					msg_out.transform.translation.z = tvec_square[2]
					msg_out.transform.rotation.w = 1.0	# Could use rvec, but need to convert from DCM to quaternion first
					msg_out.transform.rotation.x = 0.0
					msg_out.transform.rotation.y = 0.0
					msg_out.transform.rotation.z = 0.0

					self.tfbr.sendTransform(msg_out)

				self.time_finished_processing = rospy.Time.now()
				# Convert CV image to ROS image and publish
				try:
					self.pub_overlay.publish( self.bridge.cv2_to_compressed_imgmsg( cv_image ) )
					self.pub_mask.publish( self.bridge.cv2_to_compressed_imgmsg( mask_image_square ) )
				except CvBridgeError as e:
					print(e)
		

	def process_image(self, cv_image, process_type):
		#Convert the image to HSV and prepare the mask
		hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
		mask_image = None

		if process_type == 1: #Square

			hue_lower_square = (self.param_hue_center_square - self.param_hue_range_square) % 180
			hue_upper_square = (self.param_hue_center_square + self.param_hue_range_square) % 180

			thresh_lower_square = np.array([hue_lower_square, self.param_val_min_square, self.param_val_min_square])
			thresh_upper_square = np.array([hue_upper_square, self.param_val_max_square, self.param_val_max_square])


			if hue_lower_square > hue_upper_square:
				# We need to do a wrap around HSV 180 to 0 if the user wants to mask this color
				thresh_lower_wrap_square = np.array([180, self.param_sat_max_square, self.param_val_max_square])
				thresh_upper_wrap_square = np.array([0, self.param_sat_min_square, self.param_val_min_square])

				mask_lower_square = cv2.inRange(hsv_image, thresh_lower_square, thresh_lower_wrap_square)
				mask_upper_square = cv2.inRange(hsv_image, thresh_upper_wrap_square, thresh_upper_square)

				mask_image = cv2.bitwise_or(mask_lower_square, mask_upper_square)
			else:
				# Otherwise do a simple mask
				mask_image = cv2.inRange(hsv_image, thresh_lower_square, thresh_upper_square)

		# Refine image to get better results
		kernel = np.ones((5,5),np.uint8)
		mask_image = cv2.morphologyEx(mask_image, cv2.MORPH_OPEN, kernel)

		return mask_image
