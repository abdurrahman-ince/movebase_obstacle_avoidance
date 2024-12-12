#!/usr/bin/env python

import rospy
import cv2
import cv_bridge
import numpy as np
import actionlib
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal

class Follower:
    def __init__(self):
        self.bridge = cv_bridge.CvBridge()
        # Kamera görüntüleri için abone ol
        self.image_sub = rospy.Subscriber('/camera/rgb/image_raw', Image, self.image_callback)
        # LIDAR verisi için abone ol
        self.lidar_sub = rospy.Subscriber('/scan', LaserScan, self.lidar_callback)
        # Robotun hız komutlarını yayınla
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.twist = Twist()

        # MoveBase client setup
        self.client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
        self.client.wait_for_server()

        # Robotun hangi modda olduğunu belirlemek için
        self.tracking_green_line = True

        # LIDAR verisini saklamak için
        self.lidar_data = None

        # Engel algılandığında gidilecek konum (txt dosyasındaki son hedef konum)
        self.return_goal = (0.24591936188234353, 1.2705380476583898)  # Verdiğiniz X, Y konumu

        # Hedefe ulaşıldığında çizgi takibinin durması için kontrol
        self.goal_reached = False

    def lidar_callback(self, msg):
        """Engel tespiti için LIDAR verisini al"""
        self.lidar_data = msg.ranges

    def image_callback(self, msg):
        if self.tracking_green_line and not self.goal_reached:
            # ROS görüntüsünü OpenCV görüntüsüne çevir
            image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            # Renk algılaması için görüntüyü HSV renk uzayına çevir
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            # Yeşil rengin alt ve üst sınırlarını belirle
            lower_green = np.array([35, 50, 50])  # Yeşil rengin alt sınırı
            upper_green = np.array([85, 255, 255])  # Yeşil rengin üst sınırı

            # Yeşil bölgeleri maskele
            mask = cv2.inRange(hsv, lower_green, upper_green)

            # Yeşil çizgi olması gereken bölgeleri sınırlamak için ROI (Region of Interest) belirleyin
            h, w, d = image.shape
            search_top = int(3 * h / 4)
            search_bot = int(3 * h / 4 + 20)
            mask[0:search_top, 0:w] = 0
            mask[search_bot:h, 0:w] = 0

            # Maskedeki merkez hesaplamalarını yap
            M = cv2.moments(mask)

            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])

                # Robotun merkezine göre hata hesaplayın
                error = cx - w / 2

                # İleri doğru hareket için sabit hız
                self.twist.linear.x = 0.2

                # Hata ile dönüş hızını ayarlayın
                self.twist.angular.z = -float(error) / 700

                # Dönüş hızını sınırlayın
                if self.twist.angular.z > 0.5:
                    self.twist.angular.z = 0.5
                elif self.twist.angular.z < -0.5:
                    self.twist.angular.z = -0.5

                # Eğer hata büyükse, robotu yavaşlatın
                if abs(error) > 50:
                    self.twist.linear.x = 0.1
                    self.twist.angular.z = self.twist.angular.z * 1.5

                # Hız komutlarını yayınlayın
                self.cmd_vel_pub.publish(self.twist)

            else:
                # Eğer yeşil çizgi yoksa, hareketi durdur ve hedefe git
                self.tracking_green_line = False
                self.stop_robot()
                self.avoid_obstacle_and_move_to_goal()

            # Mask ve orijinal görüntüleri göster
            cv2.imshow("mask", mask)
            cv2.imshow("output", image)
            cv2.waitKey(3)
        else:
            # Eğer yeşil çizgi izlenmiyorsa veya hedefe ulaşıldıysa, robotu durdur ve hedefe git
            self.stop_robot()
            if not self.goal_reached:
                self.avoid_obstacle_and_move_to_goal()

    def stop_robot(self):
        """Robotu durdur"""
        self.twist.linear.x = 0.0
        self.twist.angular.z = 0.0
        self.cmd_vel_pub.publish(self.twist)

    def avoid_obstacle_and_move_to_goal(self):
        """Engelden kaç ve yeni hedefe git"""
        if self.lidar_data:
            # Engel tespiti için ön, sol ve sağ mesafeleri ölç
            front_dist = min(self.lidar_data[0:20] + self.lidar_data[-20:])
            left_dist = min(self.lidar_data[60:120])
            right_dist = min(self.lidar_data[240:300])

            rospy.loginfo(f"Front: {front_dist}, Left: {left_dist}, Right: {right_dist}")

            # Eğer önünde engel varsa, engelden kaçmak için yön değiştir
            if front_dist < 1.0:
                rospy.loginfo("Engel algılandı!")
                # Engel algılandığında terminale mesaj yazdır
                if left_dist > right_dist:
                    self.twist.angular.z = 0.5
                else:
                    self.twist.angular.z = -0.5
                self.twist.linear.x = 0.0
                self.cmd_vel_pub.publish(self.twist)
                rospy.sleep(2)  # Engelden kaçmak için dönüş yap

            # Engelden kaçtıktan sonra hedefe git
            self.send_goal(self.return_goal)

    def send_goal(self, goal):
        """Belirli bir hedefe gitmek için MoveBase komutunu gönder"""
        goal_x, goal_y = goal

        # Move Base hedefini oluşturun
        move_goal = MoveBaseGoal()
        move_goal.target_pose.header.frame_id = "map"
        move_goal.target_pose.header.stamp = rospy.Time.now()
        move_goal.target_pose.pose.position.x = goal_x
        move_goal.target_pose.pose.position.y = goal_y
        move_goal.target_pose.pose.orientation.w = 1.0

        rospy.loginfo(f"Yeni hedef gönderiliyor: {goal_x}, {goal_y}")
        self.client.send_goal(move_goal)
        self.client.wait_for_result()
        rospy.loginfo("Yeni hedefe ulaşıldı.")

        # Hedefe ulaştıktan sonra yeşil çizgi takibini tekrar aktif et ve durdur
        self.goal_reached = True  # Hedefe ulaşıldığında çizgi takibi duracak
        self.stop_robot()

if __name__ == '__main__':
    rospy.init_node('follower')  # ROS düğümünü başlat
    follower = Follower()  # Follower sınıfından bir nesne oluştur
    rospy.spin()  # Düğümü çalıştırmaya devam et

