import cv2
# 测试索引0
cap0 = cv2.VideoCapture(0)
ret0, frame0 = cap0.read()
print(f"Cam 0: 采集成功={ret0}, 帧尺寸={frame0.shape if ret0 else 'None'}")
# 测试索引1
cap1 = cv2.VideoCapture(1)
ret1, frame1 = cap1.read()
print(f"Cam 1: 采集成功={ret1}, 帧尺寸={frame1.shape if ret1 else 'None'}")
cap0.release()
cap1.release()