from .DianaApi import *
import numpy as np
JOINT_NUM = 7

class DianaRobot:
    def __init__(self, ip='192.168.1.10'):
        # Modify the port numbers according to your actual configuration
        self.ip = ip
        self.srv_net_st = [ip, 5001, 5002, 5003, 5004, 5005]
        setDefaultToolTcpCoordinate('pika', ipAddress=self.ip)
        initSrv(self.srv_net_st)
        releaseBrake(ip)
        if ip == '192.168.10.75':
            self.movej_joint(np.array([-45, -60, 0, 150, 0, 0, -45])*np.pi/180)
        elif ip == '192.168.10.76':
            self.movej_joint(np.array([-45, -30, 0, 120, 0, -60, 45])*np.pi/180)
        self.wait_move()
        ret = changeControlMode(mode_e.T_MODE_CART_IMPEDANCE, ip)

    def poseTransform(self, srcPose, srcMatrixPose, dstMatrixPose, dstPose):
        """Transform pose between different coordinate systems"""
        return poseTransform(srcPose, srcMatrixPose, dstMatrixPose, dstPose)

    def movej_joint(self, joint, v=0.5, a=0.5):
        """Joint space motion"""
        return moveJToTarget(joint, v, a, ipAddress=self.ip)
    
    def movej_pose(self, pose, v=0.2, a=0.2):
        """Joint space motion to reach a specific pose"""
        return moveJToPose(pose, v, a, ipAddress=self.ip)

    def movej_pose_list(self, poses, v=0.2, a=0.2, r=0.1):
        path_id = createPath(1, ipAddress=self.ip)[1]
        for pose in poses:
            joints= [0.0]*7
            inverse(pose, joints, ipAddress=self.ip)
            addMoveJ(path_id, joints, v, a, r, ipAddress=self.ip)
        runPath(path_id, ipAddress=self.ip)
        self.wait_move()
        destroyPath(path_id, ipAddress=self.ip)

    def speedl(self, speeds, t, a=(0.1, 0.5)):
        speedL(speeds, a, t, ipAddress=self.ip)

    def movel(self, pose, v=0.2, a=0.2):
        """Cartesian space motion"""
        return moveL(pose, v, a, ipAddress=self.ip)

    def sevol_l(self, target_pose):
        target_joint= [0.0]*7
        ref_joints = self.getJointPos()
        ref_joints[2] = 0
        inverse_ext(ref_joints, target_pose, target_joint, ipAddress=self.ip)
        ret = servoJ_ex(target_joint, t=1/50, ah_t=0.1, gain=200, ipAddress=self.ip)
        # ret = servoL(target_pose, t=1/50, ah_t=0.03, gain=200, scale=0.5, ipAddress=self.ip)

    def set_gripper_position(self, distance):
        pass

    def movel_list(self, poses, v=0.2, a=0.2, r=0.1):
        ret = createComplexPath(complex_path_type.NORMAL_POSE_PATH, ipAddress=self.ip)
        if ret[0] == 0:
            for pose in poses:
                addMoveJSegmentByPose(ret[1],pose,v,a,r,ipAddress=self.ip)
            runComplexPath(ret[1] , ipAddress=self.ip)
            self.wait_move()
            destroyComplexPath(ret[1] , ipAddress=self.ip)

    def wait_move(self):
        """Wait for the robot to finish its current motion or until timeout."""
        time.sleep(0.02)
        while True:
            state = getRobotState()
            if state != 0 and state != 10:
                break
            elif state == 10:
                start_joints = self.getJointPos()
                time.sleep(1.0)
                if np.max(np.abs(self.getJointPos()-start_joints)) < 0.5/180*np.pi:
                    break
            else:
                time.sleep(0.001)
        stop()

    def getJointPos(self):
        joints =[0.0]*JOINT_NUM
        getJointPos(joints, ipAddress=self.ip)
        return np.array(joints)

    def freedriving(self, mode=freedriving_mode_e.E_NORMAL_FREEDRIVING):
        """Enable/disable freedriving mode"""
        return freeDriving(mode, ipAddress=self.ip)

    def enter_force_mode(self, frame_type, force_direction, force_value, max_approach_velocity, max_allow_tcp_offset):
        """Enter force control mode"""
        frame_matrix = (1,0,0,0,0,1,0,0,0,0,1,0,0,0,0,1)
        return enterForceMode(frame_type, frame_matrix, force_direction, force_value, max_approach_velocity, max_allow_tcp_offset, ipAddress=self.ip)

    def leave_force_mode(self, mode=mode_e.T_MODE_POSITION):
        """Exit force control mode"""
        return leaveForceMode(mode, ipAddress=self.ip)

    def getjoints(self):
        """Get current joint angles"""
        joints = [0.0] * 7
        getJointPos(joints, ipAddress=self.ip)
        return joints

    def getpose(self):
        """Get current TCP pose"""
        pose = [0.0] * 6
        getTcpPos(pose, ipAddress=self.ip)
        return pose

    def get_tcp_pose(self):
        """Get current TCP pose"""
        pose = [0.0] * 6
        getTcpPos(pose, ipAddress=self.ip)
        return pose
    
    def stop(self):
        """Stop the robot"""
        return stop(ipAddress=self.ip)

    def close(self):
        """Release resources"""
        holdBrake(ipAddress=self.ip)
        destroySrv(ipAddress=self.ip)

# Example usage
if __name__ == '__main__':
    robot = DianaRobot(ip='192.168.2.76')
    print("Current joints:", robot.getjoints())
    print("Current TCP pose:", robot.getpose())