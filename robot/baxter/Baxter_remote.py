import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import JointState

import socket
import numpy as np
import copy


def joint_state_callback(data):
    joint_poses = data.position
    l_gripper_l = joint_poses[1]
    l_gripper_r = joint_poses[2]
    l_e0 = joint_poses[3]
    l_e1 = joint_poses[4]
    l_s0 = joint_poses[5]
    l_s1 = joint_poses[6]
    l_w0 = joint_poses[7]
    l_w1 = joint_poses[8]
    l_w2 = joint_poses[9]

    r_gripper_l = joint_poses[10]
    r_gripper_r = joint_poses[11]
    r_e0 = joint_poses[12]
    r_e1 = joint_poses[13]
    r_s0 = joint_poses[14]
    r_s1 = joint_poses[15]
    r_w0 = joint_poses[16]
    r_w1 = joint_poses[17]
    r_w2 = joint_poses[18]


def main():
    TCP_PORT = 6776
    BUFFER_SIZE = 65536*256

    print("Initializing node... ")
    rospy.init_node("remote_gripper_test")
    moveit_commander.roscpp_initialize([])
    rospy.Subscriber("/robot/joint_states", JointState, joint_state_callback)
    robot = moveit_commander.RobotCommander()
    # scene = moveit_commander.PlanningSceneInterface()

    # initialize interfaces
    # print("Getting robot state... ")
    # rs = baxter_interface.RobotEnable(CHECK_VERSION)
    # init_state = rs.state().enabled

    # print("Enabling robot... ")
    # rs.enable()
    left = baxter_interface.Gripper('left', CHECK_VERSION)
    right = baxter_interface.Gripper('right', CHECK_VERSION)

    # def clean_shutdown():
    #     if not init_state:
    #         print("Disabling robot...")
    #         rs.disable()
    #     print("Exited.")

    # rospy.on_shutdown(clean_shutdown)

    def gripper_ctl(which, operation):
        if which == "left":
            if operation == "open":
                left.open()
                return "OK"
            elif operation == "close":
                left.close()
                return "OK"
        elif which == "right":
            if operation == "open":
                right.open()
                return "OK"
            elif operation == "close":
                right.close()
                return "OK"
        print("gripper command error with parameter: ", which, operation)
        return "Error CMD"

    # pos(x, y, z), rot(x, y, z, w)
    def arm_ctl(which, px, py, pz, rx, ry, rz, rw):
        # just control the target pose of end point
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = float(rw)
        pose_target.orientation.x = float(rx)
        pose_target.orientation.y = float(ry)
        pose_target.orientation.z = float(rz)
        pose_target.position.x = float(px)
        pose_target.position.y = float(py)
        pose_target.position.z = float(pz)
        if which == "right":
            group = moveit_commander.MoveGroupCommander("right_arm")
        else:
            group = moveit_commander.MoveGroupCommander("left_arm")

        group.set_pose_target(pose_target)
        group.plan()
        if group.go(): # true for success, false for failed
            return "OK"
        else:
            return "Failed when making a plan"
    
    def arm_cont_ctl(*args):
        which = args[0]
        traj = args[1:]
        # traj: (1+7*N)
        N = int(traj[0])
        traj_n = np.zeros((N, 7))
        for ind in range(N):
            traj_n[ind, :] = [float(i) for i in traj[1+ind*7:1+(ind+1)*7]]
        print(traj_n)

        if which == "right":
            group = moveit_commander.MoveGroupCommander("right_arm")
        else:
            group = moveit_commander.MoveGroupCommander("left_arm")

        waypoints = []
        # start with the current pose
        waypoints.append(group.get_current_pose().pose)

        for p in traj_n:
            wpose = geometry_msgs.msg.Pose()
            wpose.position.x = p[0]
            wpose.position.y = p[1]
            wpose.position.z = p[2]
            wpose.orientation.x = p[3]
            wpose.orientation.y = p[4]
            wpose.orientation.z = p[5]
            wpose.orientation.w = p[6]
            waypoints.append(copy.deepcopy(wpose))

        plan, _ = group.compute_cartesian_path(waypoints, 0.01, 0.0)
        if group.execute(plan):
            return "OK"
        else:
            return "Failed when making a plan"


    cmds = {
        "gripper": gripper_ctl,
        "arm": arm_ctl,
        "arm_cont": arm_cont_ctl
    }

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.settimeout(100000)
    s.bind(("", TCP_PORT))
    s.listen(1)

    while not rospy.is_shutdown():
        print("Waiting for new connection ......")
        conn, addr = s.accept()
        print("Connected address:", addr)
        while not rospy.is_shutdown():
            data = conn.recv(BUFFER_SIZE)
            if data:
                parts = str(data).split('#')
                if parts[0] in cmds:
                    print('good')
                    backstr = cmds[parts[0]](*parts[1:])
                    conn.send(backstr.decode())  # echo
                else:
                    print('wrong')
                    conn.send(b"Something wrong")  # echo
            else:
                print("Connection closed by client.")
                break

    conn.close()
    print("Exited!")
    # force shutdown call if caught by key handler
    rospy.signal_shutdown("Controller terminated.")


if __name__ == '__main__':
    main()
