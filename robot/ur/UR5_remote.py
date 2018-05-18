import rospy
import baxter_interface
from baxter_interface import CHECK_VERSION

import moveit_commander
import moveit_msgs.msg
import geometry_msgs.msg
from sensor_msgs.msg import JointState

import socket


def joint_state_callback(data):
    joint_poses = data.position 
    elbow = joint_poses[0]
    shoulder_lift = joint_poses[1]
    shoulder_pan = joint_poses[2]
    wrist_1 = joint_poses[3]
    wrist_2 = joint_poses[4]
    wrist_3 = joint_poses[5]


def main():
    TCP_PORT = 6776
    BUFFER_SIZE = 1024

    print("Initializing node... ")
    rospy.init_node("remote_gripper_test")
    moveit_commander.roscpp_initialize([])
    rospy.Subscriber("/joint_states", JointState, joint_state_callback)
    robot = moveit_commander.RobotCommander()
    scene = moveit_commander.PlanningSceneInterface()

    # initialize interfaces
    # print("Getting robot state... ")
    # rs = baxter_interface.RobotEnable(CHECK_VERSION)
    # init_state = rs.state().enabled

    # print("Enabling robot... ")
    # rs.enable()
    # left = baxter_interface.Gripper('left', CHECK_VERSION)
    # right = baxter_interface.Gripper('right', CHECK_VERSION)

    # def clean_shutdown():
    #     if not init_state:
    #         print("Disabling robot...")
    #         rs.disable()
    #     print("Exited.")

    # rospy.on_shutdown(clean_shutdown)

    # def gripper_ctl(which, operation):
    #     if which == "left":
    #         if operation == "open":
    #             left.open()
    #             return "OK"
    #         elif operation == "close":
    #             left.close()
    #             return "OK"
    #     elif which == "right":
    #         if operation == "open":
    #             right.open()
    #             return "OK"
    #         elif operation == "close":
    #             right.close()
    #             return "OK"
    #     print("gripper command error with parameter: ", which, operation)
    #     return "Error CMD"

    # pos(x, y, z), rot(x, y, z, w)
    def arm_ctl(px, py, pz, rx, ry, rz, rw):
        # just control the target pose of end point
        pose_target = geometry_msgs.msg.Pose()
        pose_target.orientation.w = float(rw)
        pose_target.orientation.x = float(rx)
        pose_target.orientation.y = float(ry)
        pose_target.orientation.z = float(rz)
        pose_target.position.x = float(px)
        pose_target.position.y = float(py)
        pose_target.position.z = float(pz)
        group = moveit_commander.MoveGroupCommander("manipulator")

        group.set_pose_target(pose_target)
        group.plan()
        if group.go(): # true for success, false for failed
            return "OK"
        else:
            return "Failed when making a plan"

    cmds = {
        "gripper": gripper_ctl,
        "arm": arm_ctl
    }

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(None)
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
                    backstr = cmds[parts[0]](*parts[1:])
                    conn.send(backstr.decode())  # echo
                else:
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
