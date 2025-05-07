import numpy as np
import sympy as sym
import time

def lqr(A, B, Q, R):
    """
    Compute the lqr controller gain matrix K.
    """
    # Solve the continuous time algebraic Riccati equation
    P = np.linalg.solve_continuous_are(A, B, Q, R)
    # Compute the lqr gain matrix
    K = np.linalg.inv(R) @ B.T @ P
    return K

# Suppress the use of scientific notation when printing small numbers
np.set_printoptions(suppress=True)

params = {
    'm': 0.5,
    'Jx': 0.0023,
    'Jy': 0.0023,
    'Jz': 0.0040,
    'l': 0.175,
    'g': 9.81,
}

# components of position (meters)
p_x, p_y, p_z = sym.symbols('p_x, p_y, p_z')

# yaw, pitch, roll angles (radians)
psi, theta, phi = sym.symbols('psi, theta, phi')

# components of linear velocity (meters / second)
v_x, v_y, v_z = sym.symbols('v_x, v_y, v_z')
v_in_body = sym.Matrix([v_x, v_y, v_z])

# components of angular velocity (radians / second)
w_x, w_y, w_z = sym.symbols('w_x, w_y, w_z')
w_in_body = sym.Matrix([w_x, w_y, w_z])

# components of net rotor torque
tau_x, tau_y, tau_z = sym.symbols('tau_x, tau_y, tau_z')

# net rotor force
f_z = sym.symbols('f_z')

# parameters
m = sym.nsimplify(params['m'])
Jx = sym.nsimplify(params['Jx'])
Jy = sym.nsimplify(params['Jy'])
Jz = sym.nsimplify(params['Jz'])
l = sym.nsimplify(params['l'])
g = sym.nsimplify(params['g'])
J = sym.diag(Jx, Jy, Jz)

# rotation matrices
Rz = sym.Matrix([[sym.cos(psi), -sym.sin(psi), 0], [sym.sin(psi), sym.cos(psi), 0], [0, 0, 1]])
Ry = sym.Matrix([[sym.cos(theta), 0, sym.sin(theta)], [0, 1, 0], [-sym.sin(theta), 0, sym.cos(theta)]])
Rx = sym.Matrix([[1, 0, 0], [0, sym.cos(phi), -sym.sin(phi)], [0, sym.sin(phi), sym.cos(phi)]])
R_body_in_world = Rz @ Ry @ Rx

# angular velocity to angular rates
ex = sym.Matrix([[1], [0], [0]])
ey = sym.Matrix([[0], [1], [0]])
ez = sym.Matrix([[0], [0], [1]])
M = sym.simplify(sym.Matrix.hstack((Ry @ Rx).T @ ez, Rx.T @ ey, ex).inv(), full=True)

# applied forces
f_in_body = R_body_in_world.T @ sym.Matrix([[0], [0], [-m * g]]) + sym.Matrix([[0], [0], [f_z]])

# applied torques
tau_in_body = sym.Matrix([[tau_x], [tau_y], [tau_z]])

# equations of motion
f = sym.Matrix.vstack(
    R_body_in_world @ v_in_body,
    M @ w_in_body,
    (1 / m) * (f_in_body - w_in_body.cross(m * v_in_body)),
    J.inv() @ (tau_in_body - w_in_body.cross(J @ w_in_body)),
)

f = sym.simplify(f, full=True)

# Position of drone in world frame
p_in_world = sym.Matrix([p_x, p_y, p_z])

# Position of markers in body frame
a_in_body = sym.Matrix([0, l, 0])  # <-- marker on left rotor
b_in_body = sym.Matrix([0, -l, 0]) # <-- marker on right rotor

# Position of markers in world frame
a_in_world = p_in_world + R_body_in_world @ a_in_body
b_in_world = p_in_world + R_body_in_world @ b_in_body

# Sensor model
g = sym.simplify(sym.Matrix.vstack(a_in_world, b_in_world))

# Linearize the system
m = sym.Matrix([p_x, p_y, p_z, psi, theta, phi, v_x, v_y, v_z, w_x, w_y, w_z])
n = sym.Matrix([tau_x, tau_y, tau_z, f_z])
o = sym.Matrix([p_x, p_y, p_z, psi, theta, phi])
sub = {
    p_x: 0.0,
    p_y: 0.0,
    p_z: 0.0,
    psi: 0.0,
    theta: 0.0,
    phi: 0.0,
    v_x: 0.0,
    v_y: 0.0,
    v_z: 0.0,
    w_x: 0.0,
    w_y: 0.0,
    w_z: 0.0,
    tau_x: 0.0,
    tau_y: 0.0,
    tau_z: 0.0,
    f_z: 981./100./2.,
}
m_e = np.array(m.subs(sub)).astype(np.float64)
n_e = np.array(n.subs(sub)).astype(np.float64)
o_e = np.array(o.subs(sub)).astype(np.float64)

A = f.jacobian(m).subs(sub)
B = f.jacobian(n).subs(sub)
C = g.jacobian(m).subs(sub)

A = np.array(A).astype(np.float64)
B = np.array(B).astype(np.float64)
C = np.array(C).astype(np.float64)


Q_startend = np.diag([0.15, 0.15, 7.5, 1, 1, 1,
                        1, 1, 1, 1, 1, 1])
Q_zigzag = np.diag([0.09, 0.09, 10, 1, 1, 1,
                    1, 1, 1, 1, 1, 1]) # good for zigzag
Q_fasty = np.diag([0.1, 0.15, 10, 1, 1, 1,
                    1, 1, 1, 1, 1, 1]) # good for fast y
Q_ringx = np.diag([0.13, 0.15, 9.5, 1, 1, 1,
                    1, 1, 1, 1, 1, 1])
Q_ringy = np.diag([0.08, 0.1, 7, 1, 1, 1,
                    1, 1, 1, 1, 1, 1])
Q_13 = np.diag([0.1, 0.1, 13, 1, 1, 1,
                    1, 1, 1, 1, 1, 1]) # testing

# Rc = np.diag([.95, .95, .95, .95]) * 1e2
Rc = np.diag([1, 1, 1, 1]) * 1e2 * 0.9

K_startend, _, _ = lqr(A, B, Q_startend, Rc)
K_zigzag, _, _ = lqr(A, B, Q_zigzag, Rc)
K_fasty, _, _ = lqr(A, B, Q_fasty, Rc)
K_ringx, _, _ = lqr(A, B, Q_ringx, Rc)
K_ringy, _, _ = lqr(A, B, Q_ringy, Rc)
K_13, _, _ = lqr(A, B, Q_13, Rc)

Qo = np.diag([1, 1, 1, 1, 1, 1]) * 1e2
Ro = np.diag([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]) * 1e2
L, _, _ = lqr(A.T, C.T, np.linalg.inv(Ro), np.linalg.inv(Qo))
L = L.T


class Controller:
    def __init__(self):
        """
        List all class variables you want the simulator to log. For
        example, if you want the simulator to log "self.xhat", then
        do this:
        
            self.variables_to_log = ['xhat']
        
        Similarly, if you want the simulator to log "self.xhat" and
        "self.y", then do this:
        
            self.variables_to_log = ['xhat', 'y']
        
        Etc. These variables have to exist in order to be logged.
        """
        
        self.dt = 0.04
        self.slope = None
        self.zigzag = None
        self.ring_radius = 1.0
        self.variables_to_log = []
        self.passed_rings = []
        self.next_ring = []
        self.passed_checkpoints = []
        self.next_checkpoints = []
        self.run_times = 0
        self.x_des = np.zeros(12)

    def get_color(self):
        """
        If desired, change these three numbers - RGB values between
        0 and 1 - to change the color of your drone.
        """
        return [
            0., # <-- how much red (between 0 and 1)
            1., # <-- how much green (between 0 and 1)
            0., # <-- how much blue (between 0 and 1)
        ]

    def reset(
            self,
            p_x, p_y, p_z, # <-- approximate initial position of drone (meters)
            yaw,           # <-- approximate initial yaw angle of drone (radians)
        ):
        """
        Replace the following line (a placeholder) with your
        own code.
        """
        self.xhat = np.array([p_x, p_y, p_z, yaw, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    def run(
            self,
            pos_markers,
            pos_ring,
            dir_ring,
            is_last_ring,
            pos_others,
        ):
        """
        pos_markers is a 1d array of length 6:
        
            [
                measured x position of marker on left rotor (meters),
                measured y position of marker on left rotor (meters),
                measured z position of marker on left rotor (meters),
                measured x position of marker on right rotor (meters),
                measured y position of marker on right rotor (meters),
                measured z position of marker on right rotor (meters),
            ]
        
        pos_ring is a 1d array of length 3:
        
            [
                x position of next ring center (meters),
                y position of next ring center (meters),
                z position of next ring center (meters),
            ]
        
        dir_ring is a 1d array of length 3:
        
            [
                x component of vector normal to next ring (meters),
                y component of vector normal to next ring (meters),
                z component of vector normal to next ring (meters),
            ]
        
        is_last_ring is a boolean that is True if the next ring is the
                     last ring, and False otherwise
        
        pos_others is a 2d array of size n x 3, where n is the number of
                   all other active drones:
            
            [
                [x_1, y_1, z_1], # <-- position of 1st drone (meters)
                [x_2, y_2, z_2], # <-- position of 2nd drone (meters)
                
                ...
                
                [x_n, y_n, z_n], # <-- position of nth drone (meters)
            ]      
        """

        def reached_destination(xhat, goal):
            return np.linalg.norm(xhat - goal) < 0.75
        
        def get_next_checkpoint(ring_pos, current_checkpoint_pos, ring_radius, forward):
            # Calculate the slope of the line connecting the current ring and checkpoint
            slope = 0.9 * (current_checkpoint_pos[1] - ring_pos[1]) / (current_checkpoint_pos[0] - ring_pos[0])
            # Calculate the x and y coordinates of the next checkpoint
            if (forward):
                x_next = ring_pos[0] + self.ring_radius
                y_next = ring_pos[1] + slope * ring_radius
            else:
                x_next = ring_pos[0] - self.ring_radius
                y_next = ring_pos[1] - slope * ring_radius
            return np.array([x_next, y_next, ring_pos[2]])
        
        def get_drone_dir(initial_pos, goal):
            dir = np.array([goal[0] - initial_pos[0], goal[1] - initial_pos[1], goal[2] - initial_pos[2]])
            return dir / np.linalg.norm(dir)
        
        def turning(direction):
            threshold = 0.1
            if direction == "x":
                return abs(self.xhat[6]) < threshold
            elif direction == "y":
                return abs(self.xhat[7]) < threshold
            elif direction == "z":
                return abs(self.xhat[8]) < threshold
            else:
                raise ValueError()

        if (self.run_times == 0):
            p_x = self.xhat[0]
            p_y = self.xhat[1]
            self.run_times += 1

        # Default self.x_des, moves towards the ring
        self.x_des = np.array([*pos_ring, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        if (len(self.next_ring) == 0 or not np.allclose(pos_ring, self.next_ring[-1], atol=1e-3)): # runs everytime pos_ring changes
            self.next_ring.append(pos_ring.copy())
            if (len(self.next_ring) > 1):
                self.passed_rings.append(self.next_ring[-2].copy())
        
        # Track the rings, save each ring location to self.passed_rings
        dist_to_ring = np.linalg.norm(self.xhat[:3] - pos_ring)
        
        # If before ring 8, drone is moving forward, if after ring 8, drone is moving backward
        forward = True
        if (len(self.passed_rings) >= 8):
            forward = False

        # Ensure drone rises above initial blue ring quickly
        if (self.xhat[2] < 2.0 and len(self.passed_checkpoints) == 0):
            checkpoint = np.array([p_x, p_y, 1.075])
            self.x_des[:3] = checkpoint + np.array([0, 0, 15.0])
            if reached_destination(self.xhat[:3], checkpoint):
                
                self.passed_checkpoints.append(self.xhat[:3].copy())
                self.next_checkpoints.append(self.xhat[:3].copy())        
        # Ring 1
        if (len(self.passed_rings) == 0 and len(self.passed_checkpoints) == 1):
            self.x_des[:3] = self.next_ring[-1] + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 4.5 # Overshoot here
            if ((self.xhat[1] - self.next_ring[-1][1])**2 + (self.xhat[2] - self.next_ring[-1][2])**2 < (self.ring_radius - 0.1)**2):
                self.x_des[:3] = self.next_ring[-1] + np.array([2.0, 0, 0])

        if (len(self.passed_rings) == 1 and turning("y") and len(self.passed_checkpoints) == 1):
            self.passed_checkpoints.append(self.xhat[:3].copy())
            self.next_checkpoints.append(self.xhat[:3].copy())

        if (len(self.passed_rings) == 1 and len(self.passed_checkpoints) == 2):
            del_z = self.next_ring[-1][2] - self.passed_checkpoints[-1][2]
            self.x_des[:3] = pos_ring + np.array([0, 0, 0.0]) + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 2.5 # Overshoot here, max 2.0
            if (del_z < -0.5):
                self.x_des[2] += 1.0
            elif (del_z > 1.0):
                self.x_des[2] -= 0.5
            if ((self.xhat[0] - self.next_ring[-1][0])**2 + (self.xhat[2] - self.next_ring[-1][2])**2 < (self.ring_radius - 0.15)**2) and abs(self.xhat[1] - self.next_ring[-1][1]) > 1.25:
                self.x_des[:3] += dir_ring * 2.0

        # To get to first checkpoint after first zigzag ring, entering zigzag rings
        if len(self.passed_rings) == 2 and len(self.passed_checkpoints) == 2:
            self.slope = 0.5 * (self.passed_rings[-1][1] - self.passed_rings[-2][1]) / (self.passed_rings[-1][0] - self.passed_rings[-2][0])
            self.zigzag = np.array([self.ring_radius, self.slope * self.ring_radius, 0])
            checkpoint = self.passed_rings[-1][:3] + self.zigzag
            self.x_des[:3] = checkpoint + np.array([0, 0, 0.75]) + get_drone_dir(self.passed_rings[-1], checkpoint) * 4.0 # Overshoot here
            if reached_destination(self.xhat[:3], checkpoint):                
                self.passed_checkpoints.append(checkpoint.copy())
                self.next_checkpoints.append(checkpoint.copy())
                next_checkpoint = get_next_checkpoint(pos_ring, checkpoint, self.ring_radius, forward)
                self.next_checkpoints.append(next_checkpoint.copy())

        # Approaching ring 3
        if (len(self.passed_checkpoints) == 3):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + (get_drone_dir(self.passed_checkpoints[-1], pos_ring)) * 4.0 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        # Approaching ring 4
        if (len(self.passed_checkpoints) == 4):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 3.5 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        # Approaching ring 5
        if (len(self.passed_checkpoints) == 5):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 3.5 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        # Approaching ring 6
        if (len(self.passed_rings) == 5 and len(self.passed_checkpoints) == 6):
            self.x_des[:3] = pos_ring + np.array([0, 0, 0.5]) + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 2.5 # Overshoot here

        # Approaching ring 7
        if (len(self.passed_rings) == 6 and len(self.passed_checkpoints) == 6 and all(self.passed_rings[-1] != pos_ring)):
            self.x_des[:3] = pos_ring + get_drone_dir(self.passed_rings[-1], pos_ring) * 3.0 # Overshoot here
            if ((self.xhat[1] - pos_ring[1])**2 + (self.xhat[2] - pos_ring[2])**2 < (self.ring_radius - 0.)**2):
                self.x_des[:3] = pos_ring + dir_ring * 2.0 # Overshoot here

        if (len(self.passed_rings) == 7 and len(self.passed_checkpoints) == 6):
            checkpoint = np.array([self.passed_rings[-1][0] + 0.6, self.passed_rings[-1][1], self.passed_rings[-1][2]])
            self.x_des[:3] = checkpoint + np.array([4.5, 0, 0]) # Overshoot here
            if (reached_destination(self.xhat[:3], checkpoint)):
                self.passed_checkpoints.append(checkpoint.copy())
                
        if (len(self.passed_rings) == 7 and len(self.passed_checkpoints) == 7):
            checkpoint = np.array([pos_ring[0] + 0.50, pos_ring[1], pos_ring[2]]) ## For some reason, only pos_ring[0] + 1.0 and above works
            self.x_des[:3] = checkpoint + np.array([0, 1.5, 0]) ## Don't go above 2.0 # Overshoot here
            if ((self.xhat[1] - self.next_ring[-1][1])**2 + (self.xhat[2] - self.next_ring[-1][2])**2 < (self.ring_radius - 0.15)**2):
                self.passed_checkpoints.append(checkpoint.copy())
                self.x_des[:3] = pos_ring + dir_ring * 4.0 # Overshoot here
        
        if (len(self.passed_rings) == 7 and len(self.passed_checkpoints) == 8):
            
            self.x_des[:3] = pos_ring + dir_ring * 4.0 # Overshoot here

        if (len(self.passed_rings) == 8 and len(self.passed_checkpoints) == 8):
            
            self.x_des[:3] = self.next_ring[-1] + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 4.5 # Overshoot here

        if (len(self.passed_rings) == 8 and turning("y") and len(self.passed_checkpoints) == 8):
            self.passed_checkpoints.append(self.xhat[:3].copy())
            self.next_checkpoints.append(self.xhat[:3].copy())

        if (len(self.passed_rings) == 8 and len(self.passed_checkpoints) == 9):
            del_z = self.next_ring[-1][2] - self.passed_checkpoints[-1][2]
            self.x_des[:3] = self.next_ring[-1] + np.array([0, 0, 0.0]) + get_drone_dir(self.passed_checkpoints[-1], self.next_ring[-1]) * 2.0 # Overshoot here
            if (del_z < 0.0):
                self.x_des[2] += 0.75
            elif (del_z > 1.0):
                self.x_des[2] -= 0.25
            if ((self.xhat[0] - self.next_ring[-1][0])**2 + (self.xhat[2] - self.next_ring[-1][2])**2 < (self.ring_radius - 0.15)**2) and abs(self.xhat[1] - self.next_ring[-1][1]) > 1.25:
                self.x_des[:3] += dir_ring * 2.0 # Overshoot here

        if len(self.passed_rings) == 9 and len(self.passed_checkpoints) == 9:
            self.slope = 0.60 * (self.passed_rings[-1][1] - self.passed_rings[-2][1]) / (self.passed_rings[-1][0] - self.passed_rings[-2][0])
            self.zigzag = np.array([self.ring_radius, self.slope * self.ring_radius, 0])
            checkpoint = self.passed_rings[-1] - self.zigzag
            self.x_des[:3] = checkpoint + np.array([0, 0, 0.5]) + get_drone_dir(self.passed_rings[-1], checkpoint) * 4.0 # Overshoot here
            if reached_destination(self.xhat[:3], checkpoint):
                self.passed_checkpoints.append(checkpoint.copy())
                self.next_checkpoints.append(checkpoint.copy())
                next_checkpoint = get_next_checkpoint(pos_ring, checkpoint, self.ring_radius, forward)
                self.next_checkpoints.append(next_checkpoint.copy())

        if (len(self.passed_checkpoints) == 10):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + (get_drone_dir(self.passed_checkpoints[-1], pos_ring)) * 4.0 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        if (len(self.passed_checkpoints) == 11):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + (get_drone_dir(self.passed_checkpoints[-1], pos_ring)) * 3.5 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        if (len(self.passed_checkpoints) == 12):
            self.x_des[:3] = self.next_checkpoints[-1] + np.array([0, 0, 0.5]) + (get_drone_dir(self.passed_checkpoints[-1], pos_ring)) * 3.5 # Overshoot here
            if reached_destination(self.xhat[:3], self.next_checkpoints[-1]):                
                self.passed_checkpoints.append(self.next_checkpoints[-1].copy())
                self.next_checkpoints.append(get_next_checkpoint(pos_ring, self.next_checkpoints[-1], self.ring_radius, forward))

        if (len(self.passed_rings) == 12 and len(self.passed_checkpoints) == 13):            
            self.x_des[:3] = pos_ring + get_drone_dir(self.passed_checkpoints[-1], pos_ring) * 4.0 # Overshoot here

        if (len(self.passed_rings) == 13 and len(self.passed_checkpoints) == 13):
            self.passed_checkpoints.append(self.xhat[:3].copy())

        if (len(self.passed_rings) == 13 and all(self.passed_rings[-1] != pos_ring) and len(self.passed_checkpoints) == 14):
            self.x_des[:3] = self.next_ring[-1] + np.array([0, -0.5, 0]) + get_drone_dir(self.passed_checkpoints[-1], self.next_ring[-1]) * 2.5 # Overshoot here
            if ((self.next_ring[-1][0] - self.xhat[0])**2 + (self.next_ring[-1][1] - self.xhat[1])**2 < (self.ring_radius - 0.4)**2):
                self.x_des[:3] = self.next_ring[-1] + dir_ring * 5.0
        
        if (is_last_ring):
            if (len(self.passed_checkpoints) == 14):
                R = 2.5
                checkpoint = np.array([0, 0, 1.75])
                self.x_des[:3] = checkpoint + get_drone_dir(self.passed_rings[-1], checkpoint) * 5.0 # Overshoot here
                if (self.xhat[0] < R*np.sqrt(2)/2 and self.xhat[1] > -R*np.sqrt(2)/2):
                    self.passed_checkpoints.append(checkpoint.copy())
            else:
                self.x_des[:3] = np.array([0, 0, -50])
        
        if dist_to_ring < 1.0:
            self.x_des[:3] += dir_ring * 3.0
        if (dist_to_ring < 1.0 and abs(dir_ring[0]) == 1):
            u = -K_ringx @ (self.xhat - self.x_des)
        elif (dist_to_ring < 1.0 and abs(dir_ring[1]) == 1):
            u = -K_ringy @ (self.xhat - self.x_des)
        elif (len(self.passed_rings) < 1):
            u = -K_startend @ (self.xhat - self.x_des)
        elif (len(self.passed_rings) == 7 and len(self.passed_checkpoints) == 7):
            u = -K_fasty @ (self.xhat - self.x_des)
        elif (len(self.passed_rings) == 13):
            u = -K_13 @ (self.xhat - self.x_des)
        else:
            u = -K_zigzag @ (self.xhat - self.x_des)

        tau_x = u[0]
        tau_y = u[1]
        tau_z = u[2]
        f_z = u[3] + n_e[3][0]
        y = np.array(pos_markers)

        x_hatdot = A @ self.xhat + B @ u - L @ (C @ self.xhat - y)
        self.xhat += + x_hatdot.flatten() * self.dt

        return tau_x, tau_y, tau_z, f_z