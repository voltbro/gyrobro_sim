import numpy as np
import matplotlib.pyplot as plt
import vbcontrolpy as vb
import control

class MotorSimple():
    def __init__(self, 
                 Ts = 0.001,
                 b = 0.00001,
                 J = 0.00001,
                 K = 0.3,
                 R = 0.4,
                 L = 0.00006,
                 ):

        self.K = K
        
        self.i = 0.0

        self.voltage_ref = 0 #np.zeros(self.num_motors)

        A = np.array( [ [0, 1, 0],
                        [0, -b/J, K/J],
                        [0, -K/L, -R/L]], dtype=np.float32)
        B = np.array([  [0],
                        [0],
                        [1/L]], dtype=np.float32)
        C = np.eye(3)
        D = np.zeros((3,1))
        csys = control.ss(A, B, C, D)
        self.dsys = control.sample_system(csys, Ts, method='bilinear')

    def set_ref_voltage(self, voltage):
        self.voltage_ref = np.array([voltage], dtype=np.float32) #voltage # 

    def set_sensor_data(self, theta, omega):
        self.theta_cur = theta #np.array(theta)
        self.omega_cur = omega #np.array(omega)

    def step(self):

        x = np.array([self.theta_cur, self.omega_cur, self.i], dtype=np.float32)
        x = np.matmul(self.dsys.A, x) + np.matmul(self.dsys.B, self.voltage_ref)
        self.i = x[2]

        return x[2] * self.K, x
    

if __name__ == "__main__":
    print("MotorSimple check")
    inc = 0.00005
    mot = MotorSimple(Ts=inc,
                      b=0.0001,
                      J=0.00001,
                      K=0.42,
                      R=0.4,
                      L=0.00006)
    x = np.array([0.0, 0, 0.0], dtype=np.float32)
    t = 0
    it = 0

    theta_vec = []
    omega_vec = []
    tau_vec = []
    i_vec = []
    t_vec = []
    mot.set_sensor_data(x[0], x[1])

    while t < 3.05:
        mot.set_ref_voltage(3)

        # if it % 40 == 0:
        #     mot.set_sensor_data(x[0], x[1])

        tau, x = mot.step()

        t += inc
        it += 1

        theta_vec.append(x[0])
        omega_vec.append(x[1])
        i_vec.append(x[2])
        tau_vec.append(tau)
        t_vec.append(t)

    plt.subplot(3, 1, 1)
    plt.plot(t_vec, theta_vec)
    plt.title("theta")
    plt.subplot(3, 1, 2)
    plt.plot(t_vec, omega_vec)
    plt.title("d_theta")
    plt.subplot(3, 1, 3)
    plt.plot(t_vec, tau_vec)
    plt.title("current")
    plt.show()