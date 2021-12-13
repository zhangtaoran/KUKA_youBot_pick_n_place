import numpy as np
import fullprogram

config = np.array([[0,0,0,0,-0.5,0.2,-0.6,0,0,0,0,0,0]])
Tseinit = np.array([[0, 0, 1, 0.5],
                   [0, 1, 0, 0],
                   [-1, 0, 0, 0.5],
                   [0, 0, 0, 1]])
Kp = np.diag([2,2,2,2,2,2])
Ki = np.diag([0,0,0,0,0,0])
Tscinit = np.array([[1, 0, 0, 1],
                    [0, 1, 0, 0],
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]])
Tscgoal = np.array([[0, 1, 0, 0],
                    [-1, 0, 0, -1],
                    [0, 0, 1, 0.025],
                    [0, 0, 0, 1]])
fullprogram.runfullprogram(Tscinit,Tscgoal,config,Tseinit,Kp,Ki)
