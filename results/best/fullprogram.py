import modern_robotics as mr
import numpy as np
import matplotlib.pyplot as plt


'''
    This program is to generate a csv file that can be played in vrep to show a robot doing a 
    pick and place task.

    Example initial conditions would be:
        config = np.array([[0,0,0,0,0,0.2,-0.6,0,0,0,0,0,0]])
        Tseinit = np.array([[0, 0, 1, 0.5],
                        [0, 1, 0, 0],
                        [-1, 0, 0, 0.5],
                        [0, 0, 0, 1]])
        Kp = 20
        Ki = 0
        Tscinit = np.array([[1, 0, 0, 1],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0.025],
                            [0, 0, 0, 1]])
        Tscgoal = np.array([[0, 1, 0, 0],
                            [-1, 0, 0, -1],
                            [0, 0, 1, 0.025],
                            [0, 0, 0, 1]])

    Run the program:
        runfullprogram(Tscinit,Tscgoal,config,Tseinit,Kp,Ki)


'''

#####these below are all the configurations that won't change####### 

#standoff position
Tcestand = np.array([[-0.707, 0, 0.707, 0],
                     [0, 1, 0, 0],
                     [-0.707, 0, -0.707, 0.1],
                     [0, 0, 0, 1]])
#grasp position
Tcegrasp = np.array([[-0.707, 0, 0.707, 0],
                     [0, 1, 0, 0],
                     [-0.707, 0, -0.707, 0.005],
                     [0, 0, 0, 1]])
#integration method
method = 3
k = 1
tstep = 0.01
#body jacobian
Blist = np.array([[0, 0, 0, 0, 0], \
                  [0, -1, -1, -1, 0], \
                  [1, 0, 0, 0, 1], \
                  [0, -0.5076, -0.3526, -0.2176, 0], \
                  [0.033, 0, 0, 0, 0], \
                  [0, 0, 0, 0, 0]])
#forward kinematics
Slist = np.array([[0, 0, 0, 0, 0], \
                  [0, -1, -1, -1, 0], \
                  [1, 0, 0, 0, 1], \
                  [0, 0.147, 0.302, 0.437, 0], \
                  [0, 0, 0, 0, -0.033], \
                  [0, -0.033, -0.033, -0.033, 0]])
M = np.array([[1, 0, 0, 0.033], \
                [0, 1, 0, 0], \
                [0, 0, 1, 0.6546], \
                [0, 0, 0, 1]])
Tbo = np.array([[1, 0, 0, 0.1662], \
                [0, 1, 0, 0], \
                [0, 0, 1, 0.0026], \
                [0, 0, 0, 1]])
#chasis configuration
r = 0.0475
l = 0.235
w = 0.15
F6 = np.array([[0, 0, 0, 0], \
              [0, 0, 0, 0], \
              [r/4 * (-1/(l + w)), r/4 * (1/(l + w)),  r/4 * (1/(l + w)), r/4 * (-1/(l + w))],\
              [         r/4 * 1,         r/4 * 1,          r/4 * 1,          r/4 * 1],\
              [        r/4 * (-1),         r/4 * 1,         r/4 * (-1),          r/4 * 1], \
              [0, 0, 0, 0]])
#wheel and joint speed limit
maxv = 10

#generate ee trajectory
def TrajectoryGenerator(Tseinit,Tscinit,Tscgoal,Tcestand,Tcegrasp,k):
    Tsestand = np.dot(Tscinit,Tcestand)
    Tsegrasp = np.dot(Tscinit,Tcegrasp)
    Tsestandg = np.dot(Tscgoal,Tcestand)
    Tsegraspg = np.dot(Tscgoal,Tcegrasp)
    
    #define a empty ndarray
    refee = np.empty(shape=[0,13])
    
    # init to 1st standoff
    a = mr.ScrewTrajectory(Tseinit,Tsestand,4,400,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],0])])

    # 1st standoff to grasp
    a = mr.ScrewTrajectory(Tsestand,Tsegrasp,1,100,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],0])])

    # 1st grasp
    a = mr.ScrewTrajectory(Tsegrasp,Tsegrasp,1.4,140,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],1])])

    # back to 1st standoff
    a = mr.ScrewTrajectory(Tsegrasp,Tsestand,1,100,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],1])])

    # to final standoff
    a = mr.ScrewTrajectory(Tsestand,Tsestandg,4,400,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],1])])

    # to final release
    a = mr.ScrewTrajectory(Tsestandg,Tsegraspg,1,100,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],1])])

    # release
    a = mr.ScrewTrajectory(Tsegraspg,Tsegraspg,1.4,140,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],0])])

    # back to final standoff
    a = mr.ScrewTrajectory(Tsegraspg,Tsestandg,1,100,method)
    for i in range(len(a)):
        refee = np.vstack([refee,np.array([a[i][0,0],a[i][0,1],a[i][0,2],a[i][1,0],a[i][1,1],\
            a[i][1,2],a[i][2,0],a[i][2,1],a[i][2,2],a[i][0,3],a[i][1,3],a[i][2,3],0])])

    return refee


#simulation
def NextState(twelvecurrentconfig,ninespeed,tstep,maxv):
    # 3 chasis config theta, x, y; 5 joint config; 4 wheel config; 
    # 5 joint speed; 4 wheel speed;
    twelvefinalconfig = np.zeros(shape=[13])

    #speed limit
    for i in range(len(ninespeed)):
        if (abs(ninespeed[i]) > maxv) & (ninespeed[i] > 0):
            ninespeed[i] = maxv
        elif (abs(ninespeed[i]) > maxv) & (ninespeed[i] < 0):
            ninespeed[i] = -maxv
        else: None
    
    #5 joints velocity update
    twelvefinalconfig[3] = twelvecurrentconfig[3] + ninespeed[0] * tstep
    twelvefinalconfig[4] = twelvecurrentconfig[4] + ninespeed[1] * tstep
    twelvefinalconfig[5] = twelvecurrentconfig[5] + ninespeed[2] * tstep
    twelvefinalconfig[6] = twelvecurrentconfig[6] + ninespeed[3] * tstep
    twelvefinalconfig[7] = twelvecurrentconfig[7] + ninespeed[4] * tstep
    #4 wheels velocity update
    twelvefinalconfig[8] = twelvecurrentconfig[8] + ninespeed[5] * tstep
    twelvefinalconfig[9] = twelvecurrentconfig[9] + ninespeed[6] * tstep
    twelvefinalconfig[10] = twelvecurrentconfig[10] + ninespeed[7] * tstep
    twelvefinalconfig[11] = twelvecurrentconfig[11] + ninespeed[8] * tstep
    
    wheelangle = ninespeed[5:9].T * 0.01
    H = r/4 * np.array([[-1/(l + w), 1/(l + w),  1/(l + w), -1/(l + w)],\
                        [         1,         1,          1,          1],\
                        [        -1,         1,         -1,          1]])
    Vb = np.dot(H, wheelangle)
    if Vb[0] == 0:
        qb = np.array([[0], [Vb[1]], [Vb[2]]])
    else: 
        qb = np.array([[Vb[0]], \
                        [(Vb[1] * np.sin(Vb[0]) + Vb[2] * (np.cos(Vb[0]) - 1))/Vb[0]], \
                        [(Vb[2] * np.sin(Vb[0]) + Vb[1] * (1 - np.cos(Vb[0])))/Vb[0]]])
    trans = np.array([[1, 0, 0], \
                      [0, np.cos(twelvecurrentconfig[0]), -np.sin(twelvecurrentconfig[0])], \
                      [0, np.sin(twelvecurrentconfig[0]), np.cos(twelvecurrentconfig[0])]])
    delq = np.dot(trans, qb)
    twelvefinalconfig[0] = twelvecurrentconfig[0] + delq[0]
    twelvefinalconfig[1] = (twelvecurrentconfig[1] + delq[1])
    twelvefinalconfig[2] = (twelvecurrentconfig[2] + delq[2])
    return twelvefinalconfig


#controller
def feedbackcontrol(X, Xd, Xdnext, Kp, Ki, tstep, thetalist, Xerr_integrate):
    
    Vd = mr.se3ToVec((1/tstep) * mr.MatrixLog6(np.dot(mr.TransInv(Xd), Xdnext))) 
    
    #calculate Xerr
    Xerr = mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(X), Xd)))
    
    Xerr_integrate += Xerr * tstep
    
    #generate twist using feedforward and PI control
    V = np.dot(mr.Adjoint(np.dot(mr.TransInv(X), Xd)), Vd) + np.dot(Kp, mr.se3ToVec(mr.MatrixLog6(np.dot(mr.TransInv(X), Xd)))) + np.dot(Ki, Xerr_integrate)
    
    Jarm = mr.JacobianBody(Blist, thetalist)
    Toe = mr.FKinSpace(M, Slist, thetalist)
    Jbase = np.dot(mr.Adjoint(np.dot(mr.TransInv(Toe), mr.TransInv(Tbo))), F6)
    Je = np.hstack([Jbase, Jarm])
    
    #get velocity using pseudoinverse 
    v = np.dot(np.linalg.pinv(Je,1e-3), V)
    return v, Xerr, Xerr_integrate




def runfullprogram(Tscinit,Tscgoal,config,Tseinit,Kp,Ki):
    #generate reference ee trajectory
    refee = TrajectoryGenerator(Tseinit,Tscinit,Tscgoal,Tcestand,Tcegrasp,k)
    Xerrmatrix = np.empty(shape=[6,1480])
    Xerr_integrate = np.array([0,0,0,0,0,0]).astype(float)
    for i in range(1479):
        Tbo = np.array([[1, 0, 0, 0.1662], \
                    [0, 1, 0, 0], \
                    [0, 0, 1, 0.0026], \
                    [0, 0, 0, 1]])
        Xdline = refee[i]
        Xdlinenext = refee[i+1]
        Xd = np.array([[Xdline[0], Xdline[1], Xdline[2], Xdline[9]],\
                    [Xdline[3], Xdline[4], Xdline[5], Xdline[10]],\
                    [Xdline[6], Xdline[7], Xdline[8], Xdline[11]],\
                    [0, 0, 0, 1]        ])
        Xdnext = np.array([[Xdlinenext[0], Xdlinenext[1], Xdlinenext[2], Xdlinenext[9]],\
                        [Xdlinenext[3], Xdlinenext[4], Xdlinenext[5], Xdlinenext[10]],\
                        [Xdlinenext[6], Xdlinenext[7], Xdlinenext[8], Xdlinenext[11]],\
                        [0, 0, 0, 1]        ])
        #update thetalist
        thetalist = np.array([config[i][3], config[i][4], config[i][5], config[i][6], config[i][7]]) 

        Tsb = np.array([[np.cos(config[i][0]), -np.sin(config[i][0]), 0, config[i][1]], \
                    [np.sin(config[i][0]), np.cos(config[i][0]), 0, config[i][2]], \
                    [0, 0, 1, 0.0963], \
                    [0, 0, 0, 1]                              ])
        
        Toe = mr.FKinSpace(M, Slist, thetalist)
        X = np.linalg.multi_dot([Tsb,Tbo,Toe])
        [v, Xerrmatrix[:,i], Xerr_integrate] = feedbackcontrol(X, Xd, Xdnext, Kp, Ki, tstep, thetalist, Xerr_integrate)
        v=np.hstack([v[4:9],v[0:4]])
        newconfig = NextState(config[i],v,tstep,maxv)
        config = np.vstack([config,newconfig])
    
    #add gripper condition
    config[:,12] = refee[:,12]
    
    #plot Xerr
    xcord = np.linspace(0,14.78,1479)
    f = plt.figure()
    plt.plot(xcord,Xerrmatrix[0,0:1479],label='err_wx')
    plt.plot(xcord,Xerrmatrix[1,0:1479],label='err_wy')
    plt.plot(xcord,Xerrmatrix[2,0:1479],label='err_wz')
    plt.plot(xcord,Xerrmatrix[3,0:1479],label='err_vx')
    plt.plot(xcord,Xerrmatrix[4,0:1479],label='err_vy')
    plt.plot(xcord,Xerrmatrix[5,0:1479],label='err_vz')
    plt.legend()
    plt.show()
    f.savefig("Xerr.pdf", bbox_inches='tight')

    #generate files
    print("Writing error plot data.")
    np.savetxt('Xerr.csv', Xerrmatrix, delimiter=',')
    print("Generating animation csv file.")
    np.savetxt('config.csv', config, delimiter=',')
    print("Done.")

    return None
