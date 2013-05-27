
#  This library is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public
#  License as published by the Free Software Foundation; either
#  version 3.0 of the License, or (at your option) any later version.
#
#  The library is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
#  General Public License for more details.
#
# (c) Sam Burden, UC Berkeley, 2011 
# (c) Shai Revzen, U Penn, 2010

import numpy as np
import pylab as plt
import scipy as sp
import scipy.ndimage as si

class Obj(object):
    def __init__(self, sys, N, obs, M):
        """
        uk.Obj(sys, N, obs, M)  creates unscented Kalman object

        INPUTS
          sys - function - system evolution 
          N - int - dimension of system
          obs - function - observation
          M - int - dimension of observation

        Both  sys  and  obs  must be vectorized:
            (x,U) |--> y
        so that multiple inputs in columns of x yield multiple ouputs
        in columns of y, i.e. .y[:,k:k+1] = fun(x[:,k:k+1], U)
        U is a collection of shared inputs.

        Missing observations can be indicated as  NaN's  in  y.  These
        will be propagated assuming a null innovation (i.e. correct 
        dynamical model) by increasing the observation covariance 
        in the columns and rows of the missing measurements.

        USAGE
          uko = uk.Obj(sys, N, obs, M)
          for k in range(K):
              x[:,k:k+1] = uk.x
              uko.filter(y[:,k:k+1], U)
        """
        self.utSys = Transform(sys, N, N)
        self.utObs = Transform(obs, N, M)
        self.N = N
        self.M = M
        self.Q = np.identity(N)
        self.R = np.identity(M)
        self.Rm = np.identity(M)
        self.x = np.zeros((N,1))
        self.C = np.identity(N)
        self.x0  = np.kron(np.ones((N,1)),np.nan)
        self.Cx0 = np.kron(np.ones((N,N)),np.nan)
        self.y0  = np.kron(np.ones((M,1)),np.nan)
        self.Cy0 = np.kron(np.ones((M,M)),np.nan)
        
    def filter(self, y, **U):
        """
        filter(y, *U)  performs one UKF step

        INPUTS
          y - M x 1 - observation
          U - tuple - additional arguments to pass to unscented functions
        """
        # Update system estimate
        x0, C0, xxC = self.utSys(self, self.x, self.C, **U)
        xC = C0 + self.Q
        
        # Update observation estimate
        y0, yC, yxC = self.utObs(self, x0, xC, **U)

        # Locate missing measurements
        m = np.isnan(y)
        mi = m.ravel().nonzero()

        # Innovate
        v = y - y0
        v[m] = 0

        # Observation covariance with correction for missing measurements
        S = yC + self.R
        if any(m):
            S[mi,mi] = S[mi,mi] + self.Rm[mi,mi]

        # Kalman step
        K = np.dot( yxC.T, np.linalg.inv(S) )

        # Update state estimate
        self.x = x0 + np.dot( K, v )

        # Update covariance
        self.C = xC - np.dot( np.dot( K, S), K.T )

        # Store a-priori values
        self.x0  = x0
        self.Cx0 = C0
        self.y0  = y0
        self.Cy0 = yC

class Transform(object):
    def __init__(self, fun, N, M, W0=1.0/3.0):
        """
        uk.Transform(fun, N, M, W0=1.0/3.0) unscented transform for fun

        INPUTS
          fun - function handle - vectorized dynamics or observation
          N - scalar - size of state
          M - scalar - size of output
          W0 - scalar - weight parameter; 1/3 for Gaussian

        NOTES
        fun must be vectorized such that 
          fun : (x,U) --> y
        satisfies y[:,k:k+1] = fun(x[:,k:k+1], U)

        The Transform object is callable:
          ut = uk.Transform(fun, N, M)
          y,yCov,yxCov = ut(ukf, x, U) 
        """
        self.W0 = W0

        self.N = N
        self.fun = fun

        # Number of sigma points for unscented transform
        self.nS = 2*self.N+1

        # Weight vector for sigma points
        wS = np.kron( np.ones((1,self.nS)), (1.0-self.W0)/(2.0*self.N) )
        wS[0,0] = self.W0
        self.wS = wS.T
        self.wS_mat = np.kron( np.ones((M,1)), wS )

        # Pre-Cholesky weight
        #self.wF = self.N / (1.0+self.W0)
        self.wF = self.N / (1.0-self.W0) # Thanks to Kenich Shirakawa for fix!

        # Pre-allocate constant arrays for speed
        self.ones = np.ones((1, self.nS))
        self.zeros = np.zeros((self.N,1))

    def __call__(self, ukf, x, xCov, **U):
        """
        ut(ukf, x, xCov, **U) performs unscented Kalman transform

        INPUTS
          ukf - uk.Obj - the unscented Kalman filter
          x - N x 1 - state estimate
          xCov - N x N - state covariance

        OUTPUTS
          y - M x 1 - optimal output estimate 
          yCov - M x M - output covariance
          yxCov - M x N - output / state covariance
        """
        # Compute `matrix square root'
        rt = np.linalg.cholesky(self.wF*xCov)

        # Build symmetric sigma-point set
        xS = ( np.hstack( (self.zeros, -rt, rt) )
                + np.kron( self.ones, x ) )

        # Apply nonlinear function (i.e. dynamics or observation)
        yS = self.fun(xS, **U)

        # Obtain result as weighted average of outputs
        y = np.dot(yS, self.wS)

        # Compute covariances
        yS0 = yS.copy()
        yS = yS0 - np.kron( self.ones, y)

        # Element-wise multiplication
        wyS = self.wS_mat * yS

        # Matrix multiplication
        yCov = np.dot(wyS, yS.T)
        yxCov = np.dot(wyS, xS.T)

        return y, yCov, yxCov

def null(uk, y, **D):
    """
    Default UK filter
    """
    # only update when all observations are present
    if not(any(isnan(y))):
        # default initial state
        if not(hasattr(uk, 'x')):
            uk.x = np.zeros(y.shape)
            uk.lastgood = uk.x

        uk.x = uk.lastgood
        uk.y0 = uk.lastgood

    else:
        uk.x = y
        uk.lastgood = y

def mocap(uk, y):
    """
    Executes an unscented Kalman filter on motion capture data

    INPUTS
      uk - uk.Obj - if None, data tracked naively
      y - N x Nd x Nm - motion capture trajectories in Cartesian coords

    OUTPUTS
      x - column for the estimated system state at each sample
    """
    if not hasattr(uk, 'viz'):
        uk.viz = 0
    # Number of samples, dimensions, features
    Ns,Nd,Nm = y.shape
    # If no UKF is specified, generate a default filter
    if uk == None or not hasattr(uk, 'filter'):
        uk.filter = nullFilter
    # problem dimensions
    uk.Ns = Ns; uk.Nd = Nd; uk.Nm = Nm
    # Collect initial observations
    p = y[0,:,:]
    # Run UKF on constant observation to obtain initial state
    for k in range(uk.Ninit):
        uk.filter(p.flatten(1).reshape(-1,1), mocap=p)
    # Allocate space for UKF output
    x = np.kron( np.ones((len(uk.x),Ns)), np.nan )
    # Loop through samples
    for k in range(Ns):
        # A-priori feature locations from UKF
        ap = uk.y0.reshape((Nd,Nm))
        # Feature observations for this sample
        p = y[k,:,:]
        # Execute unscented Kalman filter
        uk.filter(p.flatten(1).reshape(-1,1), mocap=p)
        # Store state from current UKF estimate
        x[:,k:k+1] = uk.x

    return x

def mocapCam(uk, y):
    """
    Executes an unscented Kalman filter on previously-captured camera data

    INPUTS
      uk - uk.Obj - if None, data tracked naively
      y - Nd x Nf x Nc x Ns - pixel observations for features in cameras
        Nd - dimensions
        Nf - features
        Nc - cameras
        Ns - samples

    OUTPUTS
      x - Nx x Ns - column for the estimated system state at each sample
    """
    if not hasattr(uk, 'viz'):
        uk.viz = 0
    if not hasattr(uk, 'vb'):
        uk.vb = 0
    if not hasattr(uk, 'Ninit'):
        uk.Ninit = 10

    def ukViz(fgi, axap, ap, axac, y):
        axap.set_xdata(ap[1,:])
        axap.set_ydata(ap[0,:])
        axac.set_xdata(y[1,:])
        axac.set_ydata(y[0,:])
        plt.draw()
        fgi.show()


    # Number of samples, dimensions, features, cameras
    Nd,Nf,Nc,Ns = y.shape

    # If no UKF is specified, generate a default filter
    if uk == None or not hasattr(uk, 'filter'):
        uk.filter = nullFilter

    uk.Nd = Nd
    uk.Nf = Nf
    uk.Nc = Nc
    uk.Ns = Ns

    # Collect initial observations
    p = y[...,0]
    ap = uk.y0.reshape((Nd,Nf,Nc),order='F')

    # Set up figures
    fgi  = []
    axap = []
    axac = []
    fguk = []
    if uk.viz:
        # UK visualization
        fguk = plt.figure(999)
        fguk.clf()
        plt.axes([0.,0.,1.,1.])

        # Video tracking visualizations
        for c in range(Nc):
            fgi.append(plt.figure(1000+c))
            fgi[-1].clf()
            plt.axes([0.,0.,1.,1.])
            axac.append(plt.plot(p[...,c][1,:],p[...,c][0,:],
                                 'k+',ms=10,mew=2)[0])
            axap.append(plt.plot(ap[...,c][1,:],ap[...,c][0,:],
                                 'rx',ms=10,mew=2)[0])
            X = y[0,:,c,:]
            Y = y[0,:,c,:]
            #xlim = np.array([X.min(),X.max()])
            #xlim += 0.1*np.diff(xlim)*np.array([-1,1])
            #ylim = np.array([Y.min(),Y.max()])
            #ylim += 0.1*np.diff(ylim)*np.array([-1,1])
            #plt.xlim(xlim)
            #plt.ylim(ylim)
            #plt.xticks(())
            #plt.yticks(())
            #plt.plot(xlim[[0,1,1,0,0]],ylim[[0,0,1,1,0]],'k',lw=2)

    # Run UKF on constant observation to obtain initial state
    for k in range(uk.Ninit):
        # A-priori feature locations from UKF
        ap = uk.y0.reshape((Nd,Nf,Nc),order='F')

        # Execute unscented Kalman filter
        uk.filter(p.flatten(1).reshape(-1,1))

        if uk.viz:
            for c in range(Nc):
                ukViz(fgi[c], axap[c], ap[...,c], axac[c], p[...,c])

        if uk.vb:
            print 'init #'+str(k)
            print '  x0  = '+str(np.around(uk.x.flatten(),2))
            print '  e0  = '+str(np.around((ap-p).flatten(1)))

    # Allocate space for UKF output
    x = np.kron( np.ones((len(uk.x),Ns)), np.nan )

    # Loop through samples
    for k in range(Ns):
        # A-priori feature locations from UKF
        ap = uk.y0.reshape((Nd,Nf,Nc),order='F')

        # Feature observations in each camera for this sample
        p = y[...,k]

        # Execute unscented Kalman filter
        uk.filter(p.flatten(1).reshape(-1,1))

        # Store state from current UKF estimate
        x[:,k:k+1] = uk.x

        if uk.viz:
            for c in range(Nc):
                ukViz(fgi[c], axap[c], ap[...,c], axac[c], p[...,c])

        if uk.vb:
            print 'track #'+str(k)
            print '  x0  = '+str(np.around(uk.x.flatten(),2))
            print '  e0  = '+str(np.around((ap-p).flatten(1)))

    return x

try:
  from vid import src
except:
  import sys; sys.exit(0)

def track(uk, srcs, fsrc, p0, wid=8):
    """
    p, x = uk.track  tracks features in videos using UKF

    INPUTS
      uk - uk.Obj - unscented kalman object
      srcs - list - framesource pipes / plugin chains
      fsrc - list - feature detection framesource pipe
      p0 - 2 x Nf x Nc - initial feature observations
      (optional)
      wid - scalar - feature size in px

    OUTPUTS
      p - 2 x Nf x Nc x N - feature observations in each sample
      x - Ns x N - UKF state in each sample

    """
       
    def getBox(img, pos, wid):
        """
        box = getBox  extracts square region from image

        INPUTS
          img - Nr x Nc x Nh - image as numpy array
          pos - 2-tuple - center of box in image (i.e. (row,col) coords)
          wid - int - square region will have dimension 2*wid+1
        """
        pos = np.array(pos.round(),dtype=int)
        if len(img.shape) == 2:
            box = np.zeros((2*wid+1,2*wid+1))
        else: 
            box = np.zeros((2*wid+1,2*wid+1,img.shape[2]))
        mr = np.round(max([pos[0]-wid,0]))
        Mr = np.round(min([pos[0]+wid+1,img.shape[0]]))
        mc = np.round(max([pos[1]-wid,0]))
        Mc = np.round(min([pos[1]+wid+1,img.shape[1]]))
        r = np.array(np.arange(mr,Mr),dtype=int)
        c = np.array(np.arange(mc,Mc),dtype=int)

        box[np.ix_(r-pos[0]+wid,c-pos[1]+wid)] = img[r,:][:,c]
        return box 

    def findFeature(img, fsrc, pos, wid, ax=[]):
        """
        p = findFeature  finds closest feature to given position

        INPUTS
          img - Nr x Nc x Nh - image as numpy array
          fsrc - list - framesource to aid feature detection
          pos - 2-tuple - feature position in image (i.e. (row,col) coords)
          wid - int - investigated region will have dimension 2*wid+1
          ax  - (imax, ptax) - results are updated on given axes
        """
        box = src.apply(fsrc, [getBox(img, pos, wid)])[0]

        # If there is no video data for observation
        if box.shape == (0,0):
            return np.nan*pos 

        lab, nlab = si.label(box)
        com = np.zeros((0,2))
        for k in range(nlab):
            com = np.vstack((com, 
                             si.center_of_mass(lab,labels=lab,index=k+1)))

        # If no objects were detected, return nan's
        if com.shape[0] == 0:
            #1/0
            return np.nan*pos

        d = com - np.kron(np.ones((com.shape[0],1)), np.array([wid,wid]))
        d = np.sqrt( (d**2).sum(1) )
        cm = com[d.argmin(),:]

        posc = np.array(pos)+np.array([cm[0],cm[1]])-wid

        if ax:
            ax[0].set_data(box)
            ax[1].set_xdata([cm[1]])
            ax[1].set_ydata([cm[0]])

        return posc

    def ukViz(fgi, axi, img, axap, ap, axac, y):
        axi.set_data(img)
        axap.set_xdata(ap[1,:])
        axap.set_ydata(ap[0,:])
        axac.set_xdata(y[1,:])
        axac.set_ydata(y[0,:])
        plt.draw()
        fgi.show()

    if not(hasattr(uk, 'viz')):
      uk.viz = 0

    # Number of dimensions, features, cameras
    Nd,Nf,Nc = p0.shape

    # Number of samples
    N = src.info(srcs[0]).N

    # If no UKF is specified, generate a default filter
    if uk == None or not hasattr(uk, 'filter'):
      uk.filter = nullFilter

    # Collect initial observations
    p = np.kron( np.ones((2,Nf,Nc,N+1)), np.nan)
    p[...,0] = p0

    ap0 = uk.y0.reshape((2,Nf,Nc),order='f')

    imgs = [src.getIm(sc,0) for sc in srcs]

    # Set up figures
    fgi  = []
    axi  = []
    axap = []
    axac = []
    axf  = []
    fguk = []
    axuk = []
    if uk.viz:
        # UK visualization
        fguk = plt.figure(999)
        fguk.clf()
        plt.axes([0,0,1,1])
        #axuk = plt.plot(

        # Video tracking visualizations
        for c in range(Nc):
            sz = src.info(srcs[c]).sz
            fgi.append(plt.figure(1000+c))
            fgi[-1].clf()
            plt.axes([0.,0.3,1.,0.7])
            #axi.append(plt.imshow(imgs[c],origin='upper'))
            axi.append(plt.imshow(imgs[c],interpolation='nearest'))
            axac.append(plt.plot(p0[...,c][1,:],p0[...,c][0,:],
                        's',ms=8,mfc='none',mew=1,mec='k')[0])
            axap.append(plt.plot(ap0[...,c][1,:],ap0[...,c][0,:],
                        'o',ms=5,mfc='none',mew=1,mec='g')[0])
            plt.xlim((1,sz[1]))
            plt.ylim((1,sz[0]))
            plt.xticks(())
            plt.yticks(())

            axf.append([])
            aw = 1./Nf
            for n in range(Nf):
                plt.axes([n*aw,0.,aw,0.3])
                axf[c].append([plt.imshow(np.random.rand(2*wid+1,2*wid+1),
                                          interpolation='nearest'),
                            plt.plot([wid],[wid],'k+',lw=10.0,ms=20.0,mew=4)[0]])
                plt.xlim((1,2*wid+1))
                plt.ylim((1,2*wid+1))
                plt.xticks(())
                plt.yticks(())

        
    # Run UKF on constant observation to obtain initial state
    for k in range(uk.Ninit):
        print 'init #'+str(k) 
         
        ap0 = uk.y0.reshape((2,Nf,Nc),order='f')

        # Handle feature overlap by setting lower-indexed feature to nan
        d = 2.
        for c in range(ap0.shape[2]):
            for i in range(ap0[...,c].shape[1]):
                for j in range(i+1,ap0[...,c].shape[1]):
                    if np.linalg.norm(p0[:,i,0] - p0[:,j,0]) < d:
                        p0[:,i,0] = np.nan

        if uk.viz:
            for c in range(Nc):
                ukViz(fgi[c], axi[c], imgs[c], axap[c], ap0[...,c], 
                      axac[c], p0[...,c])
        print '  x0  = '+str(np.around(uk.x.flatten(),2))
        #print '  p0  = '+str(np.around(p0.flatten(1)))
        #print '  ap0 = '+str(np.around(ap0.flatten(1)))
        print '  e0  = '+str(np.around((ap0-p0).flatten(1)))


        uk.filter(p0.flatten(1).reshape(-1,1))

        #1/0

    #1/0

    # allocate space for ukf output
    x = np.kron( np.ones((len(uk.x),N+1)), np.nan )
    x[:,0:1] = uk.x0

    # loop through samples
    for k in range(N):
        print 'track #'+str(k)
        
        # a-priori feature locations from ukf
        ap = uk.y0.reshape((2,Nf,Nc),order='f')

        # find feature locations in video
        y = np.kron( np.ones((2,Nf,Nc)), np.nan)
        imgs = [src.getIm(sc,k) for sc in srcs]
        #fimgs = [src.getIm(sc+fsrc,k) for sc in srcs]
        # Handle feature overlap by setting lower-indexed feature to nan
        d = 2.
        for c in range(Nc):
            for f in range(Nf):
                if uk.viz:
                    y[:,f,c] = findFeature(imgs[c], fsrc, ap[:,f,c], wid, axf[c][f])
                else:
                    y[:,f,c] = findFeature(imgs[c], fsrc, ap[:,f,c], wid)

                for g in range(f):
                    if np.linalg.norm(y[:,f,c] - y[:,g,c]) < d:
                        y[:,g,c] = np.nan

            if uk.viz:
                ukViz(fgi[c], axi[c], imgs[c], axap[c], ap[...,c], 
                      axac[c], y[...,c])

        print '  x  = '+str(np.around(uk.x.flatten(),2))
        #print '  p  = '+str(np.around(y.flatten(1)))
        #print '  ap = '+str(np.around(ap.flatten(1)))
        print '  e  = '+str(np.around((ap-y).flatten(1)))

        # feature observations for this sample
        p[...,k+1] = y

        # execute unscented kalman filter
        uk.filter(p[...,k+1].flatten(1).reshape(-1,1))

        # store state from current ukf estimate
        x[:,k+1:k+2] = uk.x

    return p, x

