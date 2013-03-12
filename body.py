
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

import uk
from vid import cam
from util import util,geom

class Mocap(uk.Obj):
    def __init__(self, x0, g,
                 Qd=( np.hstack( (np.array([1,1,1])*5e-2, 
                                  np.array([1,1,1])*1e+2) ) ),
                 Rs=10., viz=0, Ninit=50,
                 labels=['pitch','roll','yaw','$x$','$y$','$z$']):
        """
        Body(x0, g) creates rigid body uk object

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - 6 x 1 - initial rigid body state
             - [pitch, roll, yaw, x, y, z]
          g  - Nd x Nf - Nd-dimensional feature locations in body frame
             - found, for instance, using geom.fit()
        """
        # Unit timestep assumed; derivative states require simple rescalings
        self.dt = 1
        # Number of derivatives to track
        self.ord = int(len(x0) / 6.)
        # Number of dimensions in feature space
        self.Nd = g.shape[0]
        # Number of features
        self.Nf = g.shape[1]
        # 3D Feature locations
        self.g = g
        # Create uk object with simple dynamics and observation function
        uk.Obj.__init__(self, self.sys, 6*self.ord, self.obs, self.Nd*self.Nf)
        self.Q = np.diag(Qd[0:6*self.ord])
        # Measurement noise is uniform
        self.R = Rs*np.identity(self.Nd*self.Nf)
        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)
        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10
        # Initial state
        self.x = x0.reshape( (len(x0),1) )
        # Initial observation
        self.y0 = self.obs(self.x)
        # initialization
        self.Ninit = Ninit
        # plotting
        self.labels = labels
        self.viz = viz
        self.viz0 = 0
        if self.viz:
          from mpl_toolkits.mplot3d import Axes3D
          import matplotlib.pyplot as plt

          self.fig = plt.figure(49)
          plt.clf()
          
          self.plt = {'3d': {'ax':self.fig.add_subplot(121, projection='3d')},
                      '2d': {'ax':self.fig.add_subplot(122)} }
    def __repr__( self ):
        d = {}; keys = ['dt','ord','Nd','g','Q','R','Ninit']
        for k in self.__dict__.keys():
          if k in keys:
            d[k] = self.__dict__[k]
        return str(d)

    def sys(self, x, **U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = A x
        """
        oldx = x.copy()

        if self.ord == 1:
            # Update position: x_{n+1} = x_n
            x[0:6,:] = oldx[0:6,:]
        elif self.ord == 2:
            # Update position: x_{n+1} = x_n + v_n*dt
            x[0:6,:] = oldx[0:6,:] + oldx[6:12,:]*self.dt
        elif self.ord == 3:
            # Update position: x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2
            x[0:6,:] = ( oldx[0:6,:] + oldx[6:12,:]*self.dt 
                         + 0.5*oldx[12:18,:]*self.dt**2 )
            # Update velocity: v_{n+1} = v_n + a_n*dt
            x[6:12,:] = oldx[6:12,:] + oldx[12:18,:]*self.dt

        return x

    def obs(self, x, **d):
        """
        z = obs  transforms geometry features to world coordinates

        INPUTS
          x - 6 x N - N rigid body state hypotheses

        OUTPUTS
          z - Nd*Nf x N - geometry feature locations
        """
        # Initialize observation
        z = np.kron( np.ones((self.Nd*self.Nf,x.shape[1])), np.nan);

        # Loop through state hypotheses
        for k in range(x.shape[1]):
          # Transform geometry to world coordinate system
          y = cam.rigid(self.g, geom.euler(x[0:3,k]), x[3:6,k:k+1])
          z[:,k] = y.flatten(1)

        if hasattr(self,'viz0'):
          self.viz0 = self.viz0 + 1
        if hasattr(self,'viz') and self.viz and not(self.viz0 % self.viz):
          N = z.shape[1]
          
          if 'mocap' in d.keys():
            z0 = d['mocap']

          if not('ld' in self.plt['2d'].keys()): 
            self.plt['2d']['ld'] = [[] for p in range(self.Nf)]
          if not('ld' in self.plt['3d'].keys()): 
            self.plt['3d']['ld'] = [[] for p in range(self.Nf)]

          xlim = []; ylim = []; zlim = []

          ax2 = self.plt['2d']['ax']
          ax3 = self.plt['3d']['ax']

          for p in range(self.Nf):
            dx = np.vstack((z[self.Nd*p, 1:(N+1)/2],
                            z[self.Nd*p, np.zeros((N-1)/2,dtype=int)],
                            z[self.Nd*p, (N+1)/2:]))
            dy = np.vstack((z[self.Nd*p+1, 1:(N+1)/2],
                            z[self.Nd*p+1, np.zeros((N-1)/2,dtype=int)],
                            z[self.Nd*p+1, (N+1)/2:]))
            dz = np.vstack((z[self.Nd*p+2, 1:(N+1)/2],
                            z[self.Nd*p+2, np.zeros((N-1)/2,dtype=int)],
                            z[self.Nd*p+2, (N+1)/2:]))

            xlim.append([dx.min(),dx.max()])
            ylim.append([dy.min(),dy.max()])
            zlim.append([dz.min(),dz.max()])

            if not self.plt['2d']['ld'][p]: 
              self.plt['2d']['ld'][p] = ax2.plot(dx,dy,'.-',lw=2,ms=10)
              ax2.legend(self.labels,ncol=6)
              ax2.axis('equal')
              ax2.set_xlabel('$x$ (mm)');
              ax2.set_ylabel('$y$ (mm)');
            for k,l in enumerate(self.plt['2d']['ld'][p]):
              l.set_xdata(dx[:,k]); l.set_ydata(dy[:,k])

            if not self.plt['3d']['ld'][p]: 
              self.plt['3d']['ld'][p] = [ax3.plot(dx[:,k],dy[:,k],dz[:,k],'.-',lw=2,ms=10)[0] for k in range(dx.shape[1])]
              #ax3.axis('equal')
              ax3.set_xlabel('$x$ (mm)');
              ax3.set_ylabel('$y$ (mm)');
              ax3.set_zlabel('$z$ (mm)');
              ax3.view_init(elev=20.,azim=-130.)
            for k,l in enumerate(self.plt['3d']['ld'][p]):
              l.set_xdata(dx[:,k]); l.set_ydata(dy[:,k])
              l.set_3d_properties(dz[:,k])
              #ax3.auto_scale_xyz(dx[:,k],dz[:,k],dz[:,k])

          if not('ly0' in self.plt['2d'].keys()): 
            self.plt['2d']['ly0'], = ax2.plot(z0[0,:],z0[1,:],'rx',lw=2,ms=10)
          self.plt['2d']['ly0'].set_xdata(z0[0,:]); 
          self.plt['2d']['ly0'].set_ydata(z0[1,:])
          if not('lt' in self.plt['2d'].keys()): 
            bbox = dict(facecolor='white')
            self.plt['2d']['lt'] = [ax2.text(z0[0,k],z0[1,k],'%d'%k,bbox=bbox) for k in range(z0.shape[1])]
          for k in range(z0.shape[1]):
            if not( np.isnan(z0[0,k]) ):
              self.plt['2d']['lt'][k].set_x(z0[0,k]+10)
              self.plt['2d']['lt'][k].set_y(z0[1,k]+10)

          if not('ly0' in self.plt['3d'].keys()): 
            self.plt['3d']['ly0'], = ax3.plot(z0[0,:],z0[1,:],z0[2,:],'rx',lw=2,ms=10)
          self.plt['3d']['ly0'].set_xdata(z0[0,:]); 
          self.plt['3d']['ly0'].set_ydata(z0[1,:])
          self.plt['3d']['ly0'].set_3d_properties(z0[2,:])

          xlim = np.array(xlim); xlim = np.array([xlim[:,0].min(),xlim[:,1].max()])
          xlim = (xlim[0]-0.1*np.diff(xlim),xlim[1].max()+0.1*np.diff(xlim))
          ylim = np.array(ylim); ylim = np.array([ylim[:,0].min(),ylim[:,1].max()])
          ylim = (ylim[0]-0.1*np.diff(ylim),ylim[1].max()+0.2*np.diff(ylim))
          zlim = np.array(zlim); zlim = np.array([zlim[:,0].min(),zlim[:,1].max()])
          zlim = (zlim[0]-0.1*np.diff(zlim),zlim[1].max()+0.1*np.diff(zlim))

          ax2.set_xlim(xlim); ax2.set_ylim(ylim)
          ax3.set_xlim(xlim); ax3.set_ylim(ylim); ax3.set_zlim(zlim)

          #ax2.relim()
          #ax2.autoscale_view(True,True,True)
          #ax3.relim()

          ax2.set_title('%d'%self.viz0)
          self.fig.canvas.draw()

        return z

class Cam(uk.Obj):
    def __init__(self, x0, g, cams,
                       Qd=( np.hstack( (np.array([1,1,1])*5e-2, 
                                        np.array([1,1,1])*1e-3) ) ),
                       Rs=10.):
        """
        Body(x0, g, cams) creates rigid body uk object

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - 6 x 1 - initial rigid body state
             - [pitch, roll, yaw, x, y, z]
          g  - 3 x Nf - 3D feature locations in body frame
             - found, for instance, using geom.fit()
          cams - 3 x 4 x Nc - DLTs for each camera
               - list - camera dicts
        """
        # Unit timestep assumed; derivative states require simple rescalings
        self.dt = 1

        # Number of states to track
        self.Ns = int(len(x0) / 6.)

        # Number of features
        self.Nf = g.shape[1]

        # 3D Feature locations
        self.g = g

        if isinstance(cams,list):
            # Camera dicts
            self.cams = cams

            # Number of cameras
            self.Nc = len(cams)

        else:
            # DLTs for each camera
            self.dlt = cams

            # Number of cameras
            self.Nc = cams.shape[2]

        # Create uk object with simple dynamics and observation function
        uk.Obj.__init__(self, self.sys, 6*self.Ns, self.obs, 2*self.Nf*self.Nc)
        
        self.Q = np.diag(Qd[0:6*self.Ns])

        # Measurement noise is uniform
        self.R = Rs*np.identity(2*self.Nf*self.Nc)

        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)

        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10

        # Initial state
        self.x = x0.reshape( (len(x0),1) )

        # Initial observation
        self.y0 = self.obs(self.x)

    def sys(self, x, *U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = A x
        """
        oldx = x.copy()

        if self.Ns == 1:
            # Update position: x_{n+1} = x_n
            x[0:6,:] = oldx[0:6,:]
        elif self.Ns == 2:
            # Update position: x_{n+1} = x_n + v_n*dt
            x[0:6,:] = oldx[0:6,:] + oldx[6:12,:]*self.dt
        elif self.Ns == 3:
            # Update position: x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2
            x[0:6,:] = ( oldx[0:6,:] + oldx[6:12,:]*self.dt 
                         + 0.5*oldx[12:18,:]*self.dt**2 )
 
            # Update velocity: v_{n+1} = v_n + a_n*dt
            x[6:12,:] = oldx[6:12,:] + oldx[12:18,:]*self.dt

        return x

    def stateToPX(self, x):
        """
        y = stateToPX  applies camera models to obtain pixel observations

        INPUTS
          x - 6 x 1 - rigid body state

        OUTPUTS
          y - 2 x Nf x Nc - pixel observations in each camera
        """

        # Transform geometry to world coordinate system
        p = cam.rigid(self.g, geom.euler(x[0:3,0]), x[3:6,:])

        # Initialize pixel locations
        y = np.kron( np.ones((2,self.Nf,self.Nc)), np.nan)

        # Loop through cameras
        for c in range(self.Nc):
            # Apply DLT
            if hasattr(self,'dlt'):
                y[...,c] = cam.dlt(p, self.dlt[...,c])
            # Apply Zhang camera model
            else:
                if isinstance(self.cams[c],dict):
                    R = self.cams[c]['R']
                    t = self.cams[c]['t']
                    A = self.cams[c]['A']
                    d = self.cams[c]['d']
                else:
                    R = self.cams[c].R
                    t = self.cams[c].t
                    A = self.cams[c].A
                    d = self.cams[c].d
                z = cam.zhang(p, R, t, A, d)
                y[...,c] = np.dot(np.array([[0,1],[1,0]]),z)

        return y

    def obs(self, x, *U):
        """
        z = obs  applies camera models to obtain pixel observations

        INPUTS
          x - 6 x N - rigid body state

        OUTPUTS
          z - 2*Nf*Nc x N - pixel observations in each camera
        """
        # Initialize observation
        z = np.kron( np.ones((2*self.Nf*self.Nc,x.shape[1])), np.nan);

        # Loop through state hypotheses
        for k in range(x.shape[1]):
            # Apply camera model to obtain pixel coordinates
            y = self.stateToPX(x[:,k:k+1])
            z[:,k] = y.flatten(1)

        if hasattr(self,'viz') and self.viz:

            N = z.shape[1]

            self.fig = plt.figure(999)
            plt.clf()
            
            for c in range(self.Nc):
                plt.subplot(1,self.Nc,c+1)
                for p in range(self.Nf):
                    dx = np.vstack((z[c*self.Nc+2*p, 1:(N+1)/2],
                                    z[c*self.Nc+2*p, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p, (N+1)/2:]))
                    dy = np.vstack((z[c*self.Nc+2*p+1, 1:(N+1)/2],
                                    z[c*self.Nc+2*p+1, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p+1, (N+1)/2:]))

                    plt.plot(dx,dy,'o-',lw=2,ms=5,mfc='none',mew=1,mec='auto')
                    #plt.legend(self.labels)
                    plt.axis('equal')

            plt.draw()
            plt.legend(self.labels)
            self.fig.show()
            #1/0


        return z

class NoZ(uk.Obj):
    def __init__(self, x0, g, cams):
        """
        NoZ  creates rigid body uk object for single overhead camera

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - 6 x 1 - initial rigid body state
             - [pitch, roll, yaw, x, y, z]
          g  - 3 x Nf - 3D feature locations in body frame
             - found, for instance, using geom.fit()
          cams - 3 x 4 x Nc - DLTs for each camera
               - list - camera dicts
        """
        # Unit timestep assumed; derivative states require simple rescalings
        self.dt = 1

        # Number of states to track
        self.Ns = int(len(x0) / 6.)

        # Number of features
        self.Nf = g.shape[1]

        # 3D Feature locations
        self.g = g

        if isinstance(cams,list):
            # Camera dicts
            self.cams = cams

            # Number of cameras
            self.Nc = len(cams)

        else:
            # DLTs for each camera
            self.dlt = cams

            # Number of cameras
            self.Nc = cams.shape[2]

        # Create uk object with simple dynamics and observation function
        uk.Obj.__init__(self, self.sys, 6*self.Ns, self.obs, 2*self.Nf*self.Nc)
        
        # Define noise level for each state coordinate
        self.Q = ( np.hstack( (np.array([1,1,1])*0.1, 
                               np.array([1,1,1])*1.0) ) )
        self.Q = np.diag(self.Q[0:6*self.Ns])

        # Measurement noise is uniform
        self.R = 10*np.identity(2*self.Nf*self.Nc)

        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)

        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10

        # Initial state
        self.x = x0.reshape( (len(x0),1) )

        # Fixed z
        self.z0 = x0.flatten()[5]

        # Initial observation
        self.y0 = self.obs(self.x)

    def sys(self, x, *U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = A x
        """
        z = x.copy()

        if self.Ns == 1:
            # Update position: x_{n+1} = x_n
            z[0:6,:] = x[0:6,:]
        elif self.Ns == 2:
            # Update position: x_{n+1} = x_n + v_n*dt
            z[0:6,:] = x[0:6,:] + x[6:12,:]*self.dt
        elif self.Ns == 3:
            # Update position: x_{n+1} = x_n + v_n*dt + 0.5*a_n*dt^2
            z[0:6,:] = ( x[0:6,:] + x[6:12,:]*self.dt 
                         + 0.5*x[12:18,:]*self.dt**2 )
 
            # Update velocity: v_{n+1} = v_n + a_n*dt
            z[6:12,:] = x[6:12,:] + x[12:18,:]*self.dt

        # Fixed z coordinate
        #z[5,:] = self.z0

        return z

    def stateToPX(self, x):
        """
        y = stateToPX  applies camera models to obtain pixel observations

        INPUTS
          x - 6 x 1 - rigid body state

        OUTPUTS
          y - 2 x Nf x Nc - pixel observations in each camera
        """

        # Transform geometry to world coordinate system
        p = cam.rigid(self.g, geom.euler(x[0:3,0]), x[3:6,:])

        # Initialize pixel locations
        y = np.kron( np.ones((2,self.Nf,self.Nc)), np.nan)

        # Loop through cameras
        for c in range(self.Nc):
            # Apply DLT
            if hasattr(self,'dlt'):
                y[...,c] = cam.dlt(p, self.dlt[...,c])
            # Apply Zhang camera model
            else:
                if isinstance(self.cams[c],dict):
                    R = self.cams[c]['R']
                    t = self.cams[c]['t']
                    A = self.cams[c]['A']
                    d = self.cams[c]['d']
                else:
                    R = self.cams[c].R
                    t = self.cams[c].t
                    A = self.cams[c].A
                    d = self.cams[c].d
                z = cam.zhang(p, R, t, A, d)
                y[...,c] = np.dot(np.array([[0,1],[1,0]]),z)

        return y

    def obs(self, x, *U):
        """
        z = obs  applies camera models to obtain pixel observations

        INPUTS
          x - 6 x N - rigid body state

        OUTPUTS
          z - 2*Nf*Nc x N - pixel observations in each camera
        """
        # Initialize observation
        z = np.kron( np.ones((2*self.Nf*self.Nc,x.shape[1])), np.nan);

        # Loop through state hypotheses
        for k in range(x.shape[1]):
            # Apply camera model to obtain pixel coordinates
            y = self.stateToPX(x[:,k:k+1])
            z[:,k] = y.flatten(1)

        if hasattr(self,'viz') and self.viz:

            N = z.shape[1]

            self.fig = plt.figure(999)
            plt.clf()
            
            for c in range(self.Nc):
                plt.subplot(1,self.Nc,c+1)
                for p in range(self.Nf):
                    dx = np.vstack((z[c*self.Nc+2*p, 1:(N+1)/2],
                                    z[c*self.Nc+2*p, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p, (N+1)/2:]))
                    dy = np.vstack((z[c*self.Nc+2*p+1, 1:(N+1)/2],
                                    z[c*self.Nc+2*p+1, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p+1, (N+1)/2:]))

                    plt.plot(dx,dy,'o-',lw=2,ms=5,mfc='none',mew=1,mec='auto')
                    #plt.legend(self.labels)
                    plt.axis('equal')

            plt.draw()
            plt.legend(self.labels)
            self.fig.show()
            #1/0


        return z

def rigid(p,r,r2R=geom.euler):
    """
    q = rigid  apply rigid transformation to points

    INPUTS
      p - 3 x Nf - 3D points to transform
      r - 6 x N  - rigid body state
      (optional)
      r2R - function - transforms rotation vec (3x1) to rotation mat (3x3)
        e.g. geom.euler  or  cva.rodrigues

    OUTPUTS
      q - 3 x Nf x N - transformed 3D points
    """
    Nf = p.shape[1]
    N  = r.shape[1]
    q = []
    for n in range(N):
        R = r2R(r[0:3,n:n+1])
        t = r[3:6,n:n+1]
        q.append(cam.rigid(p,R,t))

    return np.dstack(q)

if __name__ == '__main__':

  import scipy as sp
  import scipy.signal as sig

  from uk import pts, body

  # generate 3D trajectory
  N  = 100
  Np = 1
  x0 = np.array([[0.,0.,0.,0.,0.,500.]]).T
  xs = np.diag(np.hstack((np.array([1,1,1])*1.,np.array([1,1,1])*10.)))
  x  = np.cumsum(np.dot(xs,np.random.randn(6,N)),axis=1) + np.kron(np.ones((1,N)),x0)

  g = 10.*np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype=float).T

  # smooth trajectory
  B,A = sig.butter(1, 0.05)
  x = sig.filtfilt(B,A, x)

  th = np.random.rand()*2*np.pi
  om = np.array([[0,0,1],[0,0,0],[-1,0,0]])
  R = sp.linalg.expm(th*om)
  t = np.dot(R,-x0[3:6,:])+x0[3:6,:]
  A = np.array([[2000.,0,0],[0,2000.,0],[0,0,1.]])
  d = np.array([0.,0.,0.,0.])

  cams = [{'R': np.identity(3), 't': np.zeros((3,1)), 'A': A, 'd': d},
          {'R': R, 't': t, 'A': A, 'd': d}]

  # initialize UKF
  ukf = body.Cam(x0, g, cams)
  ukf.Ninit = 10
  ukf.viz = 0

  Q = np.hstack((np.array([1,1,1])*0.01,np.array([1,1,1])*0.5))
  ukf.Q = np.diag(Q.flatten())
  ukf.R = 1.*np.identity(2*ukf.Nf*ukf.Nc)
  ukf.Rm = ukf.R * (10-1)
  ukf.C = ukf.Q * 10

  # apply camera models to obtain pixel observations
  p = np.concatenate([ukf.stateToPX(xx.reshape((6,1))).reshape((2,ukf.Nf,ukf.Nc,1)) for xx in x.T],axis=3)

  # add noise to pixel observations
  sg = 4.
  e = sg*np.random.randn(*p.shape)
  pe = p+e

  # run UKF
  z = uk.mocapCam(ukf, pe)

  plt.figure(1)
  plt.clf()
  col = ['b','g','r']

  plt.subplot(2,1,1)
  xlbl = ['$\\theta_x$','$\\theta_y$','$\\theta_z$','$x$','$y$','$z$']
  zlbl = ['$\hat{\\theta_x}$','$\hat{\\theta_y}$','$\hat{\\theta_z}$',
          '$\hat{x}$','$\hat{y}$','$\hat{z}$']
  for j,c in zip([0,1,2],col):
    plt.plot(x[j,:]-x0[j],c+'--',lw=3.,label=xlbl[j])
    plt.plot(z[j,:]-x0[j],c,lw=1.,label=zlbl[j])
  plt.legend()

  plt.subplot(2,1,2)
  for j,c in zip([3,4,5],col):
    plt.plot(x[j,:]-x0[j],c+'--',lw=3.,label=xlbl[j])
    plt.plot(z[j,:]-x0[j],c,lw=1.,label=zlbl[j])
  plt.legend()

