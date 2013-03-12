
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

class Mocap(uk.Obj):

    def __repr__(self):
        return 'uk.pt  Np='+str(self.Np)+' x='+str(self.x.flatten())

    def __init__(self, x0, Qs=2., Rs=10.,
                       labels=['$x$','$y$','$z$'],viz=False):
        """
        pt = Pos  track position of points using Nd motion capture data

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - Nd x Np - initial state;
             Np determines number of points to track
          Qs - scalar - scale for state uncertainty
          Rs - scalar - scale for measurement uncertainty

        """
        # Unit timestep assumed; velocities require simple rescalings
        self.dt = 1
        # Number of points
        self.Nd,self.Np = x0.shape
        # Create uk object with simple dynamics and observation function
        uk.Obj.__init__(self, self.sys, self.Nd*self.Np, 
                              self.obs, self.Nd*self.Np)
        # Define noise level for each state coordinate
        self.Q = Qs*np.ones(self.Nd)
        self.Q = np.diag(np.kron(np.ones((1,self.Np)),self.Q[0:self.Nd]).flatten())
        # Measurement noise is uniform
        self.R = Rs*np.identity(self.Nd*self.Np)
        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)
        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10
        # Initial state
        self.x = x0.T.reshape( (self.Nd*self.Np,1) )
        # Initial observation
        self.y0 = self.obs(self.x)
        # plotting
        self.labels = labels
        self.viz = viz
        if self.viz:
          from mpl_toolkits.mplot3d import Axes3D
          import matplotlib.pyplot as plt

          self.fig = plt.figure(49)
          plt.clf()
          
          self.plt = {'3d': {'ax':self.fig.add_subplot(121, projection='3d')},
                      '2d': {'ax':self.fig.add_subplot(122)} }

    def sys(self, x, **U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = x
        """
        return x

    def obs(self, x, **d):
        """
        z = obs  applies camera models to obtain pixel observations

        INPUTS
          x - Nd*Np x N - point states
        """
        # Initialize observation
        z = np.kron( np.ones((self.Nd*self.Np,x.shape[1])), np.nan);

        # Loop through state hypotheses
        for k in range(x.shape[1]):
          # Apply camera model to obtain pixel observations
          y = x[:,k:k+1]
          z[:,k] = y.flatten(1)

        if hasattr(self,'viz') and self.viz:
          N = z.shape[1]
          
          if 'mocap' in d.keys():
            z0 = d['mocap']

          if not('ld' in self.plt['2d'].keys()): 
            self.plt['2d']['ld'] = [[] for p in range(self.Np)]
          if not('ld' in self.plt['3d'].keys()): 
            self.plt['3d']['ld'] = [[] for p in range(self.Np)]

          xlim = []; ylim = []; zlim = []

          ax2 = self.plt['2d']['ax']
          if not('ly0' in self.plt['2d'].keys()): 
            self.plt['2d']['ly0'], = ax2.plot(z0[0,:],z0[1,:],'rx',ms=20)
          self.plt['2d']['ly0'].set_xdata(z0[0,:]); 
          self.plt['2d']['ly0'].set_ydata(z0[1,:])

          ax3 = self.plt['3d']['ax']
          if not('ly0' in self.plt['3d'].keys()): 
            self.plt['3d']['ly0'], = ax3.plot(z0[0,:],z0[1,:],z0[2,:],'rx',ms=20)
          self.plt['3d']['ly0'].set_xdata(z0[0,:]); 
          self.plt['3d']['ly0'].set_ydata(z0[1,:])
          self.plt['3d']['ly0'].set_3d_properties(z0[2,:])

          for p in range(self.Np):
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
              #ax.legend(self.labels)
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

          xlim = np.array(xlim); xlim = np.array([xlim[:,0].min(),xlim[:,1].max()])
          xlim = (xlim[0]-0.1*np.diff(xlim),xlim[1].max()+0.1*np.diff(xlim))
          ylim = np.array(ylim); ylim = np.array([ylim[:,0].min(),ylim[:,1].max()])
          ylim = (ylim[0]-0.1*np.diff(ylim),ylim[1].max()+0.1*np.diff(ylim))
          zlim = np.array(zlim); zlim = np.array([zlim[:,0].min(),zlim[:,1].max()])
          zlim = (zlim[0]-0.1*np.diff(zlim),zlim[1].max()+0.1*np.diff(zlim))

          ax2.set_xlim(xlim); ax2.set_ylim(ylim)
          ax3.set_xlim(xlim); ax3.set_ylim(ylim); ax3.set_zlim(zlim)

          #ax2.relim()
          #ax2.autoscale_view(True,True,True)
          #ax3.relim()

          self.fig.canvas.draw()

        return z

class Pos(uk.Obj):

    def __repr__(self):
        return 'uk.pt  Np='+str(self.Np)+' Nc='+str(self.Nc)+' x='+str(self.x.flatten())

    def __init__(self, x0, cams, Qs=2., Rs=10.,
                       labels=['$x$','$y$','$z$']):
        """
        pt = Pos  track position of 3D points using multiple camera views

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - 3 x Np - initial 3D state;
             Np determines number of 3D points to track
          cams - list of dicts - camera transformation parameters
            A - 3 x 3 x Nc - camera matrices
            d - 1 x 4 x Nc - distortion parameters
            R - 3 x 3 x Nc - rotation matrix
            t - 3 x 1 x Nc - translation
          Qs - scalar - scale for state uncertainty
          Rs - scalar - scale for measurement uncertainty

        """
        # Unit timestep assumed; velocities require simple rescalings
        self.dt = 1

        # Number of points
        self.Np = x0.shape[1]

        # Number of cameras
        self.Nc = len(cams)

        # Camera transformations for each camera
        self.cams = cams 

        # Create uk object with simple dynamics and observation function
        uk.Obj.__init__(self, self.sys, 3*self.Np, 
                              self.obs, 2*self.Np*self.Nc)
        
        # Define noise level for each state coordinate
        self.Q = Qs*np.array([1,1,1])
        self.Q = np.diag(np.kron(np.ones((1,self.Np)),self.Q[0:3]).flatten())

        # Measurement noise is uniform
        self.R = Rs*np.identity(2*self.Np*self.Nc)

        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)

        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10

        # Initial state
        self.x = x0.T.reshape( (3*self.Np,1) )

        self.labels = labels

        # Initial observation
        self.y0 = self.obs(self.x)

    def stateToPX(self, x): 
        """
        y = stateToPX  applies camera models to obtain pixel observations

        INPUTS
          x - 3*Np x 1 - 3D positions of points

        OUTPUTS
          y - 2 x Np x Nc - 2D camera observations of points
        """
        # Extract 3D positions of points
        p = x.reshape((self.Np,3)).T

        # Initialize pixel locations
        y = np.kron( np.ones((2,self.Np,self.Nc)), np.nan)

        for c,ca in enumerate(self.cams):
            R = ca['R']
            t = ca['t']
            A = ca['A']
            d = ca['d']
            z = cam.zhang(p, R, t, A, d)
            #y[...,c] = (np.dot(np.array([[0,-1],[-1,0]]),z)
            #            + np.kron(np.ones((1,z.shape[1])),A[[1,0],2:3]))
            y[...,c] = np.dot(np.array([[0,1],[1,0]]),z)

        return y

    def sys(self, x, *U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = x
        """
        return x

    def obs(self, x, *U):
        """
        z = obs  applies camera models to obtain pixel observations

        INPUTS
          x - 3*No*Np x N - point states
        """
        # Initialize observation
        z = np.kron( np.ones((2*self.Np*self.Nc,x.shape[1])), np.nan);

        # Loop through state hypotheses
        for k in range(x.shape[1]):
            # Apply camera model to obtain pixel observations
            y = self.stateToPX(x[:,k:k+1])
            z[:,k] = y.flatten(1)

        if hasattr(self,'viz') and self.viz:

            N = z.shape[1]

            self.fig = plt.figure(999)
            plt.clf()
            
            for c in range(self.Nc):
                plt.subplot(1,self.Nc,c+1)
                for p in range(self.Np):
                    dx = np.vstack((z[c*self.Nc+2*p, 1:(N+1)/2],
                                    z[c*self.Nc+2*p, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p, (N+1)/2:]))
                    dy = np.vstack((z[c*self.Nc+2*p+1, 1:(N+1)/2],
                                    z[c*self.Nc+2*p+1, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p+1, (N+1)/2:]))

                    plt.plot(dx,dy,'.-',lw=2,ms=20)
                    plt.legend(self.labels)
                    plt.axis('equal')

            plt.draw()
            self.fig.show()
            #1/0

        return z

class Vel(uk.Obj):

    def __repr__(self):
        return 'uk.pt  Np='+str(self.Np)+' Nc='+str(self.Nc)+' x='+str(self.x.flatten())

    def __init__(self, x0, cams, labels=None):
        """
        vel = Vel  track pos & vel of 3D points using multiple camera views

        Refer to uk.Obj documentation for more details

        INPUTS
          x0 - 3*No x Np - initial 3D state;
             Np - # of 3D points to track
             No - order of dynamics (i.e. # of derivatives to track)
          cams - list of Nc dicts - camera transformation parameters
            A - 3 x 3 - camera matrices
            d - 1 x 4 - distortion parameters
            R - 3 x 3 - rotation matrix
            t - 3 x 1 - translation
        """
        self.dt = 1

        self.Np = x0.shape[1]
        self.No = x0.shape[0]/3
        self.Nc = len(cams)

        self.cams = cams 

        uk.Obj.__init__(self, self.sys, 3*self.Np*self.No, 
                              self.obs, 2*self.Np*self.Nc)
        
        # Define noise level for each state coordinate
        Qp = np.array([1,1,1])*1.0
        Qv = np.array([1,1,1])*1.0
        self.Q = np.diag(
                   np.hstack((
                     np.kron(np.ones((1,self.Np)),Qp).flatten(),
                     np.kron(np.ones((1,self.Np)),Qv).flatten()
                   ))
                 )

        # Measurement noise is uniform
        self.R = 10*np.identity(2*self.Np*self.Nc)

        # Missing measurements are 10 times less certain
        self.Rm = self.R * (10-1)

        # Initial covariance is 10 times less certain than system
        self.C = self.Q * 10

        # Initial state
        self.x = x0.T.reshape( (3*self.Np*self.No,1) )

        if labels:
            self.labels = labels
        else:
            self.labels = ['$x$','$y$','$z$']

        # Initial observation
        self.y0 = self.obs(self.x)

    def stateToPX(self, x): 
        """
        y = stateToPX  applies camera models to obtain pixel observations

        INPUTS
          x - 3*Np x 1 - 3D positions of points

        OUTPUTS
          y - 2 x Np x Nc - 2D camera observations of points
        """
        # Extract 3D positions of points
        p = x[0:3*self.Np,:].reshape((self.Np,3)).T

        # Initialize pixel locations
        y = np.kron( np.ones((2,self.Np,self.Nc)), np.nan)

        for c,ca in enumerate(self.cams):
            if isinstance(ca,dict):
                R = ca['R']
                t = ca['t']
                A = ca['A']
                d = ca['d']
            else:
                R = ca.R
                t = ca.t
                A = ca.A
                d = ca.d
            z = cam.zhang(p, R, t, A, d)
            #y[...,c] = (np.dot(np.array([[0,-1],[-1,0]]),z)
            #            + np.kron(np.ones((1,z.shape[1])),A[[1,0],2:3]))
            y[...,c] = np.dot(np.array([[0,1],[1,0]]),z)

        return y

    def sys(self, x, *U):
        """
        x = sys  implements discrete-time dynamical system
                 x' = A x
        """
        z = x.copy()

        N = 3*self.Np

        if self.No == 1:
            z[0:N,:] = x[0:N,:]
        elif self.No == 2:
            z[0:N,:] = x[0:N,:]+x[N:2*N,:]*self.dt
        elif self.No == 3:
            z[0:N,:] = x[0:N,:]+x[N:2*N,:]*self.dt+x[2*N:3*N,:]*self.dt**2
            z[N:2*N,:] = x[N:2*N,:]+x[2*N:3*N,:]*self.dt

        return z

    def obs(self, x, *U):
        """
        z = obs  applies camera models to obtain pixel observations

        INPUTS
          x - 3*Np*No x N - point states
        """
        # Initialize observation
        z = np.kron( np.ones((2*self.Np*self.Nc,x.shape[1])), np.nan);

        N = 3*self.Np

        # Loop through state hypotheses
        for k in range(x.shape[1]):
            # Apply camera model to obtain pixel observations
            y = self.stateToPX(x[0:N,k:k+1])
            z[:,k] = y.flatten(1)

        if hasattr(self,'viz') and self.viz:

            N = z.shape[1]

            self.fig = plt.figure(999)
            plt.clf()
            
            for c in range(self.Nc):
                plt.subplot(1,self.Nc,c+1)
                for p in range(self.Np):
                    dx = np.vstack((z[c*self.Nc+2*p, 1:(N+1)/2],
                                    z[c*self.Nc+2*p, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p, (N+1)/2:]))
                    dy = np.vstack((z[c*self.Nc+2*p+1, 1:(N+1)/2],
                                    z[c*self.Nc+2*p+1, np.zeros((N-1)/2,dtype=int)],
                                    z[c*self.Nc+2*p+1, (N+1)/2:]))

                    plt.plot(dx,dy,'.-',lw=2,ms=20)
                    plt.legend(self.labels)
                    plt.axis('equal')

            plt.draw()
            self.fig.show()
            #1/0

        return z

if __name__ == '__main__':

  import scipy as sp
  import scipy.signal as sig

  from vid import cam
  import uk
  from uk import pts
  import util
  from util import geom

  # Generate 3D trajectory
  N  = 100
  Np = 1
  x0 = np.array([[0.,0.,1000.]]).T
  dx = 10.
  x  = np.cumsum(dx*np.random.randn(3,N),axis=1) + np.kron(np.ones((1,N)),x0)

  # Smooth trajectory
  B,A = sig.butter(1, 0.05)
  x = sig.filtfilt(B,A, x)

  th = np.pi/8
  om = np.array([[0,0,1],[0,0,0],[-1,0,0]])
  R = sp.linalg.expm(th*om)
  t = np.dot(R,-x0)+x0
  A = np.array([[2000.,0,0],[0,2000.,0],[0,0,1.]])
  d = np.array([0.,0.,0.,0.])

  cams = [{'R': np.identity(3), 't': np.zeros((3,1)), 'A': A, 'd': d},
          {'R': R, 't': t, 'A': A, 'd': d}]

  ukf = pts.Pos(np.array([[0,0,100]]).T, cams)
  ukf.Ninit = 1
  ukf.viz = 0 

  Q = np.array([1,1,1])*0.05*dx
  ukf.Q = np.diag(np.kron(np.ones((1,Np)),Q).flatten())
  ukf.R = 1*np.identity(2*ukf.Np*ukf.Nc)
  ukf.Rm = ukf.R * (10-1)
  ukf.C = ukf.Q * 10


  # Apply camera models to obtain pixel observations
  p = np.concatenate([ukf.stateToPX(xx.reshape((3,1))).reshape((2,ukf.Np,ukf.Nc,1)) for xx in x.T],axis=3)

  # Add noise to pixel observations
  sig = 20.
  e = sig*np.random.randn(*p.shape)
  pe = p+e

  z = uk.mocapCam(ukf, pe)

  plt.figure(1)
  plt.clf()
  col = ['b','g','r']
  dy = dx*np.sqrt(N)
  for s,c in zip([0,1,2],col):
      plt.subplot(3,1,s+1)
      plt.plot(x[s,:],c+'-',lw=2.)
      plt.plot(z[s,:],c+'.',lw=2.)

  plt.figure(2)
  plt.clf()
  col = ['b','g']
  M = len(cams)
  for m,ca in enumerate(cams):
      qx = cam.zhang(x, ca['R'], ca['t'], ca['A'], ca['d'])
      qz = cam.zhang(z,  ca['R'], ca['t'], ca['A'], ca['d'])
      for s,c in zip([0,1],col):
          plt.subplot(M,2,m*M+s+1)
          plt.plot(qx[s,:],c+'-',lw=2.)
          plt.plot(qz[s,:],c+'.',lw=2.)

