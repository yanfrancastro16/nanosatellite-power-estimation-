import numpy as np
#-----------------------------------------------------------------------------------------------------
"""
input parameters of the orbit_function:
A (semi-major axis , from the surface of the earth) [km]
e (eccentricity)
I (orbit inclination) [degrees]
O (right ascension) [degrees]
W (perigee argument) [degrees]
Theta (true anomaly) [degrees]
"""
#-----------------------------------------------------------------------------------------------------
def orbit_function(A,e,I,O,W,Theta):
    # mean radius of the earth
    Re = 6378
    # gravitational constant
    u = 3.986E5
    # sideral day of earth (approximate of 24 hours = 86400 seconds)
    sd = 86400
    # conversion to minutes
    to_minutes = 60
    #-------------------------------------------------------------------------------------------------
    # conversion of degrees to radians of the angles of the classical orbital elements
    w , o , i , thetao = np.radians(W) , np.radians(O) , np.radians(I) , np.radians(Theta)
    #-------------------------------------------------------------------------------------------------
    # semi-major axis of the orbit + mean radius of the earth
    a = A+Re
    # semilatus rectum of the orbit
    p = a*(1-e**(2))
    # specific angular momentum of the orbit
    h = np.sqrt(u*p)
    # spesific energy of the orbit
    Ener = (-u/(2*a))
    # semi-minor axis of the orbit
    b = a*np.sqrt(1-e**(2))
    # perigee 
    rperi = a*(1-e)
    # apogee
    rapo = a*(1+e)
    # orbital period
    T = (((2*np.pi)/(np.sqrt(u)))*(a)**(3/2))
    # initial eccentric anomaly
    E = 2*np.arctan((np.sqrt((1-e)/(1+e))*np.tan(thetao/2)))
    # initial mean anomaly
    Mo = E-e*np.sin(E)
    # initial time of the nanosatellite
    to = ((Mo/(2*np.pi))*T)
    # number of revolutions of the nanosatellite per one earth rotation (sideral day)
    nr = round(sd/T)
    #--------------------------------------------------------------------------------------------------
    # time array of the nanosatellite
    t = np.linspace(to,to+T,101)
    # size of the time array of nanosatellite
    lt = np.size(t)
    #--------------------------------------------------------------------------------------------------
    # Euler rotation matrices
    # first rotation matrix through the perigee argument
    Rw = np.array([[np.cos(w),np.sin(w),0],
                   [-np.sin(w),np.cos(w),0],
                   [0,0,1]])
    # second rotation matrix through the orbit inclination
    Ri = np.array([[1,0,0],
                   [0,np.cos(i),np.sin(i)],
                   [0,-np.sin(i),np.cos(i)]])
    # third rotation matrix through the right ascencion
    Ro = np.array([[np.cos(o),np.sin(o),0],
                   [-np.sin(o),np.cos(o),0],
                   [0,0,1]])
    # conversion matrix from the PQW reference system to the ECI reference system
    R = np.transpose((Rw@Ri@Ro))
    #---------------------------------------------------------------------------------------------------
    # initial true anomaly array
    theta = np.zeros(lt)
    # initial arrays of the components of the nanosatellite position vector in the ECI reference system
    rx , ry , rz = np.zeros(lt) , np.zeros(lt) , np.zeros(lt)
    # initial arrays of the components of the nanosatellite velocity vector in the ECI reference system
    vx , vy , vz= np.zeros(lt) , np.zeros(lt) , np.zeros(lt)
    #---------------------------------------------------------------------------------------------------
    # initial error and tolerance values of the Newton-Raphsom algorithm to solve de kepler equation
    error , tol = np.inf, 1E-3
    # for each value of time array, we calculate the nanosatellite position and velocity vectors 
    for n in range(lt):
        # begining of the Newton-Raphson algorithm
        # estimate the initial value of the eccenctric anomaly
        M = ((2*np.pi)/T)*t[n] # mean anomaly 
        if(M < np.pi): 
            Eo = M+e/2
        elif(M > np.pi):
            Eo = M-e/2
        while error > tol: # iteration loop of the algorithm
            f = Eo-e*np.sin(Eo)-M # function f(E) in terms of eccentric anomaly
            df = 1-e*np.cos(Eo) # derivative of the function f'(E) in terms of eccentric anomaly
            En = Eo-(f/df) # iteration value of the eccentric anomaly
            error = abs(f/df) # approximation error of the algorithm
            Eo = En # update the eccentric anomaly value to the next iteration of the algorithm
        # end of the Newton-Raphson algorithm
        # we calculate the true anomaly values from each eccentric anomaly value obtained from the Newton-Raphson algorithm
        theta[n] = 2*np.arctan(np.sqrt((1+e)/(1-e))*np.tan(Eo/2))
        # magnitude of the position vector of the nanosatellite in the PQW reference system
        ro = p/(1+e*np.cos(theta[n]))
        # components of the position vector of the nanosatellite in the PQW reference system
        xp , yp = ro*np.cos(theta[n]) , ro*np.sin(theta[n])
        # components of the velocity vector of the nanosatellite in the PQW reference system
        vxp , vyp = (-u/h)*np.sin(theta[n]) , (u/h)*(e+np.cos(theta[n]))
        # position matrix in the PQW reference system
        rp = np.array([[xp,0,0],
                       [yp,0,0],
                       [0,0,0]])
        # velocity matrix in the PQW reference system
        vp = np.array([[vxp,0,0],
                       [vyp,0,0],
                       [0,0,0]])
        # position vector of the nanosatellite in the ECI reference system
        rxyz = R@rp
        rx[n] , ry[n] , rz[n] = rxyz[0,0] , rxyz[1,0] , rxyz[2,0]
        # position matrix of the nanosatellite in the ECI reference system
        r_ECI = np.transpose(np.array([rx,ry,rz,np.sqrt(rx**2+ry**2+rz**2)]))
        # velocity vector of the nanosatellite in the ECI reference system
        vxyz = R@vp
        vx[n] , vy[n] , vz[n] = vxyz[0,0] , vxyz[1,0] , vxyz[2,0]
        # velocity matrix of the nanosatellite in the ECI reference system
        v_ECI = np.transpose(np.array([vx,vy,vz,np.sqrt(vx**2+vy**2+vz**2)]))
        # end for loop
    # physical dimention of the orbit
    dimen = np.array([h,Ener,b,p,rapo,rperi,T/to_minutes,nr])
    # return parameters of the function
    return r_ECI , v_ECI , dimen , R , theta , t