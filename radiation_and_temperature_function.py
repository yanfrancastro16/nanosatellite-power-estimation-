import numpy as np
#--------------------------------------------------------------------------------
"""
input parameters of the radiation_and_temperature_function:
r_ECI (position matrix of the nanosatellite in the ECI reference system) [km/s]
rs_ECI (solar vector matrix in the ECI reference system) [UA]
theta (true anomaly array) [radians]
R (conversion matrix from PQW system to ECI system)
nJD (number of Julian date array)
t (time array) [seconds]
"""
#--------------------------------------------------------------------------------
def radiation_and_temperature_function(r_ECI,rs_ECI,theta,R,nJD,t):
    # mean radius of the earth
    Re = 6378 
    # solar flux
    Gs = 1367
    # solar flux from the earth
    Ge = 237
    # conversion of astronomical unit to kilometers
    au_to_km = 150E6
    # Boltzmann constant
    k = 5.67E-8
    # conversion to kelvin
    to_kelvin = 273.15
    #---------------------------------------------------------------------------
    # size of time array
    lt = np.size(t)
    # size of number of Julian date array
    lnJD = np.size(nJD)
    #---------------------------------------------------------------------------
    # initial orientation matrices of the six axis of the nanosatellite in the PQW reference system 
    # orientation matrix in the side no1
    no1 = np.array([[1,0,0],
                    [0,0,0],
                    [0,0,0]])
    # orientation matrix in the side no2
    no2 = np.array([[-1,0,0],
                    [0,0,0],
                    [0,0,0]])
    # orientation matrix in the side no3
    no3 = np.array([[0,0,0],
                    [0,1,0],
                    [0,0,0]]) 
    # orientation matrix in the side no4
    no4 = np.array([[0,0,0],
                    [0,-1,0],
                    [0,0,0]]) 
    # orientation matrix in the side no5
    no5 = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,1]]) 
    # orientation matrix in the side no6
    no6 = np.array([[0,0,0],
                    [0,0,0],
                    [0,0,-1]])
    #---------------------------------------------------------------------------
    # initial arrays of the components of the unit vector of the six axis of the nanosatellite in the ECI reference system  
    nx1 , ny1 , nz1 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N1 
    nx2 , ny2 , nz2 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N2 
    nx3 , ny3 , nz3 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N3 
    nx4 , ny4 , nz4 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N4 
    nx5 , ny5 , nz5 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N5 
    nx6 , ny6 , nz6 = np.zeros(lt) , np.zeros(lt) , np.zeros(lt) # components of the unit vector side N6 
    #----------------------------------------------------------------------------
    # initial radiation matrices for each side of the nanosatellite in the ECI reference system
    # direct solar radiation matrices for each side (N1 , N2 , N3 , N4 , N5 , N6)
    Q_solar_N1 , Q_solar_N2 , Q_solar_N3 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    Q_solar_N4 , Q_solar_N5 , Q_solar_N6 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    
    # albedo radiation matrices for each side (N1 , N2 , N3 , N4 , N5 , N6)
    Q_albe_N1 , Q_albe_N2 , Q_albe_N3 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    Q_albe_N4 , Q_albe_N5 , Q_albe_N6 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    
    # infrared radiation matrices for each side (N1 , N2 , N3 , N4 , N5 , N6)
    Q_ir_N1 , Q_ir_N2 , Q_ir_N3 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    Q_ir_N4 , Q_ir_N5 , Q_ir_N6 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    
    # total radiation matrices (solar + albedo + infrared) for each side (N1 , N2 , N3 , N4 , N5 , N6)
    Qt_N1 , Qt_N2 , Qt_N3 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    Qt_N4 , Qt_N5 , Qt_N6 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    #----------------------------------------------------------------------------
    # initial temperature matrices for each side of the nanosatellite in the ECI reference system
    Tt_N1 , Tt_N2 , Tt_N3 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    Tt_N4 , Tt_N5 , Tt_N6 = np.zeros((lnJD,lt)) , np.zeros((lnJD,lt)) , np.zeros((lnJD,lt))
    #----------------------------------------------------------------------------
    # taking the values of the nanosatellite position and solar vector matrices
    rsx , rsy, rsz = rs_ECI[:,0] , rs_ECI[:,1] , rs_ECI[:,2]
    rx , ry , rz = r_ECI[:,0] , r_ECI[:,1] , r_ECI[:,2]
    
    for n in range(lnJD):
        # solar vector
        rs = (au_to_km)*np.array([rsx[n],rsy[n],rsz[n]])
        # solar vector magnitude
        rsm = np.linalg.norm(rs)
        
        for m in range(lt):
            #--------------------------------------------------------------------
            # obtaining the orientation of the nanosatellite in the ECI reference system
            # nadir orientation matrix of the nanosatellite in the PQW system (rotation around the W axis through true anomaly)
            nadir = np.array([[np.cos(theta[m]) , np.sin(theta[m]) , 0],
                              [-np.sin(theta[m]) , np.cos(theta[m]) , 0],
                              [0 , 0 , 1]])
            # we calculate unit vector of the nanosatellite side N1
            No1 = np.transpose(R@nadir)@no1
            nx1[m] = No1[0,0] # component nx1
            ny1[m] = No1[1,0] # component ny1
            nz1[m] = No1[2,0] # component nz1
            N1 = np.array([nx1[m],ny1[m],nz1[m]]) # unit vector side N1
            N1m = np.linalg.norm(N1) # magnitude of unit vector side N1
            # we calculate unit vector of the nanosatellite side N2
            No2 = np.transpose(R@nadir)@no2
            nx2[m] = No2[0,0] # component nx2
            ny2[m] = No2[1,0] # component ny2
            nz2[m] = No2[2,0] # component nz2
            N2 = np.array([nx2[m],ny2[m],nz2[m]]) # unit vector side N2
            N2m = np.linalg.norm(N2) # magnitude of the unit vector side N2
            # we calculate unit vector of the nanosatellite side N3
            No3 = np.transpose(R@nadir)@no3
            nx3[m] = No3[0,1] # component nx3
            ny3[m] = No3[1,1] # component ny3
            nz3[m] = No3[2,1] # component nz3
            N3 = np.array([nx3[m],ny3[m],nz3[m]]) # unit vector N3
            N3m = np.linalg.norm(N3) # magnitude of the unit vector side N3
            # we calculate unit vector of the nanosatellite side N4
            No4 = np.transpose(R@nadir)@no4
            nx4[m] = No4[0,1] # component nx4
            ny4[m] = No4[1,1] # component ny4
            nz4[m] = No4[2,1] # component nz4
            N4 = np.array([nx4[m],ny4[m],nz4[m]]) # unit vector N4
            N4m = np.linalg.norm(N4) # magnitude of the unit vector side N4
            # we calculate unit vector of the nanosatellite side N5
            No5 = np.transpose(R@nadir)@no5
            nx5[m] = No5[0,2] # component nx5
            ny5[m] = No5[1,2] # component ny5
            nz5[m] = No5[2,2] # component nz5
            N5 = np.array([nx5[m],ny5[m],nz5[m]]) # unit vector N5
            N5m = np.linalg.norm(N5) # magnitud of the unit vector side N5
            # we calculate unit vector of the nanosatellite side N6
            No6 = np.transpose(R@nadir)@no6
            nx6[m] = No6[0,2] # component nx6
            ny6[m] = No6[1,2] # component ny6
            nz6[m] = No6[2,2] # component nz6
            N6 = np.array([nx6[m],ny6[m],nz6[m]]) # unit vector N6
            N6m = np.linalg.norm(N6) # magnitude of the unit vector side N6
            #-------------------------------------------------------------------
            # nanosatellite position vector 
            r = np.array([rx[m],ry[m],rz[m]])
            # magnitude of the nanosatellite position vector
            rm = np.linalg.norm(r)
            #-------------------------------------------------------------------
            # obtaining the earth shadow effect 
            # angle beteween mean radius of the earth and the magnitude of the nanosatellite position vector 
            xc = np.arccos(Re/rm)
            # angle beteween mean radius of the earth and the magnitude of the solar vector
            xs = np.arccos(Re/rsm)
            # angle beteween nanosatellite position vector and the solar vector 
            x =  np.arccos((np.dot(r,rs))/(rm*rsm))
            # angle comparison
            if xc + xs <= x:
                step_f = 0  # the nanosatellite is inside the eclipse zone
            else:
                step_f = 1 # the nanosatellite is outside the eclipse zone
            #------------------------------------------------------------------
            # obtaining the direct solar radiation for each side of the nanosatellite
            # view factor of the nanosatellite side N1
            solar_vf_N1 = np.dot(N1,(rs/rsm))
            if solar_vf_N1 < 0: # the side N1 does not the sun (solar vector)
               solar_vf_N1 = 0
            # direct solar radiation of the nanosatellite side N1
            Q_solar_N1[n,m] = Gs*solar_vf_N1*step_f
            # view factor of the nanosatellite side N2
            solar_vf_N2 = np.dot(N2,(rs/rsm))
            if solar_vf_N2 < 0:# the side N2 does not the sun (solar vector)
                solar_vf_N2 = 0
            # direct solar radiation of the nanosatellite side N2
            Q_solar_N2[n,m] = Gs*solar_vf_N2*step_f
            # view factor of the nanosatellite side N3
            solar_vf_N3 = np.dot(N3,(rs/rsm))
            if solar_vf_N3 < 0:# the side N3 does not the sun (solar vector)
                solar_vf_N3 = 0
            # direct solar radiation of the nanosatellite side N3
            Q_solar_N3[n,m] = Gs*solar_vf_N3*step_f
            # view factor of the nanosatellite side N4
            solar_vf_N4 = np.dot(N4,(rs/rsm))
            if solar_vf_N4 < 0: # the side N4 does not the sun (solar vector)
                solar_vf_N4 = 0
            # direct solar radiation of the nanosatellite side N4
            Q_solar_N4[n,m] = Gs*solar_vf_N4*step_f
            # view factor of the nanosatellite side N5
            solar_vf_N5 = np.dot(N5,(rs/rsm))
            if solar_vf_N5 < 0: # the side N5 does not the sun (solar vector)
               solar_vf_N5 = 0
            # direct solar radiation of the nanosatellite side N5
            Q_solar_N5[n,m] = Gs*solar_vf_N5*step_f
            # view factor of the nanosatellite side N6
            solar_vf_N6 = np.dot(N6,(rs/rsm))
            if solar_vf_N6 < 0: # the side N6 does not the sun (solar vector)
               solar_vf_N6 = 0
            # direct solar radiation of the nanosatellite side N6
            Q_solar_N6[n,m] = Gs*solar_vf_N6*step_f
            #--------------------------------------------------------------
            # Obtaining the albedo and infrared radiation for each side of the nanosatellite
            # condition to active or desactivate albedo radiation
            if x >= 0 and x <= (np.pi)/2:
                albe_f = 0.3*np.cos(x)
            elif x > (np.pi)/2:
                albe_f = 0
            # Constants
            H = rm/Re 
            fi = np.arcsin(1/H) 
            C = np.sqrt((H**2)-1)
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N1
            gam_N1 = np.acos(-((np.dot(N1,r))/(rm*N1m)))
            # view factor of the side N1
            if gam_N1 <= ((np.pi)/2)-fi: # the side N1 sees all the earth
                albe_vf_N1 = (np.cos(gam_N1))/(H**2)
            elif gam_N1 >= ((np.pi)/2)-fi and gam_N1 <= ((np.pi)/2)+fi: # the side N1 sees part of the earth
                w1_N1 = (1/2)*(np.arcsin((C)/(H*np.sin(gam_N1))))
                w2_N1 = (1/(2*(H**2)))*(np.cos(gam_N1)*np.arccos(-C*(1/(np.tan(gam_N1))))-C*np.sqrt(1-((H*np.cos(gam_N1))**2)))
                albe_vf_N1 = (1/2)-(2/(np.pi))*(w1_N1-w2_N1)
            else: # the side N1 does not sees the earth
                albe_vf_N1 = 0
            # albedo radiation of the nanosatellite side N1 
            Q_albe_N1[n,m] = Gs*albe_vf_N1*albe_f
            # infrared radiation of the nanosatellite side N1
            Q_ir_N1[n,m] = Ge*albe_vf_N1
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N2
            gam_N2 = np.ma.arccos(-((np.dot(N2,r))/(rm*N2m)))
            # view factor of the side N2
            if gam_N2 <= ((np.pi)/2)-fi:# the side N2 sees all the earth
                albe_vf_N2 = (np.cos(gam_N2))/(H**2)
            elif gam_N2 >= ((np.pi)/2)-fi and gam_N2 <= ((np.pi)/2)+fi: # the side N2 sees part of the earth
                w1_N2 = (1/2)*(np.ma.arcsin((C)/(H*np.sin(gam_N2))))
                w_N2 = (1/(2*(H**2)))*(np.cos(gam_N2)*np.ma.arccos(-1*C*(1/(np.tan(gam_N2))))-C*np.sqrt(1-(H*np.cos(gam_N2))**2))
                albe_vf_N2 = (1/2)-(2/(np.pi))*(w1_N2-w_N2)
            else: # the side N2 does not sees the earth
                albe_vf_N2 = 0
            # albedo radiation of the nanosatellite side N2
            Q_albe_N2[n,m] = Gs*albe_vf_N2*albe_f
            # infrared radiation of the nanosatellite side N2
            Q_ir_N2[n,m] = Ge*albe_vf_N2
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N3
            gam_N3 = np.ma.arccos(-((np.dot(N3,r))/(rm*N3m)))
            # view factor of the side N3
            if gam_N3 <= ((np.pi)/2)-fi: # the side N3 sees all the earth
                albe_vf_N3 = (np.cos(gam_N3))/(H**2)
            elif gam_N3 >= ((np.pi)/2)-fi and gam_N3 <= ((np.pi)/2)+fi: # the side N3 sees part of the earth
                w1_N3 = (1/2)*(np.ma.arcsin((C)/(H*np.sin(gam_N3))))
                w2_N3 = (1/(2*(H**2)))*(np.cos(gam_N3)*np.ma.arccos(-1*C*(1/(np.tan(gam_N3))))-C*np.sqrt(1-(H*np.cos(gam_N3))**2))
                albe_vf_N3 = (1/2)-(2/(np.pi))*(w1_N3-w2_N3)
            else: # the side N3 does not sees the earth
                albe_vf_N3 = 0
            # albedo radiation of the nanosatellite side N3
            Q_albe_N3[n,m] = Gs*albe_vf_N3*albe_f
            # infrared radiation of the nanosatellite side N3
            Q_ir_N3[n,m] = Ge*albe_vf_N3
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N4
            gam_N4 = np.ma.arccos(-((np.dot(N4,r))/(rm*N4m)))
            # view factor of the side N4
            if gam_N4 <= ((np.pi)/2)-fi: # the side N4 sees all the earth
                albe_vf_N4 = (np.cos(gam_N4))/(H**2)
            elif gam_N4 >= ((np.pi)/2)-fi and gam_N4 <= ((np.pi)/2)+fi: # the side N4 sees part of the earth
                w1_N4 = (1/2)*(np.ma.arcsin((C)/(H*np.sin(gam_N4))))
                w2_N4 = (1/(2*(H**2)))*(np.cos(gam_N4)*np.ma.arccos(-1*C*(1/(np.tan(gam_N4))))-C*np.sqrt(1-(H*np.cos(gam_N4))**2))
                albe_vf_N4 = (1/2)-(2/(np.pi))*(w1_N4-w2_N4)
            else: # the side N4 does not sees the earth
                albe_vf_N4 = 0
            # albedo radiation of the nanosatellite side N4
            Q_albe_N4[n,m] = Gs*albe_vf_N4*albe_f
            # infrared radiation of the nanosatellite side N4
            Q_ir_N4[n,m] = Ge*albe_vf_N4
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N5
            gam_N5 = np.ma.arccos(-((np.dot(N5,r))/(rm*N5m)))
            # view factor of the side N5
            if gam_N5 <= ((np.pi)/2)-fi: # the side N5 sees all the earth
                albe_vf_N5 = (np.cos(gam_N5))/(H**2)
            elif gam_N5 >= ((np.pi)/2)-fi and gam_N5 <= ((np.pi)/2)+fi: # the side N5 sees part of the earth
                w1_N5 = (1/2)*(np.ma.arcsin((C)/(H*np.sin(gam_N5))))
                w2_N5 = (1/(2*(H**2)))*(np.cos(gam_N5)*np.ma.arccos(-1*C*(1/(np.tan(gam_N5))))-C*np.sqrt(1-(H*np.cos(gam_N5))**2))
                albe_vf_N5 = (1/2)-(2/(np.pi))*(w1_N5-w2_N5)
            else: # the side N5 does not sees the earth
                albe_vf_N5 = 0
            # albedo radiation of the nanosatellite side N5
            Q_albe_N5[n,m] = Gs*albe_vf_N5*albe_f
            # infrared radiation of the nanosatellite side N5
            Q_ir_N5[n,m] = Ge*albe_vf_N5
            
            # we calculate the albedo and infrared radiation for the nanosatellite side N6
            gam_N6 = np.ma.arccos(-((np.dot(N6,r))/(rm*N6m)))
            # view factor of the side N6
            if gam_N6 <= ((np.pi)/2)-fi: # the side N6 sees all the earth
                albe_vf_N6 = (np.cos(gam_N6))/(H**2)
            elif gam_N6 >= ((np.pi)/2)-fi and gam_N6 <= ((np.pi)/2)+fi: # the side N6 sees part of the earth
                w1_N6 = (1/2)*(np.ma.arcsin((C)/(H*np.sin(gam_N6))))
                w2_N6 = (1/(2*(H**2)))*(np.cos(gam_N6)*np.ma.arccos(-1*C*(1/(np.tan(gam_N6))))-C*np.sqrt(1-(H*np.cos(gam_N6))**2))
                albe_vf_N6 = (1/2)-(2/(np.pi))*(w1_N6-w2_N6)
            else: # the side N6 does not sees the earth
                albe_vf_N6 = 0
            # albedo radiation of the nanosatellite side N6
            Q_albe_N6[n,m] = Gs*albe_vf_N6*albe_f
            # infrared radiation of the nanosatellite side N6
            Q_ir_N6[n,m] = Ge*albe_vf_N6
            #----------------------------------------------------------------
            # total radiation (solar + albedo + infrared) for each side of the nanosatellite
            Qt_N1[n,m] = Q_solar_N1[n,m]+Q_albe_N1[n,m]+Q_ir_N1[n,m] # total radiation side N1
            Qt_N2[n,m] = Q_solar_N2[n,m]+Q_albe_N2[n,m]+Q_ir_N2[n,m] # total radiation side N2
            Qt_N3[n,m] = Q_solar_N3[n,m]+Q_albe_N3[n,m]+Q_ir_N3[n,m] # total radiation side N3
            Qt_N4[n,m] = Q_solar_N4[n,m]+Q_albe_N4[n,m]+Q_ir_N4[n,m] # total radiation side N4
            Qt_N5[n,m] = Q_solar_N5[n,m]+Q_albe_N5[n,m]+Q_ir_N5[n,m] # total radiation side N5
            Qt_N6[n,m] = Q_solar_N6[n,m]+Q_albe_N6[n,m]+Q_ir_N6[n,m] # total radiation side N6
            
            # temperature for each side of the nanosatellite
            Tt_N1[n,m] = ((Qt_N1[n,m])/k)**(1/4)-to_kelvin # temperatura side N1
            Tt_N2[n,m] = ((Qt_N2[n,m])/k)**(1/4)-to_kelvin # temperatura side N2
            Tt_N3[n,m] = ((Qt_N3[n,m])/k)**(1/4)-to_kelvin # temperatura side N3
            Tt_N4[n,m] = ((Qt_N4[n,m])/k)**(1/4)-to_kelvin # temperatura side N4
            Tt_N6[n,m] = ((Qt_N5[n,m])/k)**(1/4)-to_kelvin # temperatura side N5
            Tt_N5[n,m] = ((Qt_N6[n,m])/k)**(1/4)-to_kelvin # temperatura side N6
        # end for loop
    # end for loop
    # total radiation of the nanosatellite      
    Qt = Qt_N1+Qt_N2+Qt_N3+Qt_N4+Qt_N5+Qt_N6
    # total temperature of the nanosatellite
    Tt = ((Qt)/(k))**(1/4)-to_kelvin
    # return parameters
    return Qt_N1 , Qt_N2 , Qt_N3 , Qt_N4 , Qt_N5 , Qt_N6 , Tt_N1 , Tt_N2 , Tt_N3 , Tt_N4 , Tt_N6 , Tt_N5 , Tt