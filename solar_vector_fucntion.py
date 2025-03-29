import numpy as np
#-------------------------------------------------------------------------------------
"""
input parameters of the solar_vector_function:
yo (year)
mth (month)
d (day)
h (hours)
m (minutes)
s (seconds)
I (orbit inclination) [degrees]
O (right ascension) [degrees]
A (semi-major axis, from the surface of the earth) [km]
"""
# ------------------------------------------------------------------------------------
def solar_vector_function(yo,mth,d,h,m,s,I,O,A):
    # mean radius of the earth
    Re = 6378 
    # gravitational constant
    u = 3.986E5
    # factores de conversion de grados y radianes
    aradianes = np.pi/180
    agrados = 180/np.pi
    # conversion to minutes
    to_minutes = 60
    #---------------------------------------------------------------------------------
    # conversion from radians to degrees of the orbit inclination and right ascencion
    i , o = np.radians(I), np.radians(O)
    # semi-major axis + mean radius of the earth
    a = A+Re
    #---------------------------------------------------------------------------------
    # initial Julian date 
    JDi = 367*yo-(7*(yo+(mth+9)/12))/4+(275*mth)/9+d+1721013.5+(h+(m+(s)/60)/24)/24
    # number of Julian date 
    nJDi = JDi-2451545
    # final yaer of the analysis 
    yf = yo+1
    # final Julian date 
    JDf = 367*yf-(7*(yf+(mth+9)/12))/4+(275*mth)/9+d+1721013.5+(h+(m+(s)/60)/24)/24
    # number of Julian date
    nJDf = JDf-2451545
    #---------------------------------------------------------------------------------
    # number of Julian date array
    nJD = np.linspace(nJDi,nJDf,12)
    # size of number of Julian date array
    lnJD = np.size(nJD)
    # initial arrays of the components and magnitud of the solar vector in the ECI reference system 
    rsx , rsy , rsz , rs = np.zeros(lnJD) , np.zeros(lnJD) , np.zeros(lnJD) , np.zeros(lnJD) 
    # initial beta angle array 
    Ba = np.zeros(lnJD)
    # initial light time and eclipse time arrays
    tlight , teclipse = np.zeros(lnJD) , np.zeros(lnJD)  
    #---------------------------------------------------------------------------------
    # for each value of Julian date, we calculate the solar vector and the light an eclipse times 
    for n in range(lnJD):
        # estimating the value of the mean longitude of the sun
        los = 280.459+0.98564736*nJD[n]
        c1 = 0
        while los > 360: # the value of mean longitude of the sun must be <= 360 degrees
            c1 = c1+1
            los = 280.459+0.98564736*nJD[n]-360*c1
        ls = los # value of the mean longitude of the sun
        # estimating the value of the mean anomaly of the sun
        Mos = 357.529+0.98560023*nJD[n]
        c1 = 0
        while Mos > 360: # the value of mean anomaly of the sun must be <= 360 degrees
            c1 = c1+1
            Mos = 357.529+0.98560023*nJD[n]-360*c1 
        Ms = Mos # value of the mean anomaly of the sun
        # ecliptic longitude of the sun
        le = ls+1.915*np.sin(np.radians(Ms))+0.0200*np.sin(2*np.radians(Ms))
        # obliquity of the ecliptic
        oblie =23.439-3.56E-7*nJD[n]
        # magnitude of solar vector in the ECI reference system
        rs[n] = 1.00014-0.01671*np.cos(np.radians(Ms))-0.00014*np.cos(2*np.radians(Ms))
        # components of the solar vector in the ECI reference system
        rsx[n] = rs[n]*np.cos(np.radians(le)) # component x
        rsy[n] = rs[n]*np.cos(np.radians(oblie))*np.sin(np.radians(le)) # component y
        rsz[n] = rs[n]*np.sin(np.radians(oblie))*np.sin(np.radians(le)) # component z
        # solar vector matrix in the ECI reference system
        rs_ECI = np.transpose(np.array([rsx,rsy,rsz,rs]))
        # angle beta
        Beta = np.sin(np.radians(oblie))*np.cos(i)*np.sin(np.radians(le))-np.cos(np.radians(oblie))*np.sin(i)*np.cos(o)*np.sin(np.radians(le))+np.sin(i)*np.sin(o)*np.cos(np.radians(le))
        Bs =  np.arcsin(Beta) # angle beta in radians
        Ba[n] = np.degrees(np.arcsin(Beta)) # angle beta in degrees
        # orbital period
        T = (((2*np.pi)/(np.sqrt(u)))*(a)**(3/2))
        # eclipse time
        teclipse[n] = np.arccos((((1-Re/a)**2)**(1/2))/(np.cos(Bs)))*(T/np.pi)
        # light time
        tlight[n] = T-teclipse[n]
        # solar parameters array
        solar_parameters = np.transpose(np.array([tlight/to_minutes,teclipse/to_minutes,Ba]))
    # end of for loop
    # return parameters
    return rs_ECI , solar_parameters , nJD