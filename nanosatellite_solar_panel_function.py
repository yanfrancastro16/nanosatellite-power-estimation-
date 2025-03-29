import numpy as np
#--------------------------------------------------------
"""
input parameters of the function nanosatellite_solar_panel_function:
Q (radiation for one side (face) of the nanosatellite) [W/m^2]
Tt (total temperature of the nanosatellite) [ºC]
Vocn (nominal open circuit voltage or standard conditions) [V]
Iscn (nominal short ciruit current or standard conditions) [A]
Kv (voltage temperature coefficient) [V/ºC]
Ki (current temperature coefficient) [I/ºC)
losses (internal solar panel losses of the nanosatellite) [%]
""" 
#--------------------------------------------------------
def nanosatellite_solar_panel_function(Q,Tt,Vocn,Iscn,Kv,Ki,losses):
    # solar flux
    Qs = 1367
    # Boltzmann constant 
    k = 1.38E-23
    # electron charge
    q = 1.6E-19
    # conversion to kelvin
    to_kelvin = 273.15
    # room temperature
    To = 28+to_kelvin
    #-----------------------------------------------------
    # shape of the radiation matrix for one side of te nanoastellite
    rows_Q , columns_Q = np.shape(Q)
    #-----------------------------------------------------
    # initial maximum power , voltage, current arrays of solar panel of the nanosatellite
    Pm = np.zeros((rows_Q,columns_Q)) # maximum power array
    Vm = np.zeros((rows_Q,columns_Q)) # maximum voltage array
    Im = np.zeros((rows_Q,columns_Q)) # maximum current array
    #-----------------------------------------------------
    # for each radiation and temperature values, we calculate the power, voltage and current of the solar panel
    for i in range(0,rows_Q):
          for j in range(0,columns_Q):
              # incident radiation of solar panel (side of the nanosatellite)
              Qn = Q[i,j] 
              # temperature of solar panel (side of the nanosatellite)
              Tn = Tt[i,j]+to_kelvin
              # delta temperature
              DT = Tn-To 
              # reference short circuit current
              Isc = (Qn/Qs)*Iscn
              # photo current
              Iph = Isc*(1+Ki*DT)
              # thermal voltage
              Vt = (k*Tn)/q
              # reverse saturation current of the diode
              Io = (Iscn+Ki*DT)/(np.ma.exp((Vocn+Kv*DT)/Vt)-1)
              # open circuit voltage
              Voc = Vt*np.ma.log((Isc/Io)+1)
              # shunt resistence
              Rp = 100*(1/(losses/100))*(Vocn/Iscn)
              # solar panel voltage array
              V = np.linspace(0,Voc,51)
              # total current of the solar panel
              I = Iph-Io*(np.ma.exp((V/Vt))-1)-(V/Rp)
              # total power of the solar panel 
              P = V*I
              # taking the maximum power values generated for solar panel (maximum power points)
              # maximum power of the solar panel
              Pm[i,j],n = P.max(0), P.argmax(0)
              # maximum voltage of the solar panel
              Vm[i,j] = V[n]
              # maximum current of the solar panel
              Im[i,j] = I[n]
            # end for loop  
    # end for loop 
    # return parameters 
    return Pm,Vm,Im
