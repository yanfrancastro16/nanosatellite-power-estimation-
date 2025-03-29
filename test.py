import orbit_function as o
import solar_vector_fucntion as sv
import radiation_and_temperature_function as rt
import nanosatellite_solar_panel_function as sp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# clasical orbital elements (keplerian elements) inputs values (for more information consult orbit_function.py) 
A , e , I , O , W , Theta = 719.87 , 0.0102 , 98.08 , 184.09 , 207.97 , 154.07

# date input values (for more information consult solar_vector_function.py)
yo , mth , d , h , m , s = 2024 , 11 , 9 , 57 , 5 , 0

# solar panel input parameters (for more information consult nanosatellite_solar_panel.py)
Ns , Np = 2 , 1 # number of panel in series Ns and parallel Np  
Vocn , Iscn , kv , ki , losses = 2.665*Ns , 0.464*Np , -0.0059 , 0.00014 , 5

# calling the orbit_function (for more information consult orbit_function.py)
r_ECI , v_ECI , dimen , R , theta , t = o.orbit_function(A,e,I,O,W,Theta)

# calling the solar_vector_function (for more information consult nanosatellite_solar_panel.py)
rs_ECI , solar_parameters , nJD = sv.solar_vector_function(yo,mth,d,h,m,s,I,O,A)

# calling the radiation_and_temperature_function (for more information consult radiation_and_Temperature_function.py)
Qt_N1 , Qt_N2 , Qt_N3 , Qt_N4 , Qt_N5 , Qt_N6 , Tt_N1 , Tt_N2 , Tt_N3 , Tt_N4 , Tt_N6 , Tt_N5 , Tt = rt.radiation_and_temperature_function(r_ECI,rs_ECI,theta,R,nJD,t)

# calling the nanosatellite_solar_panel_function for each solar panel in the six sides of the nanosatellite (for more information consult nanosatellite_solar_panel_function.py)
# nanosatellite solar panel side N1
nanosatellite_solar_panel_N1 = sp.nanosatellite_solar_panel_function(Qt_N1,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage, and current solar panel side N1
power_N1 , voltage_N1 , current_N1  = nanosatellite_solar_panel_N1
mean_power_N1 = np.mean(power_N1) # mean power generated of solar panel side N1
max_power_N1 = np.max(power_N1) # max power generated of solar panel side N1
min_power_N1 = np.min(power_N1) # min power generated of solar panel side N1

# nanosatellite solar panel side N2
nanosatellite_solar_panel_N2 = sp.nanosatellite_solar_panel_function(Qt_N2,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage , current solar panel N2
power_N2 , voltage_N2 , current_N2  = nanosatellite_solar_panel_N2
mean_power_N2 = np.mean(power_N2) # mean power generated of solar panel side N2
max_power_N2 = np.max(power_N2) # max power generated of solar panel side N2
min_power_N2 = np.min(power_N2) # min power generated of solar panel side N2

# nanosatellite solar panel side N3
nanosatellite_solar_panel_N3 = sp.nanosatellite_solar_panel_function(Qt_N3,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage , current solar panel N3
power_N3 , voltage_N3 , current_N3  = nanosatellite_solar_panel_N3
mean_power_N3 = np.mean(power_N3) # mean power generated of solar panel side N3
max_power_N3 = np.max(power_N3) # max power generated of solar panel side N3
min_power_N3 = np.min(power_N3) # min power generated of solar panel side N3

# nanosatellite solar panel side N4
nanosatellite_solar_panel_N4 = sp.nanosatellite_solar_panel_function(Qt_N4,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage , current solar panel N4
power_N4 , voltage_N4 , current_N4  = nanosatellite_solar_panel_N4
mean_power_N4 = np.mean(power_N4) # mean power generated of solar panel side N4
max_power_N4 = np.max(power_N4) # max power generated of solar panel side N4
min_power_N4 = np.min(power_N4) # min power generated of solar panel side N4

# nanosatellite solar panel side N5
nanosatellite_solar_panel_N5 = sp.nanosatellite_solar_panel_function(Qt_N5,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage , current solar panel N5
power_N5 , voltage_N5 , current_N5  = nanosatellite_solar_panel_N5
mean_power_N5 = np.mean(power_N5) # mean power generated of solar panel side N5
max_power_N5 = np.max(power_N5) # max power generated of solar panel side N5
min_power_N5 = np.min(power_N5) # min power generated of solar panel side N5

# nanosatellite solar panel side N6
nanosatellite_solar_panel_N6 = sp.nanosatellite_solar_panel_function(Qt_N6,Tt,Vocn,Iscn,kv,ki,losses)
# power , voltage , current solar panel N6
power_N6 , voltage_N6 , current_N6  = nanosatellite_solar_panel_N6
mean_power_N6 = np.mean(power_N6) # mean power generated of solar panel side N6
max_power_N6 = np.max(power_N6) # max power generated of solar panel side N6
min_power_N6 = np.max(power_N6) #  min power generated of solar panel side N6

# total mean power generated for all solar panels of the nanosatellite
total_mean_power = mean_power_N1+mean_power_N2+mean_power_N3+mean_power_N4+mean_power_N5+mean_power_N6

# total power of the nanosatellite
total_power = power_N1+power_N2+power_N3+power_N4+power_N5+power_N6
total_max_power = np.max(total_power) # max power generated for solar panels of the nanosatellite
total_min_power = np.min(total_power) # min power generated for solar panels of the nanosatellite
#-------------------------------------------------------------------------------------------------
# power data
# dictionary for clustering the total power generated data
power_data = {
    "Total Power Solar Panels": ["Average [W]" , "Maximum [W]" , "Minimum [W]"],
    "": [total_mean_power , total_max_power , total_min_power]
}
# dictionary for clustering the power generated for each solar panel data
power_data_solar_panels = {
    "Solar Panel": ["Side N1" , "Side N2" , "Side N3" , "Side N4" , "Side N5" , "Side N6"],
    "Average Power [W]": [mean_power_N1 , mean_power_N2 , mean_power_N3 , mean_power_N4 , mean_power_N5 , mean_power_N6],
    "Maximum Power [W]": [max_power_N1 , max_power_N2 , max_power_N3 , max_power_N4 , max_power_N5 , max_power_N6],
    "Minimum Power [W]": [min_power_N1 , min_power_N2 , min_power_N3 , min_power_N4 , min_power_N5 , min_power_N6],
}
# data frame to display the total power generated
df_power_data = pd.DataFrame(power_data)
# data frame to display the power generated for each solar panel
df_power_data_solar_panels = pd.DataFrame(power_data_solar_panels)
# displaying power data
print(df_power_data)
print('\n')
print(df_power_data_solar_panels)
#----------------------------------------------------------------------------------------------------
# sunlight and eclipse data
# taking the sunlight and eclipse time array from solar parameters
t_sunlight = solar_parameters[:,0]
t_eclipse = solar_parameters[:,1]
# mean , max , min sunlight time
mean_t_sunlight = np.mean(t_sunlight)
max_t_sunlight = np.max(t_sunlight)
min_t_sunlight = np.min(t_sunlight)
# mean , max , min eclipse time
mean_t_eclipse = np.mean(t_eclipse)
max_t_eclipse = np.max(t_eclipse)
min_t_eclipse = np.min(t_eclipse)
# dictionary for clustering the sunligth time data
data_sunlight_time = {
    "Sunlight Time [minutes]": ["Mean" , "Maximum" , "Minimum"],
    "": [mean_t_sunlight , max_t_sunlight , min_t_sunlight ]
}
# dictionary for clustering the eclipse time data
data_eclipse_time = {
    "Eclipse Time [minutes]": ["Mean" , "Maximum" , "Minimum"],
    "": [mean_t_eclipse , max_t_eclipse , min_t_eclipse]
}
# data frame to display the sunlight time
df_data_sunlight_time = pd.DataFrame(data_sunlight_time)
# data frame to display the eclipse time
df_data_eclipse_time = pd.DataFrame(data_eclipse_time)
# displaying light and eclipse times data
print('\n')
print(df_data_sunlight_time)
print('\n')
print(df_data_eclipse_time)
#----------------------------------------------------------------------------------------------------------
# plotting power data
# month in orbit
month = 1
# conversion to minutes
to_minutes = 60
# orbital period 
T = dimen[6]
# plotting the power generated for each sola panel of the nanosatellite
plt.subplot(3,1,1)
plt.title('Power, Voltage, and Current Generated by the Solar Panels of the Nanosatellite Month ' + str(month) + ' in Orbit')
plt.ylabel('Power (W)')
plt.plot(t/to_minutes,power_N1[month-1,:], linewidth = 2 , label = 'Solar Panel side N1' , color = 'blue')
plt.plot(t/to_minutes,power_N2[month-1,:], linewidth = 2 , label = 'Solar Panel side N2' , color = 'red')
plt.plot(t/to_minutes,power_N3[month-1,:], linewidth = 2 , label = 'Solar Panel side N3' , color = 'green')
plt.plot(t/to_minutes,power_N4[month-1,:], linewidth = 2 , label = 'Solar Panel side N4' , color = 'yellow')
plt.plot(t/to_minutes,power_N5[month-1,:], linewidth = 2 , label = 'Solar Panel side N5' , color = 'orange')
plt.plot(t/to_minutes,power_N6[month-1,:], linewidth = 2 , label = 'Solar Panel side N6' , color = 'purple')
plt.grid()
plt.legend(loc = 'upper right')

plt.subplot(3,1,2)
plt.ylabel('Voltage (V)')
plt.plot(t/to_minutes,voltage_N1[month-1,:], linewidth = 2 , label = 'Solar Panel side N1' , color = 'blue')
plt.plot(t/to_minutes,voltage_N2[month-1,:], linewidth = 2 , label = 'Solar Panel side N2' , color = 'red')
plt.plot(t/to_minutes,voltage_N3[month-1,:], linewidth = 2 , label = 'Solar Panel side N3' , color = 'green')
plt.plot(t/to_minutes,voltage_N4[month-1,:], linewidth = 2 , label = 'Solar Panel side N4' , color = 'yellow')
plt.plot(t/to_minutes,voltage_N5[month-1,:], linewidth = 2 , label = 'Solar Panel side N5' , color = 'orange')
plt.plot(t/to_minutes,voltage_N6[month-1,:], linewidth = 2 , label = 'Solar Panel side N6' , color = 'purple')
plt.grid()
plt.legend(loc = 'upper right')

plt.subplot(3,1,3)
plt.ylabel('Current (A)')
plt.xlabel('Orbital period [minutes]')
plt.plot(t/to_minutes,current_N1[month-1,:], linewidth = 2 , label = 'Solar Panel side N1' , color = 'blue')
plt.plot(t/to_minutes,current_N2[month-1,:], linewidth = 2 , label = 'Solar Panel side N2' , color = 'red')
plt.plot(t/to_minutes,current_N3[month-1,:], linewidth = 2 , label = 'Solar Panel side N3' , color = 'green')
plt.plot(t/to_minutes,current_N4[month-1,:], linewidth = 2 , label = 'Solar Panel side N4' , color = 'yellow')
plt.plot(t/to_minutes,current_N5[month-1,:], linewidth = 2 , label = 'Solar Panel side N5' , color = 'orange')
plt.plot(t/to_minutes,current_N6[month-1,:], linewidth = 2 , label = 'Solar Panel side N6' , color = 'purple')
plt.grid()
plt.legend(loc = 'upper right')
plt.show()

# plotting the total power generated for solar panels of the nanosatellite
plt.title('Total Power Generated by the Solar Panels of the Nanosatellite Month ' + str(month) + ' in Orbit')
plt.ylabel('Power (W)')
plt.xlabel('Orbital Period [minutes]')
plt.plot(t/to_minutes,total_power[month-1,:], linewidth = 2 , label = 'Total Power' , color = 'blue')
plt.grid()
plt.legend(loc = 'upper right')
plt.show()