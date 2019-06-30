import csv
from leather import Chart
from datetime import date
from numpy.fft import fft, fftfreq
import numpy as np
import matplotlib.pyplot as plt

data_x_pole = []
data_y_pole = []
data_x_y_pole = []

x_poles=[]
Y_poles=[]

for y in range(1998,2018):
	with open('data/'+str(y)+'.csv', 'r') as f:
		reader = csv.reader(f)
		your_list = list(reader)
	for x in range(1,len(your_list)):
		line=your_list[x][0].split(';')
		x_poles.append(float(line[5]))
		Y_poles.append(float(line[7]))
		data_x_pole.append((date(int(line[1]), int(line[2]), int(line[3])), float(line[5])))
		data_y_pole.append((date(int(line[1]), int(line[2]), int(line[3])), float(line[7])))
		data_x_y_pole.append((float(line[5]), float(line[7])))
	f.close()



cart_x_pole = Chart('x, time')
cart_x_pole.add_x_scale(date(1998, 1, 1), date(2018, 12, 31))
cart_x_pole.add_line(data_x_pole)
cart_x_pole.to_svg('response/x_pole.svg')


cart_y_pole = Chart('y, time')
cart_y_pole.add_x_scale(date(1998, 1, 1), date(2018, 12, 31))
cart_y_pole.add_line(data_y_pole)
cart_y_pole.to_svg('response/y_pole.svg')


chart_x_y_pole = Chart('x, y')
chart_x_y_pole.add_dots(data_x_y_pole)
chart_x_y_pole.to_svg('response/x_y_pole.svg')

x_freq=fftfreq(len(x_poles))
x_fft_values=fft(x_poles)
x_mask=x_freq>0
x_fft_theo=2.0*np.abs(x_fft_values/len(x_fft_values))

plt.figure(1)
plt.plot(x_freq, x_fft_values, label="Raw FFT Values")
plt.show()

plt.figure(2)
plt.plot(x_freq[x_mask], x_fft_theo[x_mask], label="True FFT Values")
plt.show()

y_freq=fftfreq(len(Y_poles))
y_fft_values=fft(Y_poles)
y_mask=y_freq>0
x_fft_theo=2.0*np.abs(y_fft_values/len(y_fft_values))

plt.figure(3)
plt.plot(y_freq, y_fft_values, label="Raw FFT Values")
plt.show()

plt.figure(4)
plt.plot(y_freq[y_mask], y_fft_values[y_mask], label="True FFT Values")
plt.show()