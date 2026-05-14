import numpy as np
import matplotlib.pyplot as plt

Energies = []
nats = 6
x = np.arange(0,406.5,0.25)
with open('etot.dat', 'r') as f:
	for lines in f:
		lines = lines.strip()
		energies = float(lines)
		Energies.append(energies)


#print(Energies)

Energies = np.array(Energies)

deltaE = (Energies - Energies[0]) / nats
#print(deltaE.shape)
deltaE = deltaE * 1000

plt.plot(x,deltaE, label='dt=0.25')

Energies = []
nats = 2
x = np.arange(0,400.0,0.5)
with open('./0.5/etot.dat', 'r') as f:
	for lines in f:
		lines = lines.strip()
		energies = float(lines)
		Energies.append(energies)


#print(Energies)

Energies = np.array(Energies)

deltaE = (Energies - Energies[0]) / nats
deltaE = deltaE * 1000
#print(deltaE)

plt.plot(x,deltaE, label='dt=0.5')
plt.title('2 Water System; 2 parts graph partition')
plt.xlabel('Time (fs)')
plt.ylabel('Fluctuations (meV/atom)')
plt.legend()
plt.show()
