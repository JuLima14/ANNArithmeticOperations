import numpy as np
import Substraction
# import mvnc.mvncapi as mvnc

# Look for enumerated Intel Movidius NCS device(s); quit program if none found.
# devices = mvnc.EnumerateDevices()
# if len(devices) == 0:
#     print('No devices found')
#     quit()
# # Get a handle to the first enumerated device and open it
# device = mvnc.Device(devices[0])
# device.OpenDevice()
# Read the graph file into a buffer
# with open( GRAPH_PATH, mode='rb' ) as f:
#     blob = f.read()
 
# # Load the graph buffer into the NCS
# graph = device.AllocateGraph( blob )


test_data = np.random.rand(5,2)

for i in range(5):
    for j in range(2):
        test_data[i][j] = int(test_data[i][j]*10)

print("test_data")
print("  x - y")
print(test_data)

substraction = Substraction.Substraction()
# descomentar solo si se necesita mejorar el modelo o crearlo nuevamente
substraction.train_substraction()

print("Results")
results = substraction.predict(test_data)
for i in range(len(test_data)):
    print("{} - {} = {}".format(test_data[i][0],test_data[i][1],results[i]))