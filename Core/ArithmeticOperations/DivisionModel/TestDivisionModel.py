import numpy as np
import Division
#hay que cargarle las sumas
test_data = np.random.rand(5,2)

for i in range(5):
    for j in range(2):
        test_data[i][j] = int(test_data[i][j]*10)

print("test_data")
print("  x / y")
print(test_data)

division = Division.Division()
# descomentar solo si se necesita mejorar el modelo o crearlo nuevamente
# division.train_division()

print("Resultados")
results = division.predict(test_data)
for i in range(len(test_data)):
    print("{} / {} = {}".format(test_data[i][0],test_data[i][1],results[i]))