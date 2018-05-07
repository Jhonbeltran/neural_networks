import numpy as np #Libreria para manejo avanzado de arrays
from colorama import Fore, Back, Style, init
init(autoreset=True)
 
class Perceptron(object):
    """Implements a perceptron network"""
    def __init__(self, input_size, lr=1, epochs=True):
        self.W = np.zeros(input_size+1)
        print(Back.BLUE + "Matriz inicial vacia para los pesos: {} ".format(self.W))
        # add one for bias
        self.epochs = epochs
        self.lr = lr
        
    def activation_fn(self, x):
        return 1 if x >= 0 else 0

    def predict(self, x):
        print(Fore. CYAN + "Datos de entrada {}".format(x))
        x = np.insert(x, 0, 1)
        print("Producto punto entre las columnas de la matriz W y la x")
        #Se realiza producto punto .T es para convertir filas en columnas
        print("Matriz W transformada: {}".format(self.W.T))
        print("Matriz X: {}".format(x))
        z = self.W.T.dot(x)
        print("Resultado del producto punto z = {}".format(z))
        a = self.activation_fn(z)
        print("Valor dado por la función de activación: {}".format(a))
        return a

    def fit(self, X, d):
        learn_loop = 1
        while self.epochs==True:
            verification = []
            print(Back.BLACK + "CICLO DE APRENDIZAJE: {}".format(learn_loop))
            for i in range(d.shape[0]):
                print(Back.MAGENTA + "Generación del resultado número {} esperado".format(i+1))
                print(Fore. YELLOW + "Estado actual de la matriz de pesos {} ".format(self.W))
                y = self.predict(X[i])
                if d[i] == y:
                    print(Fore.GREEN + "El resultado es Correcto")
                    verification.append(y)
                    print(Fore.MAGENTA + "{}".format(verification))
                    if len(verification)==4:
                        self.epochs = False
                else: 
                    verification = []

                print(Fore. CYAN + "Para los dato de entrada {} el resultado obtenido es {} ".format(X[i],y))
                e = d[i] - y
                print("Valor de error actual: {}".format(e))
                self.W = self.W + self.lr * e * np.insert(X[i], 0, 1)
                print(Fore. YELLOW + "Ajuste de pesos {}".format(self.W))

                learn_loop += 1

if __name__ == '__main__':

    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    d = np.array([0, 0, 0, 1])

    perceptron = Perceptron(input_size=2) # Creación del esqueleto del perceptrón
    perceptron.fit(X, d) #Entrenamiento del perceptrón
    print(Back.BLUE + "Los pesos finales son: {}".format(perceptron.W))
    
    prediccion = perceptron.predict(np.array([1,1]))
                                
    print(Fore. CYAN + "El Resultado es: {}".format(prediccion))