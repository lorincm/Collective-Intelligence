from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt
from enum import Enum, auto
from multiprocessing import Pool

class Lotka_Volterra(object):

    def __init__(self, y_0=[100, 10], parameters=[1.1, 0.4, 0.1, 0.4]):
        self.y_0 = y_0
        self.parameters = parameters
        self.t = np.linspace(0, 60000, num=1000)
        self.solution = None

    def get_differential(self, variables, t):
        x, y = variables
        alpha, beta, delta, gamma = self.parameters

        dx_dt = alpha*x - beta*x*y
        dy_dt = delta*x*y - gamma*y

        return [dx_dt, dy_dt]

    def solve_differential(self):
        self.solution = odeint(self.get_differential, self.y_0, self.t)
        return self.solution

    def plot_graph(self):
        if self.solution is None:
            print("Please solve the differential equations first.")
            return

        plt.figure(figsize=(8, 6))
        
        
        plt.plot(self.t, self.solution[:,0], color='b', label='Prey')
        plt.plot(self.t, self.solution[:,1], color='r', label='Predators')
        plt.title('Model orey vs Predator population over time')

        plt.ylabel('Population')
        plt.xlabel('Time')
        plt.legend(loc='upper right')
        
        plt.savefig('Assignment2/Rabbit-Fox-model.png')

        plt.figure(figsize=(8, 6))

        plt.plot(self.solution[:,0], self.solution[:,1], color='purple', label='Prey vs Predator')
        plt.title('Phase-plane Plot: Rabbits vs Foxes')

        plt.xlabel('Prey Population')
        plt.ylabel('Predator Population')
        plt.legend(loc='upper right')

        plt.savefig('Assignment2/Rabbit-Fox-phase')


lotka=Lotka_Volterra()
lotka.solve_differential() 
lotka.plot_graph()