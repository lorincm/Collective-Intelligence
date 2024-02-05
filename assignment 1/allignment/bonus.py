from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize
import random
import math
import cv2


A=1.70188
B=3.58785



class State(Enum):
    JOINING = auto()
    LEAVING = auto()
    STILL = auto()
    WANDERING = auto()

class Cockroach(Agent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.PJOIN = 0.8
        # self.PLEAVE = 0.0001
        self.TJOIN = 13
        self.TLEAVE = 34
        self.ticker = 0
        self.state = State.WANDERING
        self.move_save=[]

    def change_position(self):

        neighbours=list(self.in_proximity_accuracy())
        n = len(neighbours)
        self.PJOIN = 0.03 + 0.48 * (1-(math.e**(-A*n)))
        self.PLEAVE=(math.e)**(-B*n)
        # print('PJOIN: {}, PLEAVE: {}'.format(self.PJOIN, self.PLEAVE))

        changed = self.there_is_no_escape()

        prng = self.shared.prng_move

        # Always calculate the random angle so a seed could be used.
        deg = prng.uniform(-30, 30)

        # Only update angle if the agent was teleported to a different area of the simulation.
        if changed:
            self.move.rotate_ip(deg)
        
        if self.state == State.WANDERING and self.in_proximity_accuracy().count() >= 1:
            neighbours=list(self.in_proximity_accuracy())
            #allignment
            sum_v=Vector2()
            sum_x=Vector2()
            sum_difference=Vector2()
            alpha: float = 0.4
            beta: float = 0.15
            gamma: float = 0.1
            delta_time: float = 3
            mass: int = 5
            MAX_VELOCITY=float(3)

            for i in neighbours:

                object=i[0]

                v=object.move
                sum_v+=v

                x=object.pos
                sum_x+=x

                sum_difference+=(self.pos-x)
            

            n=len(neighbours)
            avg_v=sum_v/n
            alignment_force=avg_v-self.move

            avg_x=sum_x/n
            cohesion_force=avg_x-self.pos
            # c=cohesion_force-self.move

            separation=sum_difference/n

            total=(alpha*alignment_force)+(beta*separation)+(gamma*cohesion_force)
            ftotal=total/mass

            self.move += ftotal

            if Vector2.length(self.move) > MAX_VELOCITY:
                self.move=Vector2.normalize(self.move) * MAX_VELOCITY

            self.pos=self.pos+(self.move*delta_time)

        if self.state == State.WANDERING:
            if random.uniform(0, 1) <= self.PJOIN:
                self.state = State.JOINING
            else:
                self.pos+=self.move

        
        if self.state == State.JOINING:
            if self.on_site() or (self.ticker == self.TJOIN):
            # if self.ticker == self.TJOIN:
                self.state=State.STILL
                self.ticker=0
            else:
                self.ticker +=1
                self.pos+=self.move

        if self.state==State.STILL:
            if len(self.move_save) == 0:
                self.move_save.append(self.move)
            self.move=Vector2()
            self.change_image(1)
            # self.PLEAVE=math.e**(-B*n)
            # print(self.PLEAVE)
            if random.uniform(0, 1) <= self.PLEAVE:
                self.move=self.move_save[0]
                # print(self.move)
                self.ticker=0
                self.state = State.LEAVING
                self.change_image(0)
        
        if self.state==State.STILL:
            for neighbour in neighbours:
                if neighbour[1] <= 8 and neighbour[0].state == State.STILL:
                    self.move=self.move_save[0]
                    self.ticker=0
                    self.state = State.LEAVING
                    self.change_image(0)
        
        if self.state==State.LEAVING:
            if self.ticker == self.TLEAVE:
                self.state=State.WANDERING
                self.ticker=0
            else:
                self.ticker+=1
                self.pos+=self.move
                

sim=Simulation(Config(movement_speed=5, radius=36))
sim.batch_spawn_agents(200, Cockroach, ["images/green.png", 'images/red.png'])
sim.run()

