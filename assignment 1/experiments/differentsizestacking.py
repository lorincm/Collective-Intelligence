from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, HeadlessSimulation, Simulation, Matrix
from vi.config import Config, dataclass, deserialize
import random
import math
from multiprocessing import Pool
import polars as pl

A=1.70188
B=3.58785

class State(Enum):
    JOINING = auto()
    LEAVING = auto()
    STILL = auto()
    WANDERING = auto()

# class CockroachConfig(Config):
#     tjoin: int = 10
#     tleave: int = 30

class Cockroach(Agent):
    config: Config

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.PJOIN = 0.8
        # self.PLEAVE = 0.0001
        self.TJOIN = 13
        self.TLEAVE = 34
        self.ticker = 0
        self.state = State.WANDERING
        self.move_save=[]

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def change_position(self):
        neighbours=list(self.in_proximity_accuracy())
        n = len(neighbours)
        self.PJOIN = 0.03 + 0.48 * (1-(math.e**(-A*n)))

        # if n<=3:
        #     self.PLEAVE=(math.e)**(-B*n)
        # else:
        #     self.PLEAVE=(math.e)**(-B*3)
        self.PLEAVE=(math.e)**(-B*n)


        # print('PJOIN: {}, PLEAVE: {}'.format(self.PJOIN, self.PLEAVE))

        changed = self.there_is_no_escape()

        prng = self.shared.prng_move

        # Always calculate the random angle so a seed could be used.
        deg = prng.uniform(-30, 30)

        # Only update angle if the agent was teleported to a different area of the simulation.
        if changed:
            self.move.rotate_ip(deg)

        if self.state == State.WANDERING:
            #small rectangle and big rectangle bound check
            if ((185 < self.pos[0] < 265) and (335 < self.pos[1] < 415)) or ((440 < self.pos[0] < 610) and (290 < self.pos[1] < 460)):     
                if random.uniform(0, 1) <= self.PJOIN:
                    self.state = State.JOINING
                else:
                    self.pos+=self.move
            else:
                self.pos += self.move
        
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



