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
iterations=10000

class State(Enum):
    JOINING = auto()
    LEAVING = auto()
    STILL = auto()
    WANDERING = auto()

# class CockroachConfig(Config):
#     tjoin: int = 10
#     tleave: int = 30

class CockroachConfig(Config):
    tjoin: int = 7
    tleave: int = 30
    small_radius: int = 20
    radius: int = 35


class Cockroach(Agent):
    
    config: CockroachConfig

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.PJOIN = 0.8
        # self.PLEAVE = 0.0001
        self.TJOIN = self.config.tjoin
        self.TLEAVE = self.config.tleave
        self.ticker = 0
        self.state = State.WANDERING
        self.move_save=[]


    def change_position(self):
        global iterations
        iterations=iterations+1
        wide_neighbours=list(self.in_proximity_accuracy())
        wide_n=len(wide_neighbours)
        wide_still=[i for i in wide_neighbours if i[0].state == State.STILL]
        local_neighbours=[i for i in wide_neighbours if i[1] < self.config.small_radius]
        local_still=[i for i in wide_neighbours if i[0].state == State.STILL]
        wide_n_still=len(wide_still)
        local_n_still=len(local_still)
        n = len(local_neighbours)
        dispersing_time = (iterations%200000)
        

        if self.state == State.STILL and wide_n_still == local_n_still\
              and  (-1<dispersing_time<10001):
            print('KILL')
            self.state=State.LEAVING
            self.move=self.move_save[0]
            # print(self.move)
            self.ticker=0
            self.change_image(0)

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
            for neighbour in local_neighbours:
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



sim=Simulation(CockroachConfig(movement_speed=5, radius=35))
sim.batch_spawn_agents(100, Cockroach, ["images/green.png", 'images/red.png'])
sim.spawn_site('images/bubble-full.png', 525, 375)
sim.spawn_site('images/bubble-small.png', 225, 375)

sim.run()
