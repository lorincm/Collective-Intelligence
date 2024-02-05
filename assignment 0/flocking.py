from enum import Enum, auto
import pygame as pg
from pygame.math import Vector2
from vi import Agent, Simulation
from vi.config import Config, dataclass, deserialize

R=30
MAX_VELOCITY=float(0.9)

@deserialize
@dataclass
class FlockingConfig(Config):
    alignment_weight: float = 1.0
    cohesion_weight: float = 0.5
    separation_weight: float = 0.5

    delta_time: float = 3

    mass: int = 20

    def weights(self) -> tuple[float, float, float]:
        return (self.alignment_weight, self.cohesion_weight, self.separation_weight)


class Bird(Agent):
    config: FlockingConfig
    """Change the position of the agent.

    The agent's new position is calculated as follows:
    1. The agent checks whether it's outside of the visible screen area.
    If this is the case, then the agent will be teleported to the other edge of the screen.
    2. If the agent collides with any obstacles, then the agent will turn around 180 degrees.
    3. If the agent has not collided with any obstacles, it will have the opportunity to slightly change its angle.
    """
    def change_position(self):

        #from source code

        if not self._moving:
            return

        changed = self.there_is_no_escape()

        prng = self.shared.prng_move

        # Always calculate the random angle so a seed could be used.
        deg = prng.uniform(-30, 30)

        # Only update angle if the agent was teleported to a different area of the simulation.
        if changed:
            self.move.rotate_ip(deg)

        if self.in_proximity_accuracy().count() >= 1:
            neighbours=list(self.in_proximity_accuracy())
            #allignment
            sum_v=Vector2()
            sum_x=Vector2()
            sum_difference=Vector2()
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

            alpha, beta, gamma = self.config.weights()

            total=(alpha*alignment_force)+(beta*separation)+(gamma*cohesion_force)
            ftotal=total/self.config.mass

            self.move += ftotal

            if Vector2.length(self.move) > MAX_VELOCITY:
                self.move=Vector2.normalize(self.move) * MAX_VELOCITY

            self.pos=self.pos+(self.move*self.config.delta_time)

        else:
            self.pos+=self.move
        
        # for obstacle in self.obstacle_intersections:
            

        if (list(self.obstacle_intersections())): #if not empty
                self.move += (self.pos - Vector2(400,300)).normalize() 


class Selection(Enum):
    ALIGNMENT = auto()
    COHESION = auto()
    SEPARATION = auto()


class FlockingLive(Simulation):
    selection: Selection = Selection.ALIGNMENT
    config: FlockingConfig

    def handle_event(self, by: float):
        if self.selection == Selection.ALIGNMENT:
            self.config.alignment_weight += by
        elif self.selection == Selection.COHESION:
            self.config.cohesion_weight += by
        elif self.selection == Selection.SEPARATION:
            self.config.separation_weight += by

    def before_update(self):
        super().before_update()

        for event in pg.event.get():
            if event.type == pg.KEYDOWN:
                if event.key == pg.K_UP:
                    self.handle_event(by=0.1)
                elif event.key == pg.K_DOWN:
                    self.handle_event(by=-0.1)
                elif event.key == pg.K_1:
                    self.selection = Selection.ALIGNMENT
                elif event.key == pg.K_2:
                    self.selection = Selection.COHESION
                elif event.key == pg.K_3:
                    self.selection = Selection.SEPARATION

        a, c, s = self.config.weights()
        print(f"A: {a:.1f} - C: {c:.1f} - S: {s:.1f}")

sim=(
    FlockingLive(
        FlockingConfig(
            image_rotation=True,
            movement_speed=1,
            radius=50,
            seed=1,
        )
    )
)

sim.batch_spawn_agents(50, Bird, images=["images/bird.png"])
sim.spawn_obstacle("images/triangle@200px.png", 400, 300)  # Create an obstacle at position (400, 300)
sim.run()
