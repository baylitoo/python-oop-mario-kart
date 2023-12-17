import pygame
import numpy as np
CELL_SIZE=50

MAX_ANGLE_VELOCITY = 0.05
MAX_ACCELERATION = 0.25
V_boost = 25  # Param global pour la surface boost
from pygame import mixer


# Physics Class that calculates the new x, y, and velocity values based on the effects of the current surface the kart is on.
# This class is used as inherence for Kart
class Physics:

    def __init__(self):
        # Initialize the physics object with starting x and y coordinates.
        mixer.init()  # Initialize the mixer
        self.__checkpoint_sound = mixer.Sound("the-notification-email-143029.mp3") #Checkpoint sound
        self.__Boost_sound=mixer.Sound("power-up-sparkle-1-177983.mp3") # Boost sound

        self.__finish_line_sound=mixer.Sound("success-fanfare-trumpets-6185.mp3") # Finish line sound
        self._x = 0
        self._y = 0
        self._angle = 0  # θ(t)
        self._velocity = 0
        self._ac = 0                    # Acceleration
        self.__acceleration = 0
        self._angle_velocity = 0        # Angular velocity
        self.__f = 0.02  # Default friction coefficient
        self._next_checkpoint_id = 0
        self._xc = 50  # x-coordinate of last checkpoint
        self._yc = 50  # y-coordinate of last checkpoint
        self._θc = 0  # Orientation at last checkpoint
        self._got_into_lava_or_Off_limits= False  # Parameter to check if the kart got into lava or off limits for Respan animation animation
        self.boost=False  # Parameter to check if the kart is Boosted
        self._Off_L= False # Parameter to check if the kart is off limits



    # Calculates new position and velocity of the kart.
    def __Calculates_New_Positions(self):
        # Calculate acceleration based on acceleration constant, friction, and velocity.
        # Adjust the velocity and apply a maximum velocity limit.

        # Condition to check if boost is activated
        if self.boost==True:
            self._velocity = min(self._velocity, 25)
            # Update position based on velocity and angle.
            vx = self._velocity * np.cos(self._angle)
            vy = self._velocity * np.sin(self._angle)

            # Uptading the x,y positions
            self._x += vx
            self._y += vy

        else:

            #Conditons that helps Reset the angle once it does a 360 degree
            #if self._angle>=6 or self._angle<=-6:
                #self._angle=0
            #else:
            self._angle += self._angle_velocity

            self.__acceleration = self._ac - self.__f * self._velocity * np.cos(self._angle_velocity)
            self._velocity = self.__acceleration + self._velocity * np.cos(self._angle_velocity)
            self._velocity = min(self._velocity, 25)  # Maximum velocity cap

            # Update position based on velocity and angle.
            vx = self._velocity * np.cos(self._angle)
            vy = self._velocity * np.sin(self._angle)

            #Uptading the x,y positions
            self._x += vx
            self._y += vy

        #Resetting the friction coef
        self.__f=0.02
        self.__acceleration = 0
        self._ac = 0
        self._angle_velocity = 0



    # Update checkpoint ID and handle checkpoint logic.
    def _physic_checkpoint(self, current_surface, TOTAL_CHECKPOINTS):
        # Map of checkpoint identifiers to their respective IDs.
        checkpoint_id_map = {'C': 0, 'D': 1, 'E': 2, 'F': 3}

        checkpoint_id = checkpoint_id_map[current_surface]
        # Check if the current checkpoint is the next in sequence.
        if checkpoint_id == self._next_checkpoint_id:
            print(f"{checkpoint_id}={TOTAL_CHECKPOINTS}")
            if checkpoint_id == TOTAL_CHECKPOINTS:
                self.__finish_line_sound.play()

                self.has_finished = True
                print(5555)

            else:
                # Update the checkpoint coordinates and angle.
                self._xc = self._x
                self._yc = self._y
                self._θc = self._angle
                self._next_checkpoint_id += 1  # Proceed to the next checkpoint ID.
                self.__checkpoint_sound.play()

        self.__Calculates_New_Positions()  # Update position and velocity.


    # Adjust friction when on grass surface and update positions.
    def _physic_grass(self):
        self.__f = 0.2  # Increase friction on grass.
        self.__Calculates_New_Positions()  # Update position and velocity.

    # Handle kart behavior when on lava surface.
    def _physic_lava(self):
        # Respawn kart at the last passed checkpoint.
        self._x = self._xc
        self._y = self._yc
        self._angle = self._θc
        self._velocity = 0
        self._got_into_lava_or_Off_limits = True

    # Boost the kart's velocity on boost surfaces.
    def _physic_boost(self):
        self._velocity = 25  # Maximize velocity.
        self.__Calculates_New_Positions()  # Update position.
        self.__Boost_sound.play() # Play boost sound

    # Adjust friction and update positions when on road surface.
    def _physic_road(self):
        self.__f = 0.02  # Reduced friction on road.
        self.__Calculates_New_Positions()  # Update position and velocity.

    # Off limits fonction that checks if the kart is out of the field , and respawns it to the last saved checkpoint position
    def _Off_limits(self, track_string):
        # The track string is split into lines, representing the layout of the track.
        track_lines = track_string.split('\n')
        track_height = len(track_lines)  # Determine the height of the track in terms of lines.
        track_width = len(
            track_lines[0]) if track_height > 0 else 0  # Determine the width of the first line as track width.

        # Check if the kart is out of track boundaries and reset its position if it is.
        # If the kart's x or y coordinate is beyond the track limits, it is reset to the last checkpoint.
        if self._x < 0 or self._y < 0 or self._x > track_width * CELL_SIZE or self._y > track_height * CELL_SIZE:
            self._x = self._xc  # Reset x to the last checkpoint's x-coordinate.
            self._y = self._yc  # Reset y to the last checkpoint's y-coordinate.
            self._angle = self._θc  # Reset the angle to the last checkpoint's angle.
            self._velocity = 0  # Reset the velocity to 0.
            self._got_into_lava_or_Off_limits = True # Activate the animation counter
            self._Off_L = True # Activate off track parameter






