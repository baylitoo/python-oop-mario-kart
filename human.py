import time
import pygame

class Human():

    def __init__(self):
        self.kart = None
        self.__Name="Human"

    # Current surface getter
    @property
    def name(self):
        return self.__Name

    def move(self, string):
        time.sleep(0.02)
        return pygame.key.get_pressed()