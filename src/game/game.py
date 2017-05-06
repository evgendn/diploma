import collections
import os
import sys
import pygame
from random import randint
from pygame.locals import *

from game_utils import *


WINDOW_WIDTH = 404
WINDOW_HEIGHT = 500


class Bird(pygame.sprite.Sprite):
    GRAVITY = 0.6
    STANDART_SPEED = 0.4

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.velocity = Bird.STANDART_SPEED
        self.limit_speed = -8
        self.is_lift = False
        self.image = load_png("bluebird-midflap.png")
        self.rect = self.image.get_rect()

    def update(self):
        self.velocity += Bird.GRAVITY
        self.y += self.velocity

    def lift(self):
        self.velocity -= Bird.STANDART_SPEED * 20
        if self.velocity < self.limit_speed:
            self.velocity = self.limit_speed


class Pipe(pygame.sprite.Sprite):
    SCROLL_SPEED = 3
    GAP = 100
    FLUCTUATION = 150
    LOWEST_BORDER = 160
    WIDTH = 52

    def __init__(self, x):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = {}
        self.velocity = 0.1
        self.top_image = pygame.transform.rotate(
            load_png("pipe-red.png"), 180)
        self.bottom_image = load_png("pipe-red.png")
        self.reposition()

    def update(self):
        self.x -= Pipe.SCROLL_SPEED

        if self.x + Pipe.WIDTH < 0:
            self.reposition()
            self.x = WINDOW_WIDTH

    def reposition(self):
        self.y["bottom"] = randint(1, Pipe.FLUCTUATION) + Pipe.GAP + Pipe.LOWEST_BORDER
        self.y["top"] = self.y["bottom"] - Pipe.GAP - self.bottom_image.get_height()

class Game():
    def __init__(self):
        pass


def main():
    # Screen init.
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird clone")
    pygame.mouse.set_visible(False)

    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill((0, 0, 0))

    # Game objects
    bird = Bird(20, WINDOW_HEIGHT / 2)
    pipes = [Pipe(WINDOW_WIDTH),
             Pipe(WINDOW_WIDTH + Pipe.WIDTH + Pipe.GAP),
             Pipe(WINDOW_WIDTH + (Pipe.WIDTH + Pipe.GAP) * 2)
        ]
    clock = pygame.time.Clock()

    screen.blit(background, (0, 0))
    pygame.display.flip()

    # Event loop
    while True:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == QUIT:
                return
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    bird.lift()

        # Check collision
        if bird.y + bird.image.get_height() > WINDOW_HEIGHT:
            bird.y = WINDOW_HEIGHT - bird.image.get_height()
            bird.velocity = 0
        elif bird.y < 0:
            bird.y = 0
            bird.velocity = 0
        # Check score

        # Update bird and pipes
        bird.update()
        for pipe in pipes:
            pipe.update()


        # Check game over

        # Redraw objects
        screen.fill((0, 0, 0))
        screen.blit(bird.image, (bird.x, bird.y))
        for pipe in pipes:
            screen.blit(pipe.top_image, (pipe.x, pipe.y["top"]))
            screen.blit(pipe.bottom_image, (pipe.x, pipe.y["bottom"]))

        # Update display
        pygame.display.flip()


if __name__ == "__main__":
    main()
