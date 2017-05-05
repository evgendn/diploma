import os
import sys
import pygame
from pygame.locals import *

from game_utils import *


WINDOW_WIDTH = 480
WINDOW_HEIGHT = 600


class Bird(pygame.sprite.Sprite):
    GRAVITY = 0.5
    STANDART_SPEED = 0.8

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
        print(self.velocity)


class Pipe(pygame.sprite.Sprite):
    SCROLL_SPEED = 1

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.velocity = 0.1

    def update(self):
        pass


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

        # Redraw objects
        screen.fill((0, 0, 0))
        screen.blit(bird.image, (bird.x, bird.y))

        # Update objects
        bird.update()

        # Update display
        pygame.display.flip()


if __name__ == "__main__":
    main()
