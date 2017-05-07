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

    def update(self):
        self.velocity += Bird.GRAVITY
        self.y += self.velocity

    def lift(self):
        self.velocity -= Bird.STANDART_SPEED * 20
        if self.velocity < self.limit_speed:
            self.velocity = self.limit_speed

    def rect(self):
        return Rect(self.x, self.y,
                    self.image.get_width(), self.image.get_height())


class Pipe(pygame.sprite.Sprite):
    SCROLL_SPEED = 3
    GAP = 100
    FLUCTUATION = 150
    LOWEST_BORDER = 160
    WIDTH = 52
    HEIGHT = 320

    def __init__(self, x):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = {}
        self.velocity = 0.1
        self.top_image = pygame.transform.rotate(
            load_png("pipe-red.png"), 180)
        self.bottom_image = load_png("pipe-red.png")
        self.reposition()
        # Flag for scor counting
        self.bird_passed = False

    def update(self):
        self.x -= Pipe.SCROLL_SPEED

        if self.x + Pipe.WIDTH < 0:
            self.reposition()
            self.x = WINDOW_WIDTH

    def reposition(self):
        self.bird_passed = False
        self.y["bottom"] = randint(1, Pipe.FLUCTUATION) + Pipe.GAP + Pipe.LOWEST_BORDER
        self.y["top"] = self.y["bottom"] - Pipe.GAP - self.bottom_image.get_height()

    def rect(self):
        top_rect = Rect(self.x, self.y["top"],
                        Pipe.WIDTH, Pipe.HEIGHT)
        bottom_rect = Rect(self.x, self.y["bottom"],
                           Pipe.WIDTH, Pipe.HEIGHT)
        return top_rect, bottom_rect


def main():
    # Screen init.
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Flappy Bird clone")
    pygame.mouse.set_visible(False)

    color = (0, 0, 0)
    background = pygame.Surface(screen.get_size())
    background = background.convert()
    background.fill(color)

    # Game objects
    bird = Bird(20, WINDOW_HEIGHT / 2)
    pipes = [Pipe(WINDOW_WIDTH),
             Pipe(WINDOW_WIDTH + Pipe.WIDTH + Pipe.GAP),
             Pipe(WINDOW_WIDTH + (Pipe.WIDTH + Pipe.GAP) * 2)
        ]
    clock = pygame.time.Clock()
    score = 0

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
        # Window border and bird
        if bird.y + bird.image.get_height() > WINDOW_HEIGHT:
            bird.y = WINDOW_HEIGHT - bird.image.get_height()
            bird.velocity = 0
        elif bird.y < 0:
            bird.y = 0
            bird.velocity = 0

        # Pipes and bird
        collision = False
        for pipe in pipes:
            top_rect, bottom_rect = pipe.rect()
            if bird.rect().colliderect(top_rect)\
               or bird.rect().colliderect(bottom_rect):
                collision = True

        if collision:
            pass

        # Check score
        # Check position bird and pipes and increment
        # score of pipe position less then bird.
        # Shitty solution i suppose.
        for pipe in pipes:
            if pipe.bird_passed is False:
                if pipe.x + Pipe.WIDTH < bird.x:
                    score += 1
                    pipe.bird_passed = True

        print(score)

        # Update bird and pipes
        bird.update()
        for pipe in pipes:
            pipe.update()


        # Check game over

        # Redraw objects
        screen.fill(color)
        screen.blit(bird.image, (bird.x, bird.y))
        for pipe in pipes:
            screen.blit(pipe.top_image, (pipe.x, pipe.y["top"]))
            screen.blit(pipe.bottom_image, (pipe.x, pipe.y["bottom"]))

        # Update display
        pygame.display.flip()


if __name__ == "__main__":
    main()
