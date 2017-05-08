import pygame
from random import randint
from pygame.locals import *

from game_utils import *


class Bird(pygame.sprite.Sprite):
    GRAVITY = 0.6
    STANDART_SPEED = 0.4

    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = y
        self.velocity = Bird.STANDART_SPEED
        # For stopping bird
        self.moving = True
        self.limit_speed = -8
        self.image = load_png("bluebird-midflap.png")
        # For reset game
        self.old_x = x
        self.old_y = y

    def update(self):
        if self.moving:
            self.velocity += Bird.GRAVITY
            self.y += self.velocity

    def lift(self):
        self.velocity -= Bird.STANDART_SPEED * 20
        if self.velocity < self.limit_speed:
            self.velocity = self.limit_speed

    def rect(self):
        return Rect(self.x, self.y,
                    self.image.get_width(), self.image.get_height())

    def reset(self):
        self.x = self.old_x
        self.y = self.old_y
        self.velocity = Bird.STANDART_SPEED
        self.moving = True


class Pipe(pygame.sprite.Sprite):
    GAP = 100
    FLUCTUATION = 150
    LOWEST_BORDER = 160
    WIDTH = 52
    HEIGHT = 320

    def __init__(self, x):
        pygame.sprite.Sprite.__init__(self)
        self.x = x
        self.y = {}
        self.velocity = 3
        # For stopping Pipes
        self.moving = True
        self.top_image = pygame.transform.rotate(
            load_png("pipe-red.png"), 180)
        self.bottom_image = load_png("pipe-red.png")
        self.reposition()
        # Flag for score counting
        self.bird_passed = False
        # For reset game
        self.old_x = self.x
        self.old_y = self.y

    def update(self):
        if self.moving:
            self.x -= self.velocity

            if self.x + Pipe.WIDTH < 0:
                self.reposition()
                self.x = Game.WINDOW_WIDTH

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

    def reset(self):
        self.x = self.old_x
        self.y = self.old_y
        self.moving = True
        self.bird_passed = False


class Game:
    WINDOW_WIDTH = 404
    WINDOW_HEIGHT = 500

    def __init__(self):
        # Screen init.
        pygame.init()
        self.screen = pygame.display.set_mode((Game.WINDOW_WIDTH,
                                               Game.WINDOW_HEIGHT))
        pygame.display.set_caption("Flappy Bird clone")
        pygame.mouse.set_visible(False)

        # Game objects
        self.bird = Bird(20, Game.WINDOW_HEIGHT / 2)
        self.pipes = [Pipe(Game.WINDOW_WIDTH),
                      Pipe(Game.WINDOW_WIDTH + Pipe.WIDTH + Pipe.GAP),
                      Pipe(Game.WINDOW_WIDTH + (Pipe.WIDTH + Pipe.GAP) * 2)
                      ]
        self.clock = pygame.time.Clock()
        self.score = 0
        self.start = False
        self.background = pygame.transform.scale(load_png("background-day.png"),
                                                 (Game.WINDOW_WIDTH, Game.WINDOW_HEIGHT))
        self.score_numbers = [load_png("{}.png".format(i)) for i in range(10)]

        pygame.display.flip()

    def main_loop(self):
        # Event loop
        while True:
            self.clock.tick(30)
            for event in pygame.event.get():
                if event.type == QUIT:
                    return
                elif event.type == KEYDOWN:
                    # Core gameplay
                    if event.key == K_SPACE:
                        self.bird.lift()
                    # Reset game
                    elif event.key == K_r:
                        self.bird.reset()
                        for pipe in self.pipes:
                            pipe.reset()
                        self.score = 0

            # Check collision
            if self.check_collision():
                self.stop()

            # Check score
            # Check position bird, pipes and increment
            # score if pipe position less then bird.
            # Shitty solution i suppose.
            for pipe in self.pipes:
                if pipe.bird_passed is False:
                    if pipe.x + Pipe.WIDTH < self.bird.x:
                        self.score += 1
                        pipe.bird_passed = True

            # Update bird and pipes
            self.bird.update()
            for pipe in self.pipes:
                pipe.update()

            # Redraw objects
            # self.screen.fill((0, 0, 0))
            self.screen.blit(self.background, (0, 0))
            self.screen.blit(self.bird.image,
                             (self.bird.x, self.bird.y))
            for pipe in self.pipes:
                self.screen.blit(pipe.top_image, (pipe.x, pipe.y["top"]))
                self.screen.blit(pipe.bottom_image, (pipe.x, pipe.y["bottom"]))
            self.draw_score()

            # Update display
            pygame.display.flip()

    def check_collision(self):
        # Window border and bird
        if self.bird.y + self.bird.image.get_height() > Game.WINDOW_HEIGHT:
            self.bird.y = Game.WINDOW_HEIGHT - self.bird.image.get_height()
            return True
        elif self.bird.y < 0:
            self.bird.y = 0
            return True

        # Pipes and bird
        for pipe in self.pipes:
            top_rect, bottom_rect = pipe.rect()
            if self.bird.rect().colliderect(top_rect) \
                    or self.bird.rect().colliderect(bottom_rect):
                return True

    def stop(self):
        self.bird.moving = False
        for pipe in self.pipes:
            pipe.moving = False

    def draw_score(self):
        splitted_score = [int(i) for i in list(str(self.score))]
        # Almost all sprites has width 24 -> half is 12
        width = self.score_numbers[0].get_width()
        offset = (len(splitted_score) * width) / 2
        # Position of score numbers
        x = (Game.WINDOW_WIDTH / 2) - offset
        y = int(Game.WINDOW_HEIGHT / 7)

        for number in splitted_score:
            self.screen.blit(self.score_numbers[number], (x, y))
            x += width


if __name__ == "__main__":
    game = Game()
    game.main_loop()
