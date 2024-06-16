"""
Self Driving Car Simulator using PyGame module
"""
import sys
import time
import math
import pygame
import additional_controller
import fuzzy_controller

pygame.init()
pygame.display.set_caption("SELF DRIVING CAR SIMULATOR")
WINDOW_SIZE = 1280, 720
SCREEN = pygame.display.set_mode(WINDOW_SIZE)
CAR_SIZE =  40,60
CAR_CENTER = 105, 250
DELTA_DISTANCE = 30
DELTA_ANGLE = 15
WHITE_COLOR = (93, 240, 77)
TRACK = pygame.image.load('./images/map.png').convert_alpha()
TRACK_COPY = TRACK.copy()
FONT = pygame.font.SysFont("bahnschrift", 25)
CLOCK = pygame.time.Clock()
GENERATION = 0
mute_btn = pygame.image.load("./images/mute.png").convert_alpha()
mute_btn=pygame.transform.scale(mute_btn, (40,40))
rect_mute = mute_btn.get_rect(center=(1200,70))


def translate_point(point, angle, distance):
    """
    Get the new co-ordinates of a given point w.r.t an angle and distance from that point

    Args:
        center (tuple): A tuple of x co-ordinate and y co-ordinate
        angle (int): Angle of rotation of the vector
        distance (float): The distance by which the point needs
        to be translated (magnitude of the vector)

    Returns:
        tuple: Translated co-ordinates of the point
    """
    radians = math.radians(angle)
    return int(point[0] + distance * math.cos(radians)),\
        int(point[1] + distance * math.sin(radians))

class Car:
    """
    Implentation of the self driving car
    """
    def __init__(self,music_mode):
        self.corners = []
        self.edge_points = []
        self.edge_distances = []
        self.travelled_distance = 0
        self.angle = 0
        self.car_center = CAR_CENTER
        self.car = pygame.image.load("./images/car.png").convert_alpha()
        self.car = pygame.transform.scale(self.car, CAR_SIZE)
        self.crashed = False
        self.crash_sound = pygame.mixer.Sound("./sounds/crash.mp3")
        self.win_sound=pygame.mixer.Sound("./sounds/winning.mp3")
        self.rev_sound=pygame.mixer.Sound("./sounds/rev.mp3").get_raw()[700000:]
        self.rev_sound=pygame.mixer.Sound(buffer=self.rev_sound)
        self.background_sound=pygame.mixer.Sound("./sounds/background.mp3")
        self.set_sounds(music_mode)
        pygame.mixer.Sound.play(self.rev_sound)
        pygame.mixer.Sound.play(self.background_sound)
        self.update_sensor_data()

    def set_sounds(self,change_to):
        if change_to=="off":
            self.rev_sound.set_volume(0.0)
            self.background_sound.set_volume(0.0)
            self.win_sound.set_volume(0.0)
            self.crash_sound.set_volume(0.0)
        else:
            self.rev_sound.set_volume(0.25)
            self.background_sound.set_volume(1)
            self.win_sound.set_volume(1)
            self.crash_sound.set_volume(1)

    def display_car(self):
        """
        Rotate the car and the display it on the screen
        """
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        rect = rotated_car.get_rect(center=self.car_center)
        SCREEN.blit(rotated_car, rect.topleft)
        
        

    def crash_check(self):
        """
        Check if any corner of the car goes out of the track
        Returns:
            Bool: Returns True if the car is alive
        """
        for corner in self.corners:
            if TRACK.get_at(corner) == WHITE_COLOR:
                
                if 1099<=self.car_center[0] and self.car_center[0]<=1208 and self.car_center[1]>516:
                    pygame.mixer.pause()
                    pygame.mixer.Sound.play(self.win_sound)
                else:
                    pygame.mixer.pause()
                    pygame.mixer.Sound.play(self.crash_sound)
                return True
        return False

    def update_sensor_data(self):
        """
        Update the points on the edge of the track
        and the distances between the points and the center of the car
        """
        angles = [400 - self.angle, 90 - self.angle, 140 - self.angle]
        angles = [math.radians(i) for i in angles]
        edge_points = []
        edge_distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.car_center
            while TRACK_COPY.get_at((edge_x, edge_y)) != WHITE_COLOR:
                edge_x = int(self.car_center[0] + distance * math.cos(angle))
                edge_y = int(self.car_center[1] + distance * math.sin(angle))
                distance += 1
            edge_points.append((edge_x, edge_y))
            edge_distances.append(distance)
        self.edge_points = edge_points
        self.edge_distances = edge_distances


    def display_edge_points(self):
        """
        Display lines from center of the car to the edges on the  track
        """
        for point in self.edge_points:
            pygame.draw.line(SCREEN, (255, 0, 255), self.car_center, point)
            pygame.draw.circle(SCREEN, (255, 0, 255), point, 5)

    def update_position(self,output_sup_fuzzy):
        """
        Update the new position of the car
        """
        self.car_center = translate_point(
            self.car_center, 90 - self.angle, output_sup_fuzzy)
        self.travelled_distance += DELTA_DISTANCE
        dist = math.sqrt(CAR_SIZE[0]**2 + CAR_SIZE[1]**2)/2
        corners = []
        corners.append(translate_point(
            self.car_center, 60 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 120 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 240 - self.angle, dist))
        corners.append(translate_point(
            self.car_center, 300 - self.angle, dist))
        self.corners = corners


def run():
    """
    Runs the game
    """
    music_mode="on"
    with open("./sounds/setting.txt","r") as f:
        music_mode=f.readlines()[0].strip()

    car=Car(music_mode)
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    car.angle += DELTA_ANGLE
                if event.key == pygame.K_RIGHT:
                    car.angle -= DELTA_ANGLE
            if event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if rect_mute.collidepoint(x, y):
                    if music_mode=="off":
                        music_mode="on"
                        with open("./sounds/setting.txt","w") as f:
                            f.writelines("on")
                        car.set_sounds("on")
                    else:
                        music_mode="off"
                        with open("./sounds/setting.txt","w") as f:
                            f.writelines("off")
                        car.set_sounds("off")

        SCREEN.blit(TRACK, (0, 0))
        SCREEN.blit(mute_btn, rect_mute.topleft)
        # car.display_car()
        # pygame.display.update()
        # time.sleep(5)
        left_dist=math.sqrt((car.car_center[0]-car.edge_points[0][0])**2+(car.car_center[1]-car.edge_points[0][1])**2)
        center_dist=math.sqrt((car.car_center[0]-car.edge_points[1][0])**2+(car.car_center[1]-car.edge_points[1][1])**2)
        right_dist=math.sqrt((car.car_center[0]-car.edge_points[2][0])**2+(car.car_center[1]-car.edge_points[2][1])**2)
        relative_left_dist=float(100*left_dist/(left_dist+right_dist))
        relative_right_dist=float(100*right_dist/(left_dist+right_dist))
        center_dist=float(center_dist)
        if not car.crashed:
            rotate_fuzzy_system=fuzzy_controller.FuzzyController()
            output_of_fuzzy=rotate_fuzzy_system.decide(relative_left_dist,relative_right_dist)
            car.angle+=output_of_fuzzy

            gas_fuzzy_system=additional_controller.FuzzyGasController()

            car.update_position(gas_fuzzy_system.decide(center_dist))
            car.display_car()
            car.crashed = car.crash_check()
            car.update_sensor_data()
            car.display_edge_points()
        else:
            time.sleep(1)
            exit()
        
        text = FONT.render(f"Left:{relative_left_dist:.2f} , Center:{center_dist:.2f} , Right:{relative_right_dist:.2f} ,    "+
                           f"Output : {output_of_fuzzy:.2f}", True, (0, 0, 0))
        SCREEN.blit(text, (0, 0))
        pygame.display.update()
        CLOCK.tick(5)

run()