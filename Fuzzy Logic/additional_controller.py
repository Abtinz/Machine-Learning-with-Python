import numpy as np

VELOCITY = 30

class FuzzyGasController:
   
    def __init__(self):
        self.range = np.linspace(0 , 90 , 1000)#integral range
        self.step_delta = 0.09 # 90 / 1000, this is our dx
        self.max_speed = 0.0 #we will use it for maximum compotation!
        self.sum1 = 0.0
        self.sum2 = 0.0
        

    def decide(self, center_dist):
        close, moderate, far = self.fuzzify(center_dist)
        low, medium, high = self.inference(close , moderate,  far)
        return self.defuzzify(low, medium, high)
    
    def fuzzify(self, center_dist):

        #center dist
        moderate = 0.0
        close = 0.0
        far = 0.0

        if 40 <= center_dist <= 50:
            moderate = 1/10 * center_dist - 4
        elif 50 < center_dist <= 100:
            moderate = -1 / 50 * center_dist + 2
        if 0 <= center_dist <= 50: close = -0.02 * center_dist + 1
        if 90 <= center_dist <= 200: far = 1/110 * center_dist - 9/11

        return (close, moderate, far)
    
    def inference(self, close , moderate,  far):
        return (
            min(close, 1.0),
            min(moderate, 1.0),
            min(far, 1.0)
        )

    def defuzzify(self, low, medium, high):
    
        for i in self.range:
            #finding the maximum of functions ...
            updated_max_speed = 0.0

            #
            if 25 <= i <= 30: updated_max_speed = max(min(1/5 * i - 5, high),updated_max_speed)
            if 30 < i <= 90 :  updated_max_speed = max(min(-1/60 * i + 3 / 2, high),updated_max_speed)

            #moderate
            if 0 <= i <= 15: updated_max_speed = max(min(1 / 15 * i, medium),updated_max_speed)
            if 15 < i <= 30: updated_max_speed = max(min(-1/15* i + 2,medium),updated_max_speed) 

            #close        
            if 0 <= i <= 5: updated_max_speed = max(min(1/5 * i , low),updated_max_speed )  
            if 5 < i <= 10 : updated_max_speed = max(min(-1/5* i + 2, low),updated_max_speed )
    
            #updating properties by new maximum
            self.sum2 += updated_max_speed* self.step_delta  
            self.sum1 += updated_max_speed * self.step_delta  * i
            self.max_speed = updated_max_speed
    
        #calculating the center!
        try:
            return int(self.sum1 / self.sum2)
        except:
            return VELOCITY