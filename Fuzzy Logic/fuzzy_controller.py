import numpy as np

class FuzzyController:
    
    def __init__(self):
        self.range = np.linspace(-50 , 50 , 10000)#integral range
        self.step_delta = 0.01 # 50 -(-50) / 10000, this is our dx
        self.max_rotate = 0.0 #we will use it for maximum compotation!
        self.sum1 = 0.0
        self.sum2 = 0.0

    def decide(self, left_dist, right_dist):
        #for solving problem with fuzzy logic, at first, se have transmit amounts from absolute state to a fuzzy state. 
        moderate_L, close_L, far_L, moderate_R, close_R, far_R = self.fuzzify(left_dist, right_dist)
        low_left, high_left, low_right, high_right, nothing = self.inference(moderate_L, close_L, far_L, moderate_R, close_R, far_R)
        return self.defuzzify(low_left, high_left, low_right, high_right, nothing)
     
    # In fuzzify FUNCTION, the left and right distances of the car are taken as input.
    #then, a membership function is defined for each distance based on three memberships of close, medium, and far!
    # membership function is working based on ./plots/fuzzify_plot.png
    def fuzzify(self, left_dist, right_dist):

        #left distance section
        left_moderate = 0.0
        left_close = 0.0
        left_far = 0.0

        if 35 <= left_dist <= 50:
            left_moderate = 1 / 15 * left_dist - 7 / 3
        elif 50 < left_dist <= 65:
            left_moderate = -1 / 15 * left_dist + 13 / 3
        if 0 <= left_dist <= 50: left_close = -0.02 * left_dist + 1
        if 50 <= left_dist <= 100: left_far = 0.02 * left_dist - 1

        #right distance section
        right_moderate = 0.0
        right_close = 0.0
        right_far = 0.0
       
        if 0 <= right_dist <= 50: right_close = -0.02 * right_dist + 1
        
        if 35 <= right_dist <= 50:
            left_moderate = 1 / 15 * left_dist - 7 / 3
        elif 50 < right_dist <= 65:
            left_moderate = -1 / 15 * left_dist + 13 / 3
        
        if 50 <= right_dist <= 100:
            right_far = 0.02 * right_dist - 1
        
        return (left_moderate, left_close, left_far, right_moderate, right_close, right_far)

    # this function will return the rules with usage of rules.txt ...
    def inference(self, moderate_L, close_L, far_L, moderate_R, close_R, far_R):
        return (
            min(moderate_L, close_R),  # low_left
            min(far_L, close_R),  # high_left
            min(moderate_R, close_L),  # low_right
            min(far_R, close_L),  # high_right
            min(moderate_L, moderate_R)  # nothing
        )

    def defuzzify(self, low_left , high_left , low_right , high_right , nothing):
    
        for i in self.range:
            #finding the maximum of functions ...
            updated_max_rotate = 0.0
            if -50 <= i <= -20: updated_max_rotate = max(min(1 / 30 * i + 5 / 3, high_right),updated_max_rotate)
            elif -20 < i <= -5 :  updated_max_rotate = max(min(-1 / 15 * i - 1 / 3, high_right),updated_max_rotate)
            if -20 <= i <= -10: updated_max_rotate = max(min(1 / 10 * i + 2, low_right),updated_max_rotate)
            if -10 < i <= 0: updated_max_rotate = max(min(-1 / 10 * i,low_right),updated_max_rotate)         

            if -10 <= i <= 0: updated_max_rotate = max(min(1 / 10 * i + 1, nothing),updated_max_rotate )  
            if 0 < i <= 10 : updated_max_rotate = max(min(-1 / 10 * i + 1, nothing),updated_max_rotate )
    
            if 10 < i <= 20 : updated_max_rotate = max(min(-1 / 10 * i + 2, low_left),updated_max_rotate )
            if 0 <= i <= 10 : updated_max_rotate = max(min(1 / 10 * i , low_left),updated_max_rotate)

            if 20 < i <= 50 : updated_max_rotate =  max(min(-1 / 30 * i + 5 / 3,high_left),updated_max_rotate)
            if 5 <= i <= 20 : updated_max_rotate = max(min(1 / 15 * i - 1 / 3, high_left),updated_max_rotate )         

            #updating properties by new maximum
            self.sum2 += updated_max_rotate* self.step_delta  
            self.sum1 += updated_max_rotate * self.step_delta  * i
            self.max_rotate = updated_max_rotate
    
        #calculating the center!
        try:
            return self.sum1 / self.sum2
        except:
            return 0.0