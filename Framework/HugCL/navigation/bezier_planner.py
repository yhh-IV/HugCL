import numpy as np

class LocalPlanner(object):
    def __init__(self):
        return
    
    def path_generation(self, start, temp1, temp2, target, junction_check):
        if not junction_check:
            Ps = np.array([[start.x, start.y], [temp1.x, temp1.y], [temp2.x, temp2.y], [target.x, target.y]])
        else:
            Ps = np.array([[start.x, start.y], [temp1[0], temp1[1]], [temp2[0], temp2[1]], [target[0], target[1]]])
        path = []
        
        for t in np.arange(0.1, 1.2, 0.1):
            p_t = self.bezier(Ps, len(Ps), t)
            path.append(p_t)
        path = np.array(path)    
        path = np.round(path, decimals=6)
        return path  
    
    
    def bezier(self, Ps, n, t):
        if n==1:
            return Ps[0]
        return (1-t) * self.bezier(Ps[0:n-1], n-1, t) + t * self.bezier(Ps[1:n], n-1, t)
