from scenic.domains.driving.roads import Network
model scenic.domains.driving.model
from rulebook_benchmark.realization import Realization, RealizationObject, State
import numpy as np

monitor bench():
    realization = globalParameters['realization']
    realization.network = Network.fromFile(globalParameters['map'])
    realization.isScenic = True
    objects = simulation().objects
    #max_steps = realization.max_steps
    objs = []
    ids = 0
    for obj in objects:
        objs.append(RealizationObject(ids, (obj.length, obj.width), type(obj).__name__))
        ids += 1
    realization.objects = objs

    step = 0
    while True:
        objects = simulation().objects

        for i in range(len(objects)):
            obj = realization.objects[i]
            object = objects[i]
            yaw = object.orientation.yaw + np.pi / 2
            yaw = (yaw + np.pi) % (2 * np.pi) - np.pi  # Normalize to [-pi, pi]
            orient = Orientation.fromEuler(yaw, object.orientation.pitch, object.orientation.roll)
            obj.trajectory.append(State(obj, np.array([object.position.x, object.position.y]), np.array([object.velocity.x, object.velocity.y]), orient, step))
        step += 1
        wait
