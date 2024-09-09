import pybullet as p
import pybullet_data


class HomeEnvironment:
    def __init__(self):
        # Start pybullet and connect to the simulation
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        # Load the plane (ground)
        self.plane = p.loadURDF("plane.urdf")

        # Create the house with connected walls
        self.house_position = [0, 0, 0]
        self.create_house()

        # Create a front lawn in front of the house
        self.create_front_lawn()

        # Create a landing zone in front of the house, just near the door
        self.landing_zone_position = [0, 4, 0]
        self.landing_zone_size = [1, 1, 0.01]
        self.create_landing_zone()

        # Create a parking area
        self.parking_position = [-2, 2, 0]
        self.parking_size = [2, 2, 0.01]
        self.create_parking_area()

    def create_house(self):
        # House walls (connected walls with a door opening)
        wall_thickness = 0.1
        wall_height = 2
        wall_length = 5
        wall_width = 5

        # Front wall with a door (door at [0, 0])
        front_wall_position = [0, -wall_length / 2, wall_height / 2]
        door_width = 1.0  # Width of the door opening
        front_wall_size = [wall_width / 2 - door_width / 2, wall_thickness, wall_height / 2]  # Leave space for the door

        # Left section of front wall
        self.create_wall([-door_width / 2 - wall_width / 4, -wall_length / 2, wall_height / 2], front_wall_size)

        # Right section of front wall
        self.create_wall([door_width / 2 + wall_width / 4, -wall_length / 2, wall_height / 2], front_wall_size)

        # Create other connected walls
        self.create_wall([0, wall_length / 2, wall_height / 2],
                         [wall_width / 2, wall_thickness, wall_height / 2])  # Back wall
        self.create_wall([-wall_width / 2, 0, wall_height / 2],
                         [wall_thickness, wall_length / 2, wall_height / 2])  # Left wall
        self.create_wall([wall_width / 2, 0, wall_height / 2],
                         [wall_thickness, wall_length / 2, wall_height / 2])  # Right wall

        # Roof (flat for simplicity)
        roof_position = [0, 0, wall_height + 0.2]
        roof_size = [wall_width / 2, wall_length / 2, 0.1]
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=roof_size,
                                              rgbaColor=[0.5, 0.5, 0.5, 1])  # Gray roof
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=roof_size)
        self.roof = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id,
                                      basePosition=roof_position)

    def create_wall(self, position, half_extents):
        # Create a simple box shape for the wall
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=half_extents,
                                              rgbaColor=[1, 0.8, 0.6, 1])  # Light brown for walls
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=half_extents)
        return p.createMultiBody(baseMass=0,
                                 baseCollisionShapeIndex=collision_shape_id,
                                 baseVisualShapeIndex=visual_shape_id,
                                 basePosition=position)

    def create_landing_zone(self):
        # Create a simple flat box to represent the landing zone
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=self.landing_zone_size,
                                              rgbaColor=[0, 1, 0, 1])  # Green color for landing zone
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=self.landing_zone_size)
        self.landing_zone = p.createMultiBody(baseMass=0,
                                              baseCollisionShapeIndex=collision_shape_id,
                                              baseVisualShapeIndex=visual_shape_id,
                                              basePosition=self.landing_zone_position)

    def create_parking_area(self):
        # Create a simple flat box to represent the parking area
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=self.parking_size,
                                              rgbaColor=[1, 0, 0, 1])  # Red color for parking area
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=self.parking_size)
        self.parking_area = p.createMultiBody(baseMass=0,
                                              baseCollisionShapeIndex=collision_shape_id,
                                              baseVisualShapeIndex=visual_shape_id,
                                              basePosition=self.parking_position)

    def create_front_lawn(self):
        # Create a flat box for the lawn in front of the house
        lawn_position = [0, 2.5, 0]  # In front of the house
        lawn_size = [3, 3, 0.01]  # Slightly larger than the landing zone
        visual_shape_id = p.createVisualShape(shapeType=p.GEOM_BOX,
                                              halfExtents=lawn_size,
                                              rgbaColor=[0.3, 1, 0.3, 1])  # Green color for lawn
        collision_shape_id = p.createCollisionShape(shapeType=p.GEOM_BOX,
                                                    halfExtents=lawn_size)
        self.lawn = p.createMultiBody(baseMass=0,
                                      baseCollisionShapeIndex=collision_shape_id,
                                      baseVisualShapeIndex=visual_shape_id,
                                      basePosition=lawn_position)

    def reset_environment(self):
        # Reset the environment to the initial state
        p.resetBasePositionAndOrientation(self.house, self.house_position, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.landing_zone, self.landing_zone_position, [0, 0, 0, 1])
        p.resetBasePositionAndOrientation(self.parking_area, self.parking_position, [0, 0, 0, 1])

    def step_simulation(self):
        # Step through the simulation
        p.stepSimulation()

    def close(self):
        # Disconnect the simulation
        p.disconnect()

import time

# Initialize the environment
env = HomeEnvironment()

# Run the simulation for a few seconds
for i in range (1000):
    env.step_simulation()
    time.sleep(1./240.)  # Slow down the simulation to a realistic pace

# Close the environment when done
env.close()
