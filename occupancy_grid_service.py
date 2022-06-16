
import argparse
import logging
import time
import threading
from typing import List
import numpy as np
import numpy.ma as ma
import cv2 as cv

import bosdyn
import bosdyn.client
import bosdyn.client.util

from bosdyn.client.directory_registration import (DirectoryRegistrationClient,
                                                  DirectoryRegistrationKeepAlive)

from bosdyn.client.fault import FaultClient
from bosdyn.client.server_util import GrpcServiceRunner, populate_response_header
from bosdyn.client.world_object import WorldObjectClient

from bosdyn.client.util import setup_logging

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks

from bosdyn.api import local_grid_pb2, local_grid_service_pb2_grpc 

from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *
from bosdyn.api import geometry_pb2 as geom
from bosdyn.api import world_object_pb2

from skimage.draw import line


DIRECTORY_NAME = 'pcl-occupancy-grid'
AUTHORITY = 'pcl-occupancy-grid'
SERVICE_TYPE = 'bosdyn.api.LocalGridService'


LOGGER = logging.getLogger(__name__)


def get_pcl_in_frame(cloud, inertial, offset):
    """
    Create a 3 x N numpy array of points in the grid frame. 
    :return: a 3 x N numpy array in the seed frame.
    """
    #TODO: vision, odom, or ground plane?
    inertial_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, inertial,
                                     cloud.source.frame_name_sensor)
    grid_tform_inertial = SE3Pose.from_se2(offset)
    grid_tform_cloud = grid_tform_inertial*inertial_tform_cloud
    point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    return grid_tform_cloud.transform_cloud(point_cloud_data)



def _update_thread(async_task, stop_signal):
    while not stop_signal.is_set():
        async_task.update()
        time.sleep(0.01)


class AsyncPointCloud(AsyncPeriodicQuery):
    def __init__(self, point_cloud_client, pcl_sources):
        super(AsyncPointCloud, self).__init__("point_clouds", point_cloud_client, LOGGER,
                                              period_sec=0.2)
        self._sources = pcl_sources

    def _start_query(self):
        return self._client.get_point_cloud_from_sources_async(self._sources)


class AsyncRobotState(AsyncPeriodicQuery):
    def __init__(self, robot_state_client):
        super().__init__("robot_state", robot_state_client, LOGGER,
                                              period_sec=0.1)

    def _start_query(self):
        return self._client.get_robot_state_async()



def gen_grid_frametree(grid_anchor_x, grid_anchor_y, grid_anchor_angle, anchor_april_tag, apriltag_frametree):
    #we expect the fiducial to be posted on a wall, and its normal (z axis) defining the 
    # x direction of the grid

    anchortag_tform_grid = SE3Pose(position=Vec3(grid_anchor_x,grid_anchor_y,0),rotation=Quat.from_yaw(grid_anchor_angle*math.pi/180))

    frame_tree_edges = {}
    frame_tree_edges = add_edge_to_tree(frame_tree_edges, anchortag_tform_grid, f'filtered_fiducial_{anchor_april_tag:03}', GRID_FRAME_NAME)

    for edge in apriltag_frametree.child_to_parent_edge_map:
        frame_tree_edges = add_edge_to_tree(frame_tree_edges, edge.parent_tform_child*anchortag_tform_grid, edge.parent_frame_name, GRID_FRAME_NAME)

    # Pack the dictionary into a FrameTreeSnapshot proto message.
    return geom.FrameTreeSnapshot(child_to_parent_edge_map=frame_tree_edges)


#constantly update the apriltag frame-tree in memory
#all we really care about though is the x/y/theta
def find_anchor_fiducial_transform_snapshot(apriltag_world_objects, anchor_april_tag):
    for apriltag in apriltag_world_objects:
        if apriltag.apriltag_properties.tag_id != anchor_april_tag:
            continue
        if apriltag.apriltag_properties.fiducial_filtered_pose_status != world_object_pb2.AprilTagProperties.AprilTagPoseStatus.STATUS_OK:
            raise RuntimeError('fiducial location not accurately know')
        # Look for the custom frame that was included in the add-request, where the child frame name was "my_special_frame"
        return apriltag.transforms_snapshot
    raise KeyError('fiducial not found')
        
            
class OccupancyGrid:
    OCCUPIED=0
    UNKNOWN=0x80
    FREE=0xff
    def __init__(self, grid_width, grid_height, grid_scale, occupancy_threshold,
                    data_frame, data_frame_centered=False,
                    data_frame_offset_x=0.0, data_frame_offset_y=0.0, data_frame_offset_theta=0.0,
                    data_floor=0.2, data_ceiling=2, dtype=np.uint8,
                    logger=None) -> None:
        #compute static grid transform to tag
        self.grid_width = grid_width
        self.grid_height = grid_width
        self.grid_scale = grid_scale
        self.occupancy_threshold = occupancy_threshold

        self.data_frame = data_frame

        grid_x, grid_y = grid_width*grid_scale, grid_height*grid_scale

        if data_frame_centered:
            self.data_frame_tform_grid = SE2Pose(grid_x/2.0, grid_y/2.0, 0)
        else:
            self.data_frame_tform_grid = SE2Pose(data_frame_offset_x*grid_scale, data_frame_offset_y*grid_scale, data_frame_offset_theta)

        self.data_floor = data_floor
        self.data_ceiling = data_ceiling
        
        #a belief with a mask        
        self._occupancy = ma.masked_all((grid_width, grid_height), dtype=np.float32)


    @property
    def occupied(self):
        return np.where(np.logical_and(self._occupancy.data > self.occupancy_threshold, np.logical_not(self._occupancy.mask)), 255, 0).astype(dtype=np.uint8)

    @property
    def free(self):
        return np.where(np.logical_and(self._occupancy.data < self.occupancy_threshold, np.logical_not(self._occupancy.mask)), 255, 0).astype(dtype=np.uint8)
    
    @property
    def unknown(self):
        return np.where(self._occupancy.mask, 255, 0).astype(dtype=np.uint8)

    @property
    def composite(self):
        pass

    def add_pcl(self, pcl):
        inertial_tform_cloud = get_a_tform_b(pcl.source.transforms_snapshot, self.data_frame,
                                     pcl.source.frame_name_sensor)
        grid_tform_inertial = SE3Pose.from_se2(self.data_frame_tform_grid)
        grid_tform_cloud = grid_tform_inertial*inertial_tform_cloud
        point_cloud_data = np.frombuffer(pcl.data, dtype=np.float32).reshape(int(pcl.num_points), 3)
        point_cloud_data = grid_tform_cloud.transform_cloud(point_cloud_data)
        
        start = self._real_to_grid(grid_tform_cloud.x, grid_tform_cloud.y)
        for point in point_cloud_data:
            if point.z < self.data_floor or point.z > self.data_ceiling:
                continue
            end = self._real_to_grid(point.x, point.y)

            free_trace = line(start[0],start[1],end[0],end[1])
            for pt in zip(free_trace):
                try:
                    if self._occupancy.mask[pt.x,pt.y]:
                        self._occupancy[pt.x,pt.y] = 0.0
                    else:
                        self._occupancy[pt.x,pt.y] *= 0.9
                except:
                    pass

            try:
                self._occupancy[end[0],end[1]] = 1
            except IndexError:
                pass
            

    @property
    def grid_proto(self):
        lg = local_grid_pb2.LocalGrid()
        # lg.acquisition_time
        # lg.transforms_snapshot
        lg.frame_name_local_grid_data = 'occupancy' #wrt....
        lg.local_grid_type_name = 'occupancy'
        lg.extent.cell_size = self.grid_scale
        lg.extent.num_cells_x=self.grid_width
        lg.extent.num_cells_y=self.grid_height
        lg.cell_format = local_grid_pb2.CELL_FORMAT_UINT8
        lg.encoding = local_grid_pb2.ENCODING_RAW
        lg.data = self._data
        return lg


    def _real_to_grid(self, x:float, y:float):
        #assumes x,y are already in the local coordinate frame
        return round(x/self.grid_scale), round(y/self.grid_scale)

    # def shift(self, x_offset, y_offset):
    #     np.roll(self._data, (x_offset, y_offset), axis=(0,1)) 

    # def regrid(self, x_offset, y_offset, theta):
    #     pass

    # def clear(self):
    #     pass


class OccupancyGridUpdater:
    def __init__(self, robot, grid, pcl_source_name, logger=None) -> None:
        self._grid = grid
        fields = pcl_source_name.split(':')
        pcl_service_name = fields[0]
        pcl_source_name = fields[1]

        

       

        self._grid_update_thread = threading.Thread(target=self.update_runner)

    def update_runner(self):
        while not self._stop_signal.is_set():
            for task in self._point_cloud_task.values():
                if not task.proto:
                    continue
                for pcl_response in task.proto:
                    self._grid.add_pcl(pcl_response.point_cloud)

    def stop(self):
        self._stop_signal.set()
        self._async_updates_thread.join()
        self._grid_update_thread.join()


class GlobalGridUpdater:
    pass

class GraphNavGridUpdater:
    pass

class BodyGridUpdater:
    pass


class OccupancyGridServicer(local_grid_service_pb2_grpc.LocalGridServiceServicer):
    """GRPC service to provide access to multiple different localgrid sources.

    Args:
        bosdyn_sdk_robot (Robot): The robot instance for the service to connect to.
        service_name (string): The name of the image service.
        image_sources(List[PointCloudSource]): The list of image sources (provided as a PointCloudSource).
        logger (logging.Logger): Logger for debug and warning messages.
        use_background_capture_thread (bool): If true, the image service will create a thread that continuously
            captures images so the image service can respond rapidly to the GetImage request. If false,
            the image service will call an image sources' blocking_capture_function during the GetImage request.
    """

    _all_grid_types = ['occupied-odom', 'occupied-visual', 'occupied-graphnav', 'occupied-body',
                       'free-space-odom', 'free-space-visual', 'free-space-graphnav', 'free-space-body',
                       'unknown-odom', 'unknown-visual', 'unknown-graphnav', 'unknown-body',
                       'UFO-odom', 'UFO-visual', 'UFO-graphnav', 'UFO-body']

    def __init__(self, bosdyn_sdk_robot, service_name, pcl_source_id, 
                    grid_width, grid_height, grid_scale, 
                    
                    anchor_april_tag, 
                    grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                    logger=None, use_background_capture_thread=True):
        super().__init__()
        if logger is None:
            # Set up the logger to remove duplicated messages and use a specific logging format.
            setup_logging(include_dedup_filter=True)
            self.logger = LOGGER
        else:
            self.logger = logger

        self.robot = bosdyn_sdk_robot
        fields = pcl_source_id.split(':')
        pcl_service = fields[0]
        pcl_source = fields[1]
        
        # Get a timesync endpoint from the robot instance such that the image timestamps can be
        # reported in the robot's time.
        self.robot.time_sync.wait_for_sync()
        
        # Service name this servicer is associated with in the robot directory.

        # Fault client to report service faults
        self.fault_client = self.robot.ensure_client(FaultClient.default_service_name)

        self._robot_state_client = self.robot.ensure_client(RobotStateClient.default_service_name)
        self._point_cloud_client = self.robot.ensure_client(pcl_service)
        
        self._point_cloud_task = AsyncPointCloud(self._point_cloud_client, [pcl_source])
        self._robot_state_task = AsyncRobotState(self._robot_state_client)

        _async_tasks = AsyncTasks([self._robot_state_task, self._point_cloud_task])

        self._stop_signal = threading.Event()
        self._async_updates_thread = threading.Thread(target=_update_thread, args=[_async_tasks, self._stop_signal])
        self._async_updates_thread.daemon = True
        self._async_updates_thread.start()





        self._grid = OccupancyGrid(grid_width, grid_height, grid_scale, grid_anchor_x, grid_anchor_y, grid_anchor_angle)
        self._grid_updater = OccupancyGridUpdater(bosdyn_sdk_robot, self._grid, pcl_source_name)

        



        #TODO: spawn thread to manage iterating over each source and adding data to
        # the occupancy grid as soon as it is ready
        # have thread lock so that the GetLocalGrids can reach in and make a copy

    def GetLocalGridTypes(self, request, context):
        response = local_grid_pb2.GetLocalGridTypesResponse()
        for grid in self.grids.values():
            response.local_grid_type.add().CopyFrom(grid.localgrid_type)
        populate_response_header(response, request)
        return response

    def GetLocalGrids(self, request, context):
        response = local_grid_pb2.GetLocalGridsResponse()
        for grid_request in request.local_grid_requests:
            grid_resp = response.local_grid_responses.add()
            src_name = grid_request.local_grid_type_name
            grid_resp.local_grid_type_name.CopyFrom(src_name)
            if src_name not in self.grids.keys():
                grid_resp.status = local_grid_pb2.LocalGridResponse.STATUS_NO_SUCH_GRID
                self.logger.warning("Occupancy grid '%s' is unknown.", src_name)
                continue
            
            grid_resp.local_grid.CopyFrom(self.grids[src_name].grid_proto)
        populate_response_header(response, request)




def make_occupancy_grid_service(bosdyn_sdk_robot, service_name, pcl_source_name,
                                grid_width, grid_height, grid_scale, anchor_april_tag, 
                                grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                                data_floor, data_ceiling, show_debug_information=False,
                                logger=None):
    


    return OccupancyGridServicer(bosdyn_sdk_robot, service_name, pcl_source_name, grid_width, grid_height, grid_scale, anchor_april_tag, 
                                grid_anchor_x, grid_anchor_y, grid_anchor_angle, logger)


def run_service(bosdyn_sdk_robot, port, service_name, pcl_source_name, 
                grid_width, grid_height, grid_scale,
                anchor_april_tag, grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                data_floor, data_ceiling,
                show_debug_information=False,
                logger=None):
    # Proto service specific function used to attach a servicer to a server.
    add_servicer_to_server_fn = local_grid_service_pb2_grpc.add_LocalGridServiceServicer_to_server

    # Instance of the servicer to be run.
    service_servicer = make_occupancy_grid_service(bosdyn_sdk_robot, service_name, pcl_source_name,
                                                    grid_width, grid_height, grid_scale,
                                                    anchor_april_tag, grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                                                    data_floor, data_ceiling,
                                                    show_debug_information=False,
                                                    logger=None)
    return GrpcServiceRunner(service_servicer, add_servicer_to_server_fn, port, logger=logger)


def add_occupancy_grid_arguments(parser):
    parser.add_argument('--pointcloud-source', required=False, type=str, default="velodyne-point-cloud:velodyne-point-cloud",
                        help="Name of a point-cloud as <service>:<source>.")
    parser.add_argument('--grid-width', type=int, help="number of grid cells in the x dimension", required=True)
    parser.add_argument('--grid-height', type=int, help="number of grid cells in the y dimension", required=True)
    parser.add_argument('--grid-scale', type=float, help="meters per grid cell edge", required=True)
    parser.add_argument('--anchor-tag', type=int, help="APRIL tag used to orient and anchor grid", required=True)
    parser.add_argument('--anchor-x', type=float, help="x-coordinate of anchor tag in the grid (meters)", required=False, default=0)
    parser.add_argument('--anchor-y', type=float, help="y-coordinate of anchor tag in the grid (meters)", required=False, default=0)
    parser.add_argument('--anchor-angle', type=float, help="orientation of anchor tag in the grid (degrees)", required=False, default=0)
    parser.add_argument('--data-floor', type=float, help="clip point cloud below this height (meters)", required=False, default=0.2)
    parser.add_argument('--data-ceiling', type=float, help="clip point cloud above this height (meters)", required=False, default=1.4)





def main(argv):
    # The last argument should be the IP address of the robot. The app will use the directory to find
    # the velodyne and start getting data from it.
    parser = argparse.ArgumentParser()
    bosdyn.client.util.add_base_arguments(parser)
    bosdyn.client.util.add_payload_credentials_arguments(parser)
    bosdyn.client.util.add_service_endpoint_arguments(parser)
    add_occupancy_grid_arguments(parser)
    options = parser.parse_args(argv)

    sdk = bosdyn.client.create_standard_sdk('OccupancyGrid')
    robot = sdk.create_robot(options.hostname)
    bosdyn.client.util.authenticate(robot)
    robot.sync_with_directory()

    sources = options.pointcloud_source
    if not sources:
        LOGGER.error('At least one pointcloud-source must be specified')
        return

    service_runner = run_service(robot, options.port, DIRECTORY_NAME, sources, options.grid_width,
                                 options.grid_height, options.grid_scale, options.anchor_tag,
                                 options.anchor_x, options.anchor_y, options.anchor_angle, 
                                 options.data_floor, options.data_ceilting, 
                                 show_debug_information=options.show_debug_info, logger=LOGGER)

    # Use a keep alive to register the service with the robot directory.
    dir_reg_client = robot.ensure_client(DirectoryRegistrationClient.default_service_name)
    keep_alive = DirectoryRegistrationKeepAlive(dir_reg_client, logger=LOGGER)
    keep_alive.start(DIRECTORY_NAME, SERVICE_TYPE, AUTHORITY, options.host_ip, service_runner.port)

    # Attach the keep alive to the service runner and run until a SIGINT is received.
    with keep_alive:
        service_runner.run_until_interrupt()



if __name__=='__main__':
    main()