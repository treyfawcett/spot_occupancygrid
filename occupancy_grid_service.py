
import argparse
import collections
from email.policy import default
import logging
import sys
import time
import threading
from typing import List
import numpy as np

import bosdyn
import bosdyn.client
import bosdyn.client.util

from bosdyn.client.directory_registration import (DirectoryRegistrationClient,
                                                  DirectoryRegistrationKeepAlive)

from bosdyn.client.fault import FaultClient
from bosdyn.client.server_util import GrpcServiceRunner, populate_response_header
from bosdyn.client.world_object import WorldObjectClient, make_add_world_object_req

from bosdyn.client.util import setup_logging

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks

from bosdyn.api import local_grid_pb2, local_grid_service_pb2_grpc 

from bosdyn.client.frame_helpers import *
from bosdyn.client.math_helpers import *
from bosdyn.api import geometry_pb2 as geom
from bosdyn.api import world_object_pb2

from skimage.draw import line


DIRECTORY_NAME = 'occupancy-grid'
AUTHORITY = 'occupancy-grid'
SERVICE_TYPE = 'bosdyn.api.LocalGridService'

GRID_FRAME_NAME='occupancy-grid'

LOGGER = logging.getLogger(__name__)


def get_point_cloud_data_in_grid_frame(pcl):
    """
    Create a 3 x N numpy array of points in the grid frame. 
    :return: a 3 x N numpy array in the seed frame.
    """
    
    cloud = pcl
    #TODO: vision, odom, or ground plane?
    vision_tform_cloud = get_a_tform_b(cloud.source.transforms_snapshot, VISION_FRAME_NAME,
                                     cloud.source.frame_name_sensor)
    point_cloud_data = np.frombuffer(cloud.data, dtype=np.float32).reshape(int(cloud.num_points), 3)
    return vision_tform_cloud.transform_cloud(point_cloud_data)




def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.01)


class AsyncPointCloud(AsyncPeriodicQuery):
    def __init__(self, point_cloud_client, pcl_sources):
        super(AsyncPointCloud, self).__init__("point_clouds", point_cloud_client, LOGGER,
                                              period_sec=0.2)

        self._sources = pcl_sources

    def _start_query(self):
        return self._client.get_point_cloud_from_sources_async(self._sources)

class AsyncFiducialObjects(AsyncPeriodicQuery):
    def __init__(self, query_name, client, period_sec):
        super().__init__(query_name, client, LOGGER, period_sec)

    def _start_query(self):
        return self._client.list_world_objects_async(object_type=[world_object_pb2.WORLD_OBJECT_APRILTAG])

class PointCloudProxy:
    def __init__(self, robot, pcl_service: str, pcl_sources: List[str], use_async: bool = True) -> None:
        
        self._point_cloud_client = robot.ensure_client(pcl_service)
        
        
        self._point_cloud_task = AsyncPointCloud(self._point_cloud_client, pcl_sources)


        self.stop_capturing_event = threading.Event()

        # Track the last pcl and timestamp for this pcl source.
        self.last_captured_pcl = None
        self.last_captured_time = None

        # Lock for the thread.
        self._thread_lock = threading.Lock()
        self._thread = None


        update_thread = threading.Thread(target=_update_thread, args=[self._point_cloud_task])
        update_thread.daemon = True
        update_thread.start()
        
    def set_logger(self, logger):
        """Override the existing logger for the AsyncPointCloudProxy class."""
        if logger is not None:
            self.logger = logger

    def get_latest(self):
        pcl = self._point_cloud_task.proto[0].point_cloud
        if pcl:
            data = get_point_cloud_data_in_grid_frame(pcl)
            t = pcl.source.acquisition_time


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
    def __init__(self, grid_width, grid_height, grid_scale, anchor_april_tag, 
                    grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                    data_floor, data_ceiling, max_ray_trace,
                    logger=None) -> None:
        #compute static grid transform to tag
        self.grid_width = grid_width
        self.grid_height = grid_width
        self.grid_scale = grid_scale
        
        #observation grid (unknown, ray traced free, occupied))
        self._data = np.zeros((grid_width, grid_height), dtype=np.uint8)
        #occupancy belief(floating point)


    def add_pcl(self, anchor, pcl):
        # transform to grid frame via anchor tag frame
        start = self._real_to_grid(anchor.x, anchor.y)
        for point in pcl:
            end = self._real_to_grid(point.x, point.y)


            pixels = line(*start, *end) #([x coords], [y coords])
            
            np.subtract.at(self._data, (pixels[0][:-1],pixels[1][:-1]), 10)

            self._data[pixels[0][-1],pixels[1][-1]] += 20
        

    def get_current_grid(self):
        pass

    def clear(self):
        pass

    def _real_to_grid(self, x, y):
        return x/self.grid_scale, y/self.grid_scale

    def _is_in_grid(self, x, y):
        return x >= 0 and x < self.grid_width and y >=0 and y < self.grid_height

    def _clip_to_grid(self, start, end):
        #could enter and leave grid....
        return start, end


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

    def __init__(self, bosdyn_sdk_robot, service_name, pcl_sources, 
                    grid_width, grid_height, grid_scale, anchor_april_tag, 
                    grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                    logger=None, use_background_capture_thread=True):
        super().__init__()
        if logger is None:
            # Set up the logger to remove duplicated messages and use a specific logging format.
            setup_logging(include_dedup_filter=True)
            self.logger = LOGGER
        else:
            self.logger = logger

        self.bosdyn_sdk_robot = bosdyn_sdk_robot

        # Service name this servicer is associated with in the robot directory.
        self.service_name = service_name

        # Fault client to report service faults
        self.fault_client = self.bosdyn_sdk_robot.ensure_client(FaultClient.default_service_name)

        # Get a timesync endpoint from the robot instance such that the image timestamps can be
        # reported in the robot's time.
        self.bosdyn_sdk_robot.time_sync.wait_for_sync()

        # A list of all the image sources available by this service. List[VisualImageSource]
        self.pcl_sources_mapped = dict()  # Key: source name (string), Value: VisualImageSource
        for source in pcl_sources:
            # Set the logger for each visual image source to be the logger of the camera service class.
            source.set_logger(self.logger)
            # Set up the fault client so service faults can be created.
            source.initialize_faults(self.fault_client, self.service_name)
            # Potentially start the capture threads in the background.
            if use_background_capture_thread:
                source.create_capture_thread()
            # Save the visual image source class associated with the image source name.
            self.pcl_sources_mapped[source.image_source_name] = source


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


def make_occupancy_grid_service(bosdyn_sdk_robot, service_name, pcl_source_names,
                                grid_width, grid_height, grid_scale, anchor_april_tag, 
                                grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                                data_floor, data_ceiling, show_debug_information=False,
                                logger=None):
    pcl_source_groupings = {}
    for source_name in pcl_source_names:
        source_fields = source_name.split(':')
        assert len(source_fields) == 2, "pcl source names must be formatted as <service>:<source>"
        if source_fields[0] not in pcl_source_groupings:
            pcl_source_groupings[source_fields[0]] = []
        pcl_source_groupings[source_fields[0]].append(source_fields[1])
    pcl_source_proxies = []
    for service in pcl_source_groupings.keys():
        pcl_proxy = PointCloudProxy(bosdyn_sdk_robot, service, pcl_source_groupings[service])
        pcl_source_proxies.append(pcl_proxy)
    return OccupancyGridServicer(bosdyn_sdk_robot, service_name, pcl_source_proxies, grid_width, grid_height, grid_scale, anchor_april_tag, 
                                grid_anchor_x, grid_anchor_y, grid_anchor_angle, logger)


def run_service(bosdyn_sdk_robot, port, service_name, pcl_source_names, 
                grid_width, grid_height, grid_scale,
                anchor_april_tag, grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                data_floor, data_ceiling,
                show_debug_information=False,
                logger=None):
    # Proto service specific function used to attach a servicer to a server.
    add_servicer_to_server_fn = local_grid_service_pb2_grpc.add_LocalGridServiceServicer_to_server

    # Instance of the servicer to be run.
    service_servicer = make_occupancy_grid_service(bosdyn_sdk_robot, service_name, pcl_source_names,
                                                    grid_width, grid_height, grid_scale,
                                                    anchor_april_tag, grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                                                    data_floor, data_ceiling,
                                                    show_debug_information=False,
                                                    logger=None)
    return GrpcServiceRunner(service_servicer, add_servicer_to_server_fn, port, logger=logger)


def add_occupancy_grid_arguments(parser):
    parser.add_argument('--pointcloud-source',
                        help="Name of a point-cloud as <service>:<source>.", action="append", required=True, default=[])
    parser.add_argument('--grid-width', type=int, help="number of grid cells in the x dimension", required=True)
    parser.add_argument('--grid-height', type=int, help="number of grid cells in the y dimension", required=True)
    parser.add_argument('--grid-scale', type=float, help="meters per grid cell edge", required=True)
    parser.add_argument('--anchor-tag', type=int, help="APRIL tag used to orient and anchor grid", required=True)
    parser.add_argument('--anchor-x', type=float, help="x-coordinate of anchor tag in the grid (meters)", required=False, default=0)
    parser.add_argument('--anchor-y', type=float, help="y-coordinate of anchor tag in the grid (meters)", required=False, default=0)
    parser.add_argument('--anchor-angle', type=float, help="orientation of anchor tag in the grid (degrees)", required=False, default=0)
    parser.add_argument('--data-floor', type=float help="clip point cloud below this height (meters)", required=False, default=0.2)
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