
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

from bosdyn.client.util import setup_logging

from bosdyn.client.robot_state import RobotStateClient
from bosdyn.client.async_tasks import AsyncPeriodicQuery, AsyncTasks

from bosdyn.client.math_helpers import Quat, SE3Pose
from bosdyn.client.frame_helpers import get_odom_tform_body

from bosdyn.api import local_grid_pb2, local_grid_service_pb2_grpc 



DIRECTORY_NAME = 'occupancy-grid'
AUTHORITY = 'occupancy-grid'
SERVICE_TYPE = 'bosdyn.api.LocalGridService'

LOGGER = logging.getLogger(__name__)




def _update_thread(async_task):
    while True:
        async_task.update()
        time.sleep(0.01)


class AsyncPointCloud(AsyncPeriodicQuery):
    """Grab robot state."""

    def __init__(self, point_cloud_client, pcl_sources):
        super(AsyncPointCloud, self).__init__("point_clouds", point_cloud_client, LOGGER,
                                              period_sec=0.2)

        self._sources = pcl_sources

    def _start_query(self):
        return self._client.get_point_cloud_from_sources_async(self._sources)

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
            data = np.fromstring(pcl.data, dtype=np.float32)
            t = pcl.source.acquisition_time



   

class OccupancyGrid:
    def __init__(self, grid_width, grid_height, grid_scale, anchor_april_tag, 
                    grid_anchor_x, grid_anchor_y, grid_anchor_angle, 
                    data_floor, data_ceiling, 
                    logger=None) -> None:
        #compute static grid transform to tag
        pass

    def add_pcl(self, pcl):
        # transform to tag grid frame via tag frame
        pass

    def get_current_grid(self):
        pass

    def clear(self):
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
        for grid in self.occupancy_grids.values():
            response.local_grid_type.add().CopyFrom(grid.image_source_proto)
        populate_response_header(response, request)
        return response

    def GetLocalGrids(self, request, context):
        pass



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


    
    _task_list = [_point_cloud_task]
    _async_tasks = AsyncTasks(_task_list)
    print('Connected.')

    update_thread = threading.Thread(target=_update_thread, args=[_async_tasks])
    update_thread.daemon = True
    update_thread.start()


if __name__=='__main__':
    main()