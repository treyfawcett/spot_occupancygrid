
import logging
import math
import threading
import time
from typing import Optional
import numpy as np
import cv2 as cv

from skimage.draw import line


from bosdyn.api import local_grid_pb2

LOGGER = logging.getLogger(__name__)


class GridAnalyzer:
    """Identify structural features in occupancy grids.
    
    Try to identify aisles, junctions, and freespace.
    """

    def __init__(self, robot, source_grid: str, update_period: float=1.0, logger: Optional[logging.Logger]=None) -> None:
        self.occupancy_grid_client = robot.ensure_client('occupancy-grid')
        self.source_grid = source_grid

        self.update_period = update_period

        self._stop_signal = threading.Event()
        self.thread = threading.Thread(target=self._thread_loop)
        
        self.logger = logger or LOGGER



    def _thread_loop(self):
        last_update = 0.0
        while not self._stop_signal.is_set():
            if time.time() - last_update > self.update_period:
                grid_responses = self.occupancy_grid_client.get_local_grids([self.source_grid])
                for grid_response in grid_responses:
                    self.process_response(grid_response)
            else:
                time.sleep(0.1)

    def process_response(self, response: local_grid_pb2.LocalGridResponse):
        if response.local_grid_type_name != self.source_grid:
            self.logger.warning(f'received unexpected grid type from occupancy grid client ({response.local_grid_type_name})')
        elif response.status != local_grid_pb2.LocalGridResponse.STATUS_OK:
            self.logger.warning(f'recived error in grid response: status {response.status}')
        else:
            self.process_grid(response.local_grid)
    
    def process_grid(self, grid: local_grid_pb2.LocalGrid):
        if grid.encoding != local_grid_pb2.LocalGrid.ENCODING_RAW:
            #TODO: enable RLE (on both sides) to reduce bandwidth and latency
            self.logger.warning(f'GridAnalyzer requires RAW encoding of grid')
            return
        
        dtype=None
        if grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT32:
            dtype=np.float32
        elif grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_FLOAT64:
            dtype=np.float64
        elif grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT8:
            dtype=np.int8
        elif grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT8:
            dtype=np.uint8
        elif grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_INT16:
            dtype=np.int16
        elif grid.cell_format == local_grid_pb2.LocalGrid.CELL_FORMAT_UINT16:
            dtype=np.uint16
        if dtype is None:
            self.logger.warning(f'grid format not recognized')
            return

        occupancy = np.frombuffer(grid.data, dtype=dtype).reshape((grid.extent.num_cells_x, grid.extent.num_cells_y))
        
        blur_img = cv.GaussianBlur(occupancy, (3,3), 0)
        edge_img = cv.Canny(blur_img, threshold1=100, threshold2=200)

        hough_lines = cv.HoughLinesP(edge_img, 
                                    rho=1, 
                                    theta=np.pi/180.0, 
                                    threshold=15, 
                                    minLineLength=5.0/grid.extent.cell_size, 
                                    maxLineGap=1.0/grid.extent.cell_size)

        zone_raster = np.zeros(occupancy.shape,dtype=np.uint8)

        # lsd = cv.createLineSegmentDetector()
        # lsd_lines = lsd.detect(edge_img)[0]

        offset=2
        aisle_width_pixels = 3.0/grid.extent.cell_size
        for line in hough_lines:
            x1,y1,x2,y2 = line[0]
            #for each line, compute which side is occupied
            if x1 > x2:
                #reverse ordering 
                xt, yt = x1, y1
                x1, y1 = x2, y2
                x2, y2 = xt, yt

            dx = x2 - x1
            dy = y2 - y1
            horizontalish = abs(dx) > abs(dy)
            #            up/down shift                   if horizontalish else left/right shift
            l1_indices = line(x1,y1+offset,x2,y2+offset) if horizontalish else line(x1+offset,y1,x2+offset,y2)
            l2_indices = line(x1,y1-offset,x2,y2-offset) if horizontalish else line(x1-offset,y1,x2-offset,y2)
            l1_sum = np.sum(occupancy[l1_indices])
            l2_sum = np.sum(occupancy[l2_indices])
            if l1_sum == l2_sum:
                self.logger.warning('ambiguous gradient')
                continue

            zone_poly = [(x1,y1), (x2,y2)]
            # if l1_sum > l2_sum != horizontalish  -> occupied space is right of line or below line
            offset_x, offset_y = self._corner_offset(l1_sum > l2_sum != horizontalish, dx, dy, aisle_width_pixels)
            
            zone_poly.append((x2+offset_x, y2+offset_y))
            zone_poly.append((x1+offset_x, y1+offset_y))

            cv.fillConvexPoly(zone_raster, zone_poly, 0x1)

        #hopefully most pixels are covered once, twice, or not at all



    def _corner_offset(self, CCW: bool, ref_dx, ref_dy, offset):
        dx = ref_dy if CCW else -ref_dy
        dy = -ref_dx if CCW else ref_dx
        l = math.sqrt(dx*dx+dy*dy)
        offset_x = round(dx*offset/l)
        offset_y = round(dy*offset/l)
        return offset_x,offset_y

