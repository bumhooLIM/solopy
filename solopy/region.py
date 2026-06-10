class SOLORegion:
    
    def __init__(self, image_shape, base_tile_size=500):
        """
        Manage image tiling and coordinate mapping.
        
        Parameters:
        - image_shape: tuple, (height, width) of the CCD data.
        - base_tile_size: int, default size of the spatial regions (e.g., 500).
        """
        self.image_h, self.image_w = image_shape
        self.base_tile_size = base_tile_size
        
        # Floor division ensures the final tile absorbs the remainder
        self.num_tiles_x = self.image_w // self.base_tile_size
        self.num_tiles_y = self.image_h // self.base_tile_size

    def find_region(self, x, y):
        """
        Identify the region indices (i, j) for a specific (x, y) coordinate.
        """
        i = int(x // self.base_tile_size)
        j = int(y // self.base_tile_size)
        
        # Cap at the maximum index to correctly map coordinates falling into the extended edge regions
        i = min(i, self.num_tiles_x - 1)
        j = min(j, self.num_tiles_y - 1)
        
        return i, j

    def get_tile_info(self, i, j):
        """
        Get the exact bounding box and center for region (i, j), 
        dynamically extending the edges to solve the ceiling problem.
        """
        x_start = i * self.base_tile_size
        y_start = j * self.base_tile_size
        
        # Extend the final tile to the absolute edge of the sensor
        x_end = self.image_w if i == self.num_tiles_x - 1 else (i + 1) * self.base_tile_size
        y_end = self.image_h if j == self.num_tiles_y - 1 else (j + 1) * self.base_tile_size
        
        size_x = x_end - x_start
        size_y = y_end - y_start
        
        x_center = x_start + (size_x / 2.0)
        y_center = y_start + (size_y / 2.0)
        
        return {
            'x_start': x_start, 'x_end': x_end,
            'y_start': y_start, 'y_end': y_end,
            'size_x': size_x,   'size_y': size_y,
            'x_center': x_center, 'y_center': y_center
        }