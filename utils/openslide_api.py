import requests
from PIL import Image
from io import BytesIO
import math


class OpenSlideError(Exception):
    """Custom exception to mimic OpenSlideError."""
    pass


class OpenSlideAPI:
    """
    OpenSlideAPI class to interact with whole-slide images via remote APIs.

    Attributes and methods mirror those of the OpenSlide class.
    """

    BASE_URL = "https://mdi.hkust-gz.edu.cn/wsi/metaservice/api"

    @classmethod
    def parse_region(cls, region_url):
        # region_url = f"{cls.BASE_URL}/region/openslide/{self.filename}/{x_level}/{y_level}/{w_level}/{h_level}/{level}"
        filename, x, y, w, h, level = region_url.split('/')[-6:]
        x, y, w, h, level = int(x), int(y), int(w), int(h), int(level)
        return filename, x, y, w, h, level

    def __init__(self, filename):
        """
        Initialize the OpenSlideAPI object by fetching slide metadata.

        Args:
            filename (str): The filename of the WSI.

        Raises:
            OpenSlideError: If unable to retrieve or parse slide information.
        """
        self.filename = filename
        self._closed = False
        self._cache = {}

        # Fetch slice information
        slice_info_url = f"{self.BASE_URL}/sliceInfo/openslide/{self.filename}"
        try:
            response = requests.get(slice_info_url)
            response.raise_for_status()
            self._info = response.json()
        except requests.RequestException as e:
            raise OpenSlideError(f"Failed to fetch slice info: {e}")
        except ValueError:
            raise OpenSlideError("Invalid JSON response for slice info.")

        # Parse slice information
        try:
            self.base_magnification = int(round(float(self._info.get("aperio.AppMag", 40.0))))
            self.properties = {k: v for k, v in self._info.items() if not k.startswith("openslide.level")}
            self.level_count = int(self._info.get("openslide.level-count", 1))
            self.level_dimensions = []
            self.level_downsamples = []
            for level in range(self.level_count):
                width = int(self._info.get(f"openslide.level[{level}].width"))
                height = int(self._info.get(f"openslide.level[{level}].height"))
                downsample = float(self._info.get(f"openslide.level[{level}].downsample"))
                self.level_dimensions.append((width, height))
                self.level_downsamples.append(downsample)
            self.dimensions = self.level_dimensions[0]
            self.color_profile = None  # Not provided by the API
            self.associated_images = {}  # Not provided by the API
        except (KeyError, ValueError, TypeError) as e:
            raise OpenSlideError(f"Error parsing slice info: {e}")

    def __repr__(self):
        return f"<OpenSlideAPI filename={self.filename}>"

    def close(self):
        """Close the OpenSlideAPI object and clear resources."""
        self._closed = True
        self._cache.clear()

    def __del__(self):
        self.close()

    def __enter__(self):
        """Enter the runtime context related to this object."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the runtime context and close the object."""
        self.close()

    def get_best_level_for_downsample(self, downsample):
        """
        Return the best level for displaying the given downsample.

        Args:
            downsample (float): The desired downsample factor.

        Returns:
            int: The level number that best matches the downsample.
        """
        if self._closed:
            raise OpenSlideError("Attempted to operate on a closed OpenSlideAPI object.")

        best_level = 0
        min_diff = float('inf')
        for level, ds in enumerate(self.level_downsamples):
            diff = abs(ds - downsample)
            if diff < min_diff:
                min_diff = diff
                best_level = level
        return best_level

    def read_region(self, location, level, size):
        """
        Return a PIL.Image containing the contents of the region.

        Args:
            location (tuple): (x, y) tuple in level 0 reference frame.
            level (int): The level number.
            size (tuple): (width, height) of the region.

        Returns:
            PIL.Image: The image of the specified region.

        Raises:
            OpenSlideError: If the API request fails or parameters are invalid.
        """
        if self._closed:
            raise OpenSlideError("Attempted to operate on a closed OpenSlideAPI object.")

        if level < 0 or level >= self.level_count:
            raise OpenSlideError(f"Invalid level: {level}")

        downsample = self.level_downsamples[level]
        x0, y0 = location
        x_level = int(math.floor(x0 / downsample))
        y_level = int(math.floor(y0 / downsample))
        w_level = int(size[0])
        h_level = int(size[1])

        region_url = f"{self.BASE_URL}/region/openslide/{self.filename}/{x_level}/{y_level}/{w_level}/{h_level}/{level}"

        cache_key = ('region', level, x_level, y_level, w_level, h_level)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = requests.get(region_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGBA")
            self._cache[cache_key] = image
            return image
        except requests.RequestException as e:
            raise OpenSlideError(f"Failed to fetch region: {e}")
        except IOError:
            raise OpenSlideError("Failed to parse image data from region response.")

    def get_thumbnail(self, size):
        """
        Return a PIL.Image containing an RGB thumbnail of the image.

        Args:
            size (tuple): The maximum size (width, height) of the thumbnail.

        Returns:
            PIL.Image: The thumbnail image.

        Raises:
            OpenSlideError: If the API request fails.
        """
        if self._closed:
            raise OpenSlideError("Attempted to operate on a closed OpenSlideAPI object.")

        thumbnail_url = f"{self.BASE_URL}/thumbnail/openslide/{self.filename}"

        cache_key = ('thumbnail', size)
        if cache_key in self._cache:
            return self._cache[cache_key]

        try:
            response = requests.get(thumbnail_url)
            response.raise_for_status()
            image = Image.open(BytesIO(response.content)).convert("RGBA")
            if size:
                image.thumbnail(size, Image.ANTIALIAS)
            self._cache[cache_key] = image
            return image
        except requests.RequestException as e:
            raise OpenSlideError(f"Failed to fetch thumbnail: {e}")
        except IOError:
            raise OpenSlideError("Failed to parse image data from thumbnail response.")

    def set_cache(self, cache):
        """
        Use the specified cache to store recently decoded slide tiles.

        Args:
            cache (dict-like): A cache object supporting get and set operations.

        Note:
            This implementation uses a simple dictionary for caching.
            For more advanced caching, implement a cache with eviction policies.
        """
        if self._closed:
            raise OpenSlideError("Attempted to operate on a closed OpenSlideAPI object.")
        self._cache = cache

    @classmethod
    def detect_format(cls, filename):
        """
        Return a string describing the format vendor of the specified file.

        Args:
            filename (str): The filename of the WSI.

        Returns:
            str or None: The format vendor or None if unrecognized.

        Note:
            This implementation assumes the format can be inferred from the filename extension.
        """
        supported_formats = {
            '.svs': 'Aperio',
            '.tiff': 'TIFF',
            '.mrxs': 'Mirax',
            # Add more formats as needed
        }
        for ext, vendor in supported_formats.items():
            if filename.lower().endswith(ext):
                return vendor
        return None

    @property
    def level_count_property(self):
        """The number of levels in the image."""
        return self.level_count

    @property
    def level_dimensions_property(self):
        """A list of (width, height) tuples, one for each level of the image."""
        return self.level_dimensions

    @property
    def level_downsamples_property(self):
        """A list of downsampling factors for each level of the image."""
        return self.level_downsamples

    @property
    def dimensions_property(self):
        """A (width, height) tuple for level 0 of the image."""
        return self.dimensions

    @property
    def properties_property(self):
        """Metadata about the image."""
        return self.properties

    @property
    def associated_images_property(self):
        """Images associated with this whole-slide image."""
        return self.associated_images

    @property
    def color_profile_property(self):
        """Color profile for the whole-slide image, or None if unavailable."""
        return self.color_profile

# Example Usage
if __name__ == "__main__":
    try:
        with OpenSlideAPI("249823-24.tiff") as slide:
            print(f"Dimensions: {slide.dimensions}")
            print(f"Level Count: {slide.level_count}")
            print(f"Level Downsamples: {slide.level_downsamples}")
            print(f"Level Dimensions: {slide.level_dimensions}")

            # Get best level for a downsample of 4
            best_level = slide.get_best_level_for_downsample(4)
            print(f"Best level for downsample 4: {best_level}")

            # Get a thumbnail
            thumbnail = slide.get_thumbnail((512, 512))
            thumbnail.show()

            # Read a specific region
            region = slide.read_region(location=(19582, 52490), level=1, size=(768, 768))
            region.show()

    except OpenSlideError as e:
        print(f"Error: {e}")
