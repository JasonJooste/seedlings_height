import gdal
import os.path
from pathlib import Path
import skimage as skim
import skimage.transform
import skimage.io
EPS = 1e-1
def resize_files(fn1, fn2):
    """
    Take two geotiff images and resize the second so they are on the same scale
    :return:
    """
    data1 = gdal.Open(fn1)
    data2 = gdal.Open(fn2)
    if not data1 or not data2:
        return # This could be better. Maybe an exception should be thrown
    gt1 = data1.GetGeoTransform()
    gt2 = data2.GetGeoTransform()
    # Check that the top left coordinates are the same
    # Index 0 is the x coord and index 3 is the y coord of the top left hand pixel
    # TODO: This needs to be better handled. It is also a function of the resolution of the pixels of both images.
    # TODO: Could also check the number of channels
    if gt1[0] - gt2[0] > EPS or gt1[3] - gt2[3] > EPS:
        return # This could be extended to shift the second image as well
    # Entries 1 and 5 show the distance moved per pixel in x and y
    scale1x = gt1[1]
    scale1y = gt1[5]
    scale2x = gt2[1]
    scale2y = gt2[5]
    # Now we need the size of the second image to resize it
    x_size = data2.RasterXSize
    y_size = data2.RasterYSize
    # Now we calculate the new scale for the second image
    ratio_x = scale2x / scale1x
    ratio_y = scale2y / scale1y
    # Now we can rescale the image
    new_x = round(x_size * ratio_x)
    new_y = round(y_size * ratio_y)
    # Finally we need to actually write the image
    im_arr = data2.ReadAsArray()
    new_im_arr = skim.transform.resize(im_arr, (new_y, new_x), mode='constant')
    # This will not include the geo info in the file. But it isn't necessary at the moment
    split = os.path.splitext(fn2)
    new_fn = split[0] + "_large" + split[1]
    skim.io.imsave(new_fn, new_im_arr)

def split_images(filename, slice_w=256, slice_h=256):
    """
    Split a large image into a series of small tiles and save them
    :return:
    """
    image = skim.io.imread(filename)
    im_h = image.shape[0]
    im_w = image.shape[1]
    pos_x = 0
    pos_y = 0
    splitname = os.path.splitext(filename)
    while pos_y < im_h:
        while pos_x < im_w:
            im_sec = image[pos_y:pos_y+slice_h, pos_x:pos_x+slice_w]
            filename = splitname[0] + "_cut-{}-{}".format(pos_x, pos_y) + splitname[1]
            skim.io.imsave(filename, im_sec)
            pos_x += slice_w
        pos_y += slice_h
        pos_x = 0
gdal.UseExceptions()
image_path = r'C:/Users/joost/Dokuments/Uni/practical/raw/site_464_201710_030m_ortho_als11_3channels.tif'
height_path = r'C:/Users/joost/Dokuments/Uni/practical/raw/site_464__201710_CHM10cm.tif'
resize_files(image_path, height_path)
# Remove the 33 pixel top-left pad on both images (first 33 are white and there is another 2 pixels off in the xml files)
image = skim.io.imread(image_path)
im_splitname = os.path.splitext(image_path)
im_new_name = im_splitname[0] + "_buffer_removed" + im_splitname[1]
shift = 35
skim.io.imsave(im_new_name, image[shift+1:, shift+1:, :])
#Second image
h_splitname = os.path.splitext(height_path)
height_path = h_splitname[0] + "_large" + h_splitname[1]
height = skim.io.imread(height_path)
h_splitname = os.path.splitext(height_path)
h_new_name = h_splitname[0] + "_buffer_removed" + h_splitname[1]
skim.io.imsave(h_new_name, height[shift+1:, shift+1:])
# Split the images
split_images(im_new_name)
split_images(h_new_name)