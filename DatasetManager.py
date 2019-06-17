import os
from osUtils import *

def get_bottleneck_path_new(image_path, label_name, bottleneck_dir):
	image_path = os.path.join(bottleneck_dir, label_name, image_path)
	return image_path + '.txt'
    
def create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model):
  """Create a single bottleneck file."""
  print ('Creating bottleneck at ' + bottleneck_path)
  try:
    bottleneck_values = inceptionV3Model.run_bottleneck_on_image(
        sess, image_data)
  except Exception as e:
    print (e)
    raise RuntimeError('Error during processing file %s' % bottleneck_path)

  bottleneck_string = ','.join(str(x) for x in bottleneck_values)
  with open(bottleneck_path, 'w') as bottleneck_file:
    bottleneck_file.write(bottleneck_string)


"""
Retrieves or calculates bottleneck values for an image.
Args:
    sess: The current active TensorFlow Session.
    image_data:image data.
    image_index: location of image in the list
    image_category: train/test
    bottleneck_dir: Folder string holding cached files of bottleneck values.
Returns:
    Numpy array of values produced bget_bottleneck_pathy the bottleneck layer for the image.
"""
def get_or_create_bottleneck_new(sess, image_index, image_data, image_category, bottleneck_dir, inceptionV3Model):
  sub_dir_path = os.path.join(bottleneck_dir, image_category)
  ensure_dir_exists(sub_dir_path)
  bottleneck_path = get_bottleneck_path_new(image_index, image_category, bottleneck_dir)

  if not os.path.exists(bottleneck_path):
    create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model)
  with open(bottleneck_path, 'r') as bottleneck_file:
    bottleneck_string = bottleneck_file.read()
  did_hit_error = False
  try:
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  except ValueError:
    print('Invalid float found, recreating bottleneck')
    did_hit_error = True
  if did_hit_error:
    create_bottleneck_file_new(bottleneck_path, image_data, sess, inceptionV3Model)
    with open(bottleneck_path, 'r') as bottleneck_file:
      bottleneck_string = bottleneck_file.read()
    # Allow exceptions to propagate here, since they shouldn't happen after a
    # fresh creation
    bottleneck_values = [float(x) for x in bottleneck_string.split(',')]
  return bottleneck_values

"""
Retrieves bottleneck values for images.
Args:
    sess: Current TensorFlow Session.
    image_data: array of image data.
    batch_offset: offset of image batch in the list
    image_category: train/test
    bottleneck_dir: Folder string holding cached files of bottleneck values.
    Returns:
    List of bottleneck arrays.
"""
def get_random_cached_bottlenecks(sess, batch_offset, image_data,image_category,bottleneck_dir, inceptionV3Model):
  bottlenecks = []
  for i in range(image_data.shape[0]):
      #print str(i)+"/"+str(image_data.shape[0])
      bottleneck = get_or_create_bottleneck_new(sess,str(batch_offset+i),image_data[i],image_category,bottleneck_dir, inceptionV3Model)
      bottlenecks.append(bottleneck)
  return bottlenecks