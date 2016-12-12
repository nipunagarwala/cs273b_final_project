import tensorflow as tf
import numpy as np
import json

# Train and evaliation epoch definition
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

# Pixels to crop in each dimensions
CROP = 4

# Phenotype data is vector of size (29,)


def gaussian_noise(image, threshold=1, mean=0.0, stddev=0.01, seed=None):

	delta = tf.truncated_normal(image.get_shape(), mean=mean, stddev=stddev, dtype=tf.float32, seed=seed, name=None)
	with tf.name_scope(None, 'gaussian_noise', [image, delta]) as name:
		image = tf.convert_to_tensor(image, name='image')
	    # Remember original dtype to so we can convert back if needed
		orig_dtype = image.dtype
		flt_image = tf.image.convert_image_dtype(image, tf.float32)

		minimum = tf.reduce_min(flt_image)
		min_thresh = tf.add(minimum, tf.constant(threshold, dtype=tf.float32))
		poi = tf.greater(flt_image, min_thresh)
		noise = tf.add(flt_image, delta, name=name)
		noise_minimum = tf.reduce_min(noise)

		adjusted = tf.select(poi, noise, tf.fill(image.get_shape(), noise_minimum))

		return tf.image.convert_image_dtype(adjusted, orig_dtype, saturate=True)


def random_flip_fwd_back(image, seed=None):
	image = tf.convert_to_tensor(image, name='image')
	tf.image._Check3DImage(image, require_static=False)
	uniform_random = tf.random_uniform([], 0, 1.0, seed=seed)
	mirror = tf.less(tf.pack([uniform_random, 1.0, 1.0]), 0.5)
	return tf.reverse(image, mirror)


def crop_3d(image, offset_height, offset_width, offset_depth, target_height, target_width, target_depth):
	image = tf.convert_to_tensor(image, name='image')

	assert_ops = []
	assert_ops += tf.image._Check3DImage(image, require_static=False)

	height, width, depth = tf.image._ImageDimensions(image)

	assert_ops += tf.image._assert(offset_width >= 0, ValueError,
	                    'offset_width must be >= 0.')
	assert_ops += tf.image._assert(offset_height >= 0, ValueError,
	                    'offset_height must be >= 0.')
	assert_ops += tf.image._assert(offset_depth >= 0, ValueError,
	                    'offset_depth must be >= 0.')
	assert_ops += tf.image._assert(target_width > 0, ValueError,
	                    'target_width must be > 0.')
	assert_ops += tf.image._assert(target_height > 0, ValueError,
	                    'target_height must be > 0.')
	assert_ops += tf.image._assert(target_depth > 0, ValueError,
	                    'target_depth must be > 0.')
	assert_ops += tf.image._assert(width >= (target_width + offset_width), ValueError,
	                    'width must be >= target + offset.')
	assert_ops += tf.image._assert(height >= (target_height + offset_height), ValueError,
	                    'height must be >= target + offset.')
	assert_ops += tf.image._assert(depth >= (target_depth + offset_depth), ValueError,
	                    'depth must be >= target + offset.')
	with tf.control_dependencies(assert_ops):
  		image = tf.identity(image)

	cropped = tf.slice(
		image,
		tf.pack([offset_height, offset_width, offset_depth]),
		tf.pack([target_height, target_width, target_depth])
	)

	cropped_shape = [None if tf.image._is_tensor(i) else i
	               for i in [target_height, target_width, target_depth]]
	cropped.set_shape(cropped_shape)

	return cropped


def read_binary(filename_queue, dimensions):
	class BinaryRecord(object):
		pass
	result = BinaryRecord()

	# Float type of elements
	float_type = np.dtype('float32')
	float_bytes = float_type.itemsize

	# Calculate label dimensions
	label_bytes = 1

	# Calculate data dimensions
	data_bytes = 29

	# Calculate image dimensions
	# result.height = 91
	# result.width = 109
	# result.depth = 91
	result.height = dimensions[0]
	result.width = dimensions[1]
	result.depth = dimensions[2]
	image_bytes = result.height * result.width * result.depth

	# Every record has label followed by image
	record_bytes = (label_bytes + data_bytes + image_bytes) * float_bytes

	# Read a record - from filename in filename_queue
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)

	# Convert from a string to a vector of float32 that is of record_bytes long
	record_bytes = tf.decode_raw(value, tf.float32)

	# Extract label
	result.label = tf.slice(record_bytes, [0], [label_bytes])

	# Extract data
	data_raw = tf.slice(record_bytes, [label_bytes], [data_bytes])
	result.data = tf.reshape(data_raw, [data_bytes, 1])

	# Extract image
	image_raw = tf.slice(record_bytes, [label_bytes + data_bytes], [result.height * result.width * result.depth])
	# result.image = tf.reshape(image_raw, [result.height, result.width, result.depth, 1])
	result.image = tf.reshape(image_raw, [result.height, result.width, result.depth])

	return result

def _generate_image_and_label_batch(image, data, label, min_queue_examples, batch_size, shuffle):
	# Create queue that shuffles examples and reads 'batch_size' images/labels from queue
	train_preprocess_threads = 16
	test_preprocess_threads = 1         # TODO: subject to change- explain

	if shuffle:
		images, data, label_batch = tf.train.shuffle_batch(
			[image, data, label],
			batch_size=batch_size,
			num_threads=train_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size, #TODO: subject to change
			min_after_dequeue=min_queue_examples,
			seed=273
		)
	else:
		images, data, label_batch = tf.train.batch(
			[image, data, label],
			batch_size=batch_size,
			num_threads=train_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size #TODO: subject to change
		)

	return images, data, tf.reshape(label_batch, [batch_size])


def distorted_inputs(train, data_list, batch_size, dimensions):
	with open(data_list, 'r') as data_file:
		filenames = json.load(data_file)

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read
	if train:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
		filename_queue = tf.train.string_input_producer(filenames, shuffle=True, seed=273, num_epochs = 10)
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

	# Read examples from files in filename queue
	read_input = read_binary(filename_queue, dimensions)

	# # Crop input
	# cropped_dimensions = [(d - CROP) for d in dimensions]
	# distorted_input = tf.random_crop(read_input.image, cropped_dimensions)

	# # Randomly flip image
	# distorted_input = tf.image.random_flip_left_right(read_input.image)
	# # distorted_input = tf.image.random_flip_left_right(distorted_input)
	# distorted_input = tf.image.random_flip_up_down(distorted_input)
	# distorted_input = random_flip_fwd_back(distorted_input)

	# # Distort Brightness and Contrast - make some condition to change ordering of distortions
	# distorted_input = tf.image.random_brightness(distorted_input, max_delta=0.25)
	# distorted_input = tf.image.random_contrast(distorted_input, lower=0.2, upper=0.8)
	#
	# # Add Gaussian noise
	# distorted_input = gaussian_noise(distorted_input)

	# # Normalize - zero mean, unit variance
 	# norm_input = tf.image.per_image_whitening(read_input.image)
 	# norm_input = tf.image.per_image_whitening(distorted_input)

	correct_shape_input = tf.reshape(read_input.image, dimensions + [1])
	# correct_shape_input = tf.reshape(norm_input, dimensions + [1])
	# correct_shape_input = tf.reshape(norm_input, cropped_dimensions + [1])

	# Ensure that random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
				min_fraction_of_examples_in_queue)

	# Generate a batch of images and label by building up a queue of examples
	return _generate_image_and_label_batch(correct_shape_input, read_input.data,
					read_input.label, min_queue_examples, batch_size, train)


def inputs(train, data_list, batch_size, dimensions):
	with open(data_list, 'r') as data_file:
		filenames = json.load(data_file)

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read
	if train:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
		filename_queue = tf.train.string_input_producer(filenames, shuffle=True, seed=273, num_epochs = 10)
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

	# Read examples from files in filename queue
	read_input = read_binary(filename_queue, dimensions)

	# # Resize to center of image
	# offset = CROP / 2
	# cropped_dimensions = [(d - CROP) for d in dimensions]
	# resized_input = crop_3d(
	# 	read_input.image,
	# 	offset,
	# 	offset,
	# 	offset,
	# 	cropped_dimensions[0],
	# 	cropped_dimensions[1],
	# 	cropped_dimensions[2]
	# )
	# correct_shape_input = tf.reshape(resized_input, cropped_dimensions + [1])

	correct_shape_input = tf.reshape(read_input.image, dimensions + [1])

	# Ensure that random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
				min_fraction_of_examples_in_queue)

	# Generate a batch of images and label by building up a queue of examples
	return _generate_image_and_label_batch(correct_shape_input, read_input.data,
					read_input.label, min_queue_examples, batch_size, train)
	# return _generate_image_and_label_batch(read_input.image, read_input.data,
	# 				read_input.label, min_queue_examples, batch_size, train)
