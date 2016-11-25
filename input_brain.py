import tensorflow as tf
import numpy as np
import json

# Train and evaliation epoch definition
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 1000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 200

def read_binary(filename_queue):
	class BinaryRecord(object):
		pass
	result = BinaryRecord()

	# Float type of elements
	float_type = np.dtype('float32')
	float_bytes = float_type.itemsize

	# Calculate dimensions
	label_bytes = 1
	result.height = 91
	result.width = 109
	result.depth = 91
	image_bytes = result.height * result.width * result.depth

	# Every record has label followed by image
	record_bytes = (label_bytes + image_bytes) * float_bytes

	# Read a record - from filename in filename_queue
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue)
	print result.key

	# Convert from a string to a vector of float32 that is of record_bytes long
	record_bytes = tf.decode_raw(value, tf.float32)

	# Extract label
	result.label = tf.slice(record_bytes, [0], [label_bytes])

	# Extract image
	image_raw = tf.slice(record_bytes, [label_bytes], [result.height * result.width * result.depth])
	result.image = tf.reshape(image_raw, [result.height, result.width, result.depth, 1])

	return result

def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	# Create queue that shuffles examples and reads 'batch_size' images/labels from queue
	train_preprocess_threads = 1
	test_preprocess_threads = 1         # TODO: subject to change- explain

	if shuffle:
		images, label_batch = tf.train.shuffle_batch(
			[image, label],
			batch_size=batch_size,
			num_threads=train_preprocess_threads,
			capacity=min_queue_examples + 3 * batch_size, #TODO: subject to change
			min_after_dequeue=min_queue_examples
		)
	else:
                images, label_batch = tf.train.batch(
                        [image, label],
                        batch_size=batch_size,
                        num_threads=train_preprocess_threads,
                        capacity=min_queue_examples + 3 * batch_size #TODO: subject to change
                )

	return images, tf.reshape(label_batch, [batch_size])


################################### TODO ######################################

# def distorted_inputs(data_dir, batch_size):

# Modification of the fuction that can feed in distorted inputs
# Absolutely not necessary, but a potential option

###############################################################################


def inputs(train, data_list, batch_size):
	with open(data_list, 'r') as data_file:
		filenames = json.load(data_file)

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read
	if train:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
		filename_queue = tf.train.string_input_producer(filenames, shuffle=True, num_epochs = 10)
	else:
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL
        filename_queue = tf.train.string_input_producer(filenames, shuffle=False)

	# Read examples from files in filename queue
	read_input = read_binary(filename_queue)

	# Ensure that random shuffling has good mixing properties
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch *
				min_fraction_of_examples_in_queue)

	# Generate a batch of images and label by building up a queue of examples
	return _generate_image_and_label_batch(read_input.image, read_input.label,
					min_queue_examples, batch_size, train)
