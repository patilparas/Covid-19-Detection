import tensorflow as tf
import cv2
import numpy as np
import os
import shutil


_INFECTED_IMGSDIR = r"data\covid-chestxray-dataset\images\Infected"
_NORMAL_IMGSDIR = r"data\covid-chestxray-dataset\images\Normal"
_OUTPUT_DIR = r"data\covid-chestxray-dataset\output"


def _bytes_feature(value):
	return tf.train.Feature(
		bytes_list=tf.train.BytesList(value=value))


def _int64_feature(value):
	return tf.train.Feature(
		int64_list=tf.train.Int64List(value=value))


def write_records(imgs_dir, label):
	output_path = os.path.join(
		os.path.abspath(_OUTPUT_DIR),
		os.path.basename(imgs_dir))
	if os.path.exists(output_path):
		shutil.rmtree(output_path)
	os.makedirs(output_path)

	for root, _, images in os.walk(imgs_dir):
		for image in images:
			imgpath = os.path.join(root, image)
			tfrec_path = os.path.join(
				output_path,
				os.path.splitext(image)[0] + ".tfrecord")
			writer = tf.io.TFRecordWriter(tfrec_path)
			img = open(imgpath, "rb").read()
			features = dict(label=_int64_feature([label]), image=_bytes_feature([img]))
			example = tf.train.Example(
				features=tf.train.Features(feature=features))

			writer.write(example.SerializeToString())

	print("Records written at ", output_path)


_INFECTED_IMGSDIR = os.path.abspath(_INFECTED_IMGSDIR)
_NORMAL_IMGSDIR = os.path.abspath(_NORMAL_IMGSDIR)
write_records(_INFECTED_IMGSDIR, 1)
write_records(_NORMAL_IMGSDIR, 0)


ds = tf.data.TFRecordDataset("test.tfrecord")
desc = dict(label=tf.io.FixedLenFeature([], tf.int64),
			image=tf.io.FixedLenFeature([], tf.string))
