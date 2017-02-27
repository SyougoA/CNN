# coding: utf-8
import cv2
import tensorflow as tf
import numpy as np
import os

NUM_CLASSES = 10 #分類に使うクラス
IMG_SIZE = 28
COLOR_CHANNELS = 3
IMG_PIXELS = IMG_SIZE * IMG_SIZE * COLOR_CHANNELS
# 学習ステップ数
STEPS = 100
BATCH_SIZE = 20

# tensorflow特有のinference, loss, training, accuracy構造
def inference(images_placeholder, keep_prob):
	def weight_variable(shape):
		initial = tf.truncated_normal(shape, stddev=0.1)
		return tf.Variable(initial)

	def bias_variable(shape):
		initial = tf.constant(0.1, shape=shape)
		return tf.Variable(initial)

	def conv2d(x, w):
		return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
		return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
			strides=[1, 2, 2, 1], padding='SAME')

	x_image = tf.reshape(images_placeholder, [-1, IMG_SIZE, IMG_SIZE, COLOR_CHANNELS])

	with tf.name_scope('conv1') as scope:
		w_conv1 = weight_variable([5, 5, COLOR_CHANNELS, 32])
		b_conv1 = bias_variable([32])
		h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_conv1)

	with tf.name_scope('pool1') as scope:
		h_pool1 = max_pool_2x2(h_conv1)

	with tf.name_scope('conv2') as scope:
		w_conv2 = weight_variable([5, 5, 32, 64])
		b_conv2 = bias_variable([64])
		h_conv2 = tf.nn.relu(conv2d(h_pool1, w_conv2) + b_conv2)

	with tf.name_scope('pool2') as scope:
		h_pool2 = max_pool_2x2(h_conv2)

	with tf.name_scope('fc1') as scope:
		w_fc1 = weight_variable([7 * 7 * 64, 1024])
		b_fc1 = bias_variable([1024])
		h_pool2_flat = tf.reshape(h_pool2, [-1, 7 * 7 * 64])
		h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, w_fc1) + b_fc1)
		h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	with tf.name_scope('fc2') as scope:
		w_fc2 = weight_variable([1024, NUM_CLASSES])
		b_fc2 = bias_variable([NUM_CLASSES])

	with tf.name_scope('softmax') as scope:
		y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, w_fc2) + b_fc2)

	return y_conv


def loss(logits, labels):
	cross_entropy = -tf.reduce_sum(labels * tf.log(logits))
	tf.summary.scalar("cross_entropy", cross_entropy)
	return cross_entropy


def training(loss, learning_rate):
	train_step = tf.train.AdamOptimizer(learning_rate).minimize(loss)
	return train_step


def accuracy(logits, labels):
	correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	tf.summary.scalar("accuracy", accuracy)
	return accuracy

if __name__ == "__main__":
	train_dirs = [""] #分類に使うディレクトリ名
	train_image = []
	train_label = []


	for i, d in enumerate(train_dirs):
		# Mac環境だとDS_Storeとぶつかるのでそのため
		files = [x for x in os.listdir("train_data/" + d) if x != ".DS_Store"]
		for f in files:
			try:
				img = cv2.imread("train_data/" + d + '/' + f)
				img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
				img = img.flatten().astype(np.float32) / 255.0
				train_image.append(img)
			except:
				print("error:" "\"train_data/" + d + '/' + f + "\"")

			tmp = np.zeros(NUM_CLASSES)
			tmp[i] = 1
			train_label.append(tmp)

	train_image = np.asarray(train_image)
	train_label = np.asarray(train_label)

	test_image = []
	test_label = []

	for i, d in enumerate(train_dirs):
		# Mac環境だとDS_Storeとぶつかるのでそのため
		files = [x for x in os.listdir("test_data/" + d) if x != ".DS_Store"]
		for f in files:
			try:
				img = cv2.imread("test_data/" + d + '/' + f)
				img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
				img = img.flatten().astype(np.float32) / 255.0
				test_image.append(img)
			except:
				print("error:" "\"test_data/" + d + '/' + f + "\"")

			tmp = np.zeros(NUM_CLASSES)
			tmp[i] = 1
			test_label.append(tmp)

	test_image = np.asarray(test_image)
	test_label = np.asarray(test_label)

	with tf.Graph().as_default():
		images_placeholder = tf.placeholder("float", shape=(None, IMG_PIXELS))
		labels_placeholder = tf.placeholder("float", shape=(None, NUM_CLASSES))
		keep_prob = tf.placeholder("float")

		logits = inference(images_placeholder, keep_prob)
		loss_value = loss(logits, labels_placeholder)
		train_op = training(loss_value, 1e-4)
		acc = accuracy(logits, labels_placeholder)

		saver = tf.train.Saver()
		sess = tf.Session()
		sess.run(tf.global_variables_initializer())
		summary_op = tf.summary.merge_all()
		summary_writer = tf.summary.FileWriter("summary", sess.graph_def)

		for step in range(STEPS):
			for i in range(int(len(train_image) / BATCH_SIZE)):
				batch = BATCH_SIZE * i
				sess.run(train_op, feed_dict={
					images_placeholder: train_image[batch:batch + BATCH_SIZE],
					labels_placeholder: train_label[batch:batch + BATCH_SIZE],
					keep_prob: 0.8})

			train_accuracy = sess.run(acc, feed_dict={
				images_placeholder: train_image,
				labels_placeholder: train_label,
				keep_prob: 1.0})
			print("step %d, training accuracy %g" % (step, train_accuracy))

			summary_str = sess.run(summary_op, feed_dict={
				images_placeholder: train_image,
				labels_placeholder: train_label,
				keep_prob: 1.0})
			summary_writer.add_summary(summary_str, step)

	print("test accuracy %g" % sess.run(acc, feed_dict={
		images_placeholder: test_image,
		labels_placeholder: test_label,
		keep_prob: 1.0}))

	save_path = saver.save(sess, "model.ckpt")# モデルを保存