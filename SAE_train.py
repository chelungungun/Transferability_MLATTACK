from keras.layers import Input , Dense , Lambda , Concatenate
from keras import Model , regularizers
import numpy as np
from keras import optimizers
from data import load_data
from sklearn.metrics import f1_score
from termcolor import colored
import tensorflow as tf
from scipy.sparse import csr_matrix
import keras.backend as K
from keras.callbacks import EarlyStopping
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

glo_seed = 2020
tf.random.set_seed(glo_seed)

def extract_samples(size: object , X: object , Y: object , indecies: object) -> object:
	if size == "full":
		return X[indecies , :] , Y[indecies , :]
	else:
		return X[indecies[:1000] , :] , Y[indecies[:1000] , :]


def binary_label(pred_pro):
	predY = np.zeros([np.shape(pred_pro)[0] , np.shape(pred_pro)[1]] , dtype=np.int32)
	for i in range(np.shape(pred_pro)[0]):
		predY[i][pred_pro[i] >= 0.5] = 1
		predY[i][pred_pro[i] < 0.5] = 0
	return predY


def binary_label_svm(pred_pro):
	predY = np.zeros([np.shape(pred_pro)[0] , np.shape(pred_pro)[1]] , dtype=np.int32)
	for i in range(np.shape(pred_pro)[0]):
		predY[i][pred_pro[i] >= 0] = 1
		predY[i][pred_pro[i] < 0] = -1
	return predY


def untarget_grads(x, ite_num):
	# x: batch_size * num_label * grad_dim = n * m * d
	batch_size = tf.shape(x)[0]
	num_label = K.shape(x)[1]
	dim = K.shape(x)[2]
	up_x = x
	remain_record = x
	select_record = []

	for i in range(ite_num):
		x_sum = K.sqrt(K.sum(K.square(up_x), axis=-1))  #n * m
		[values , indices] = tf.math.top_k(x_sum)
		cat_idx = tf.stack([tf.range(0 , batch_size) , K.squeeze(indices, axis=-1)] , axis=1)
		x_max_gather = K.expand_dims(tf.gather_nd(up_x, cat_idx), axis=1)
		select_record.append(K.expand_dims(tf.gather_nd(remain_record, cat_idx), axis=1))
		cp_xmax = K.tile(x_max_gather, (1, num_label, 1))

		[values_neg , indices_neg] = tf.math.top_k(-1 * x_sum, k=num_label - 1)
		cp_ind = K.tile(K.expand_dims(tf.range(0 , batch_size), axis=1), (1, num_label-1))
		gather_neg_id = K.reshape(K.concatenate([K.expand_dims(cp_ind, axis=2), K.expand_dims(indices_neg, axis=2)], axis=2), shape=(batch_size * (num_label-1), 2))
		x_neg_gather = K.reshape(tf.gather_nd(remain_record, gather_neg_id), shape=(batch_size, num_label-1, dim))

		zo = tf.zeros(shape=(batch_size, 1, dim))
		remain_record = K.concatenate([x_neg_gather, zo], axis=1)
		up_x = cp_xmax + remain_record

	x_sumo = K.sqrt(K.sum(K.square(up_x) , axis=-1))  # n * m
	[valueso , indiceso] = tf.math.top_k(x_sumo)
	cat_idxo = tf.stack([tf.range(0 , batch_size) , K.squeeze(indiceso , axis=-1)] , axis=1)
	x_max_gathero = tf.gather_nd(up_x , cat_idxo)
	select_record.append(K.expand_dims(tf.gather_nd(remain_record , cat_idxo) , axis=1))
	select_record_tensor = K.concatenate(select_record, axis=1)
	select_record_sepsum = K.sum(K.sqrt(K.sum(K.square(select_record_tensor), axis=-1)), axis=1, keepdims=True) # batch * 1
	batch_max = K.expand_dims(K.sqrt(K.sum(K.square(x_max_gathero), axis=1, keepdims=True)), axis=1) # batch * 1 * 1

	return [batch_max, select_record_sepsum, select_record_tensor]


def regul(x):
	batch_size = K.shape(x)[0]
	loss = K.sum(K.sqrt(K.sum(K.square(K.sum(x, axis=1)), axis=1))) / K.cast(batch_size,tf.float32)

	return loss


def one_grad(x,i):
	# x[0]: outputs; x[1]:inputs; x[2]:labels
	one_out = K.expand_dims(x[0][:,i], axis=1)
	grad = tf.gradients(one_out , x[1])[0]
	grad_dim = K.shape(grad)[1]
	cp_one_out = K.tile(one_out, (1,grad_dim))
	one_label = K.expand_dims(x[2][:,i], axis=1)
	cp_one_label = K.tile(one_label, (1,grad_dim))

	#The method to calculate normalized gradients
	grad_nl = -1 * grad * cp_one_label / K.log(K.exp(0.01 ) + K.exp(K.exp(cp_one_label * cp_one_out)))
	one_grad = K.expand_dims(grad_nl , axis=(1))

	return one_grad


def svm(inputshape_x , inputshape_y, labelshape , target, label_ind, loss_lambda, ite_num):
	inputs_x = Input(shape=(inputshape_x ,))
	inputs_y = Input(shape=(inputshape_y ,))
	outputs = Dense(labelshape , activation='linear' , kernel_regularizer=regularizers.l1_l2(l1=0.000000 , l2=0.0000005))(
		inputs_x)
	grads_list = []
	fun_one_grad = Lambda(one_grad)
	for m in range(labelshape):
		fun_one_grad.arguments = {'i': m}
		grad = fun_one_grad([outputs, inputs_x, inputs_y])
		grads_list.append(grad)
	grads = Concatenate(1)(grads_list)
	fun_rel = Lambda(regul)
	fun_unta_grads = Lambda(untarget_grads)
	fun_unta_grads.arguments = {'ite_num': ite_num}
	[untarget_norml_grads , select_record_sepsum , select_record_tensor] = fun_unta_grads(grads)
	loss = fun_rel(untarget_norml_grads) * loss_lambda

	model = Model(inputs = [inputs_x, inputs_y] , outputs = outputs )
	model.add_loss(loss)

	return model


def fit(dataset , folds , classifier):
	unlabeled_idx , cv_splits , X , Y = load_data(dataset , folds=folds , rng=glo_seed)

	if classifier == 'svm':
		Y = Y.toarray() * 2 - 1
		Y = csr_matrix(Y)

	num_of_label = Y.shape[1]
	num_of_sample = X.shape[0]
	num_of_feature = X.shape[1]
	n_epochs = 500
	learning_rate = 0.01
	batch_size = 32

	loss_lambda = 0.05
	label_ind = [0 , 1]
	ite_num = num_of_label
	method = 'SAE'

	for j in range(1):
		# j=2
		training_samples = []
		testing_samples = []
		for k in range(len(cv_splits)):
			if k != j:
				training_samples += cv_splits[k]
			else:
				unlabeled_idx = np.hstack((unlabeled_idx , cv_splits[k]))
				testing_samples += cv_splits[k]
		x_train_fe , y_train = extract_samples("full" , X , Y , training_samples)
		x_test , y_test = extract_samples("full" , X , Y , testing_samples)
		x_train_fe , y_train , x_test_fe , y_test = x_train_fe.toarray() , y_train.toarray() , x_test.toarray() , y_test.toarray()
		x_train = x_train_fe
		x_test = x_test_fe

		if classifier == 'svm':
			model = svm(inputshape_x=num_of_feature , inputshape_y= num_of_label, labelshape=num_of_label,
			            label_ind=label_ind , loss_lambda= loss_lambda , ite_num=ite_num)

		stg_optimizer = optimizers.Adam(lr=learning_rate)
		optimizer = stg_optimizer
		if classifier == 'svm':
			model.compile(optimizer=optimizer , loss="hinge", metrics=[tf.keras.metrics.Hinge()])
		earlystop_cb = EarlyStopping(monitor='val_loss' , patience=50 , verbose=1 , mode='auto')
		callback = [earlystop_cb]
		model.fit([x_train,y_train] , y_train , epochs=n_epochs , batch_size=batch_size , shuffle=True , validation_split=0.3 ,
		          callbacks=callback)
		model.save_weights('model_trans/' + dataset + method + 'lambda' + str(loss_lambda) + '.h5')

		pred_pro = model.predict([x_test, y_test])
		if classifier == 'svm':
			predY = binary_label_svm(pred_pro=pred_pro)
		else:
			predY = binary_label(pred_pro=pred_pro)
		if classifier == 'svm':
			y_test = (y_test + 1)/2
			predY = (predY + 1)/2

		micro_f1 = f1_score(y_test , predY , average='micro')
		macro_f1 = f1_score(y_test , predY , average='macro')
		print(colored("Micro Score () --> F1 == {0}\n" , 'red').format(micro_f1))
		print(colored("Macro Score () --> F1 == {0}\n" , 'red').format(macro_f1))


if __name__ == '__main__':
	dataset = 'creepware'
	classifier = 'svm'
	folds = 5
	fit(dataset=dataset , folds=folds , classifier=classifier)
