import pandas as pd
import numpy as np
import tensorflow as tf

#数据预处理
#1 加载数据集，输入与结果分开
train=pd.read_csv("C:/Users/hp/Desktop/train.csv")
images=train.iloc[:,1:].values
labels_flat=train['label'].values.ravel()

#2 对输入进行处理
images=images.astype(np.float)
images=np.multiply(images, 1.0 / 255.0)
print('输入数据的数量：(%g,%g)'% images.shape)
image_size=images.shape[1]
print('输入数据的维度=> {0}'.format(image_size))
image_width=image_height=np.ceil(np.sqrt(image_size)).astype(np.uint8)
print('图片的长=> {0}\n 图片的高=> {1}'.format(image_width,image_height))
x=tf.placeholder('float',shape=[None,image_size])

#3 对结果进行处理
labels_count=np.unique(labels_flat).shape[0]
print('结果的种类 =>{0}'.format(labels_count))
def dense_to_one_hot(labels_dense, num_classes):
    num_labels=labels_dense.shape[0]
    index_offset=np.arange(num_labels)*num_classes
    labels_one_hot=np.zeros((num_labels,num_classes))
    labels_one_hot.flat[index_offset+labels_dense.ravel()]=1
    return labels_one_hot
labels=dense_to_one_hot(labels_flat,labels_count)
labels=labels.astype(np.uint8)
y=tf.placeholder('float',shape=[None,labels_count])

print('结果的数量：({0[0]},{0[1]})'.format(labels.shape))

#4 把数据分成训练集和验证集 40000/2000
VALIDATION_SIZE=2000
validation_images=images[:VALIDATION_SIZE]
validation_labels=labels[:VALIDATION_SIZE]
train_images=images[VALIDATION_SIZE:]
train_labels=labels[VALIDATION_SIZE:]

#5 训练集分批
batch_size=100
n_batch=len(train_images)//batch_size

#6 创建一个简单的神经网络
weights=tf.Variable(tf.zeros([784,10]))
biases=tf.Variable(tf.zeros([10]))
result=tf.matmul(x,weights)+biases
prediction=tf.nn.softmax(result)

#7 创建损失函数，这里以交叉熵平均值为衡量
loss=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y,logits=prediction))

#8 梯度下降法优化参数
train_step=tf.train.GradientDescentOptimizer(0.1).minimize(loss)

#9 初始化变量
init=tf.global_variables_initializer()

#10 计算准确度
correct_prediction=tf.equal(tf.argmax(y,1),tf.argmax(prediction,1))
accuracy=tf.reduce_mean(tf.cast(correct_prediction,tf.float32))

with tf.Session() as sess:
    sess.run(init)
    for epoch in range(50):
        for batch in range(n_batch):
            batch_x=train_images[batch*batch_size:(batch+1)*batch_size]
            batch_y=train_labels[batch*batch_size:(batch+1)*batch_size]
            sess.run(train_step,feed_dict={x:batch_x,y:batch_y})
        accuracy_n=sess.run(accuracy,feed_dict={x:validation_images, y:validation_labels})
        print("第"+ str(epoch+1)+"轮，精确度为："+str(accuracy_n))
